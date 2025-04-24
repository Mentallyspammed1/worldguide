# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Version 1.4.0: Completed Signal Generator logic, added main trading loop,
#               integrated position management (BE/TSL) logic within the loop,
#               added SL/TP calculation helper, refined signal return structure,
#               improved overall execution flow and error handling.

"""
Pyrmethus Volumatic Bot: A Python Trading Bot for Bybit V5 (v1.4.0)

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
    - Returns structured signal result including calculated initial SL/TP for entries.
-   Calculates position size based on risk percentage, stop-loss distance, and market constraints (precision, limits).
-   Sets leverage for contract markets (optional).
-   Places market orders to enter and exit positions.
-   Advanced Position Management:
    -   Sets initial Stop Loss (SL) and Take Profit (TP) based on ATR multiples.
    -   Implements Trailing Stop Loss (TSL) activation via API based on profit percentage and callback rate.
    -   Implements Break-Even (BE) stop adjustment based on ATR profit targets.
    -   **NOTE:** Break-Even (BE) and Trailing Stop Loss (TSL) activation states are managed
        **in-memory per cycle** and are **not persistent** across bot restarts. If the bot restarts,
        it relies on the exchange's reported SL/TP/TSL values for management.
-   Robust API interaction with configurable retries, detailed error handling (Network, Rate Limit, Auth, Exchange-specific codes), and validation.
-   Secure handling of API credentials via `.env` file.
-   Flexible configuration via `config.json` with validation, default values, and auto-update of missing/invalid fields.
-   Detailed logging with a Neon color scheme for console output and rotating file logs (UTC timestamps).
-   Sensitive data (API keys/secrets) redaction in logs.
-   Graceful shutdown handling (Ctrl+C, SIGTERM).
-   Sequential multi-symbol trading capability.
-   Structured code using classes for Strategy Calculation (`VolumaticOBStrategy`) and Signal Generation (`SignalGenerator`).
-   Includes main trading loop orchestrating data fetching, analysis, signal generation, position management, and order execution.
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
BOT_VERSION = "1.4.0" # Current bot version

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

class SignalResult(TypedDict):
    """Structured result from the SignalGenerator."""
    signal: str                 # The generated signal: "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD"
    reason: str                 # A brief explanation for the generated signal
    initial_sl: Optional[Decimal] # Calculated initial stop loss price (for BUY/SELL signals)
    initial_tp: Optional[Decimal] # Calculated initial take profit price (for BUY/SELL signals)

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
        key = API_KEY
        secret = API_SECRET
        msg = super().format(record)
        try:
            if key and isinstance(key, str) and key in msg:
                msg = msg.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and secret in msg:
                msg = msg.replace(secret, self._api_secret_placeholder)
        except Exception as e:
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

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Set the base level to capture all messages; handlers filter output

    # --- File Handler (DEBUG level, Rotating, Redaction, UTC Timestamps) ---
    try:
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S' # Standard ISO-like date format for files
        )
        file_formatter.converter = time.gmtime # type: ignore
        fh.setFormatter(file_formatter)
        fh.setLevel(logging.DEBUG) # Log everything from DEBUG upwards to the file
        logger.addHandler(fh)
    except Exception as e:
        print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")

    # --- Console Handler (Configurable Level, Neon Colors, Local Timezone Timestamps) ---
    try:
        sh = logging.StreamHandler(sys.stdout) # Explicitly use stdout for console output
        level_colors = {
            logging.DEBUG: NEON_CYAN + DIM,      # Dim Cyan for Debug
            logging.INFO: NEON_BLUE,             # Bright Cyan (or Blue) for Info
            logging.WARNING: NEON_YELLOW,        # Bright Yellow for Warning
            logging.ERROR: NEON_RED,             # Bright Red for Error
            logging.CRITICAL: NEON_RED + BRIGHT, # Bright Red + Bold for Critical
        }

        class NeonConsoleFormatter(SensitiveFormatter):
            _level_colors = level_colors
            _tz = TIMEZONE # Use the globally configured timezone object

            def format(self, record: logging.LogRecord) -> str:
                level_color = self._level_colors.get(record.levelno, NEON_BLUE) # Default to Info color if level unknown
                log_fmt = (
                    f"{NEON_BLUE}%(asctime)s{RESET} - " # Timestamp color
                    f"{level_color}%(levelname)-8s{RESET} - " # Level color
                    f"{NEON_PURPLE}[%(name)s]{RESET} - " # Logger name color
                    f"%(message)s" # Message (will be colored by context)
                )
                formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
                formatter.converter = lambda *args: datetime.now(self._tz).timetuple() # type: ignore
                return super(NeonConsoleFormatter, self).format(record)

        sh.setFormatter(NeonConsoleFormatter())
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

    logger.propagate = False
    return logger

# Initialize the 'init' logger early for messages during startup and configuration loading
init_logger = setup_logger("init")
init_logger.info(f"{Fore.MAGENTA}{BRIGHT}Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing...{Style.RESET_ALL}")
init_logger.info(f"Using Timezone: {TIMEZONE_STR}")

# --- Utility Functions ---
def quantize_price(price: Decimal, price_tick: Decimal, side: str, is_tp: bool = False) -> Decimal:
    """
    Quantizes a price to the nearest valid tick size, rounding appropriately based on side and whether it's TP or SL.

    - SL Long: Round DOWN
    - SL Short: Round UP
    - TP Long: Round DOWN (conservative)
    - TP Short: Round UP (conservative)

    Args:
        price (Decimal): The raw price to quantize.
        price_tick (Decimal): The minimum price movement (step size) for the market. Must be positive.
        side (str): The side of the potential order ('long' or 'short').
        is_tp (bool): True if the price is a Take Profit level, False if it's a Stop Loss level.

    Returns:
        Decimal: The quantized price, adjusted to the nearest valid tick. Returns input price if tick is invalid.
    """
    if not isinstance(price, Decimal) or not price.is_finite() or \
       not isinstance(price_tick, Decimal) or not price_tick.is_finite() or price_tick <= 0:
        # init_logger.warning(f"Invalid input for quantize_price: price={price}, tick={price_tick}") # Potentially verbose
        return price # Return original price if inputs are invalid

    # Determine rounding direction
    rounding_mode = ROUND_DOWN # Default rounding
    if side == 'long':
        rounding_mode = ROUND_DOWN # SL Long: Down, TP Long: Down
    elif side == 'short':
        rounding_mode = ROUND_UP # SL Short: Up, TP Short: Up

    # Apply quantization: floor/ceil(price / tick) * tick
    quantized_steps = (price / price_tick).quantize(Decimal('1'), rounding_mode)
    quantized_price = quantized_steps * price_tick

    # Ensure quantized price is still positive
    return max(quantized_price, price_tick) # Return at least one tick if rounding results in zero/negative

def get_logger_for_symbol(symbol: str) -> logging.Logger:
    """Retrieves the existing logger instance for a given symbol or initializes it."""
    safe_name = symbol.replace('/', '_').replace(':', '-')
    logger_name = f"pyrmethus_bot_{safe_name}"
    # Check if logger already exists, otherwise setup_logger will create it
    if logger_name in logging.Logger.manager.loggerDict:
        return logging.getLogger(logger_name)
    else:
        # If not found (shouldn't happen after initial setup), create it
        init_logger.warning(f"Logger for {symbol} not found, re-initializing...")
        return setup_logger(symbol)

# --- Configuration Loading & Validation (Function Definitions as before) ---
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
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Created default configuration file: {filepath}{RESET}")
            global QUOTE_CURRENCY
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config # Return the defaults immediately
        except IOError as e:
            init_logger.critical(f"{NEON_RED}FATAL: Error creating default config file '{filepath}': {e}. Cannot proceed.{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config # Use internal defaults

    # --- File Loading ---
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        if not isinstance(loaded_config, dict):
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
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        return default_config # Fallback to internal defaults

    # --- Validation and Merging ---
    try:
        # Ensure all keys from default_config exist in loaded_config
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys:
            config_needs_saving = True # Mark for saving if keys were added

        # --- Type and Range Validation Helper ---
        def validate_numeric(cfg: Dict, key_path: str, min_val: Union[int, float, Decimal], max_val: Union[int, float, Decimal],
                             is_strict_min: bool = False, is_int: bool = False, allow_zero: bool = False) -> bool:
            nonlocal config_needs_saving
            keys = key_path.split('.')
            current_level = cfg
            default_level = default_config
            try:
                for key in keys[:-1]:
                    current_level = current_level[key]
                    default_level = default_level[key]
                leaf_key = keys[-1]
                original_val = current_level.get(leaf_key)
                default_val = default_level.get(leaf_key)
            except (KeyError, TypeError):
                init_logger.error(f"Config validation error: Invalid key path '{key_path}'. Cannot validate.")
                return False

            if original_val is None:
                init_logger.warning(f"Config validation: Value missing at '{key_path}'. Using default: {repr(default_val)}")
                current_level[leaf_key] = default_val
                config_needs_saving = True
                return True

            corrected = False
            final_val = original_val

            try:
                num_val = Decimal(str(original_val))
                min_dec = Decimal(str(min_val))
                max_dec = Decimal(str(max_val))

                min_check_passed = num_val > min_dec if is_strict_min else num_val >= min_dec
                range_check_passed = min_check_passed and num_val <= max_dec
                zero_is_valid = allow_zero and num_val.is_zero()

                if not range_check_passed and not zero_is_valid:
                    raise ValueError("Value outside allowed range.")

                target_type = int if is_int else float
                converted_val = target_type(num_val)

                needs_correction = False
                if isinstance(original_val, bool): raise TypeError("Boolean value found where numeric value expected.")
                elif is_int and not isinstance(original_val, int): needs_correction = True
                elif not is_int and not isinstance(original_val, float):
                    if isinstance(original_val, int): converted_val = float(original_val); needs_correction = True
                    else: needs_correction = True
                elif isinstance(original_val, float) and abs(original_val - converted_val) > 1e-9: needs_correction = True
                elif isinstance(original_val, int) and original_val != converted_val: needs_correction = True

                if needs_correction:
                    init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type/value for '{key_path}' from {repr(original_val)} to {repr(converted_val)}.{RESET}")
                    final_val = converted_val
                    corrected = True

            except (ValueError, InvalidOperation, TypeError) as e:
                range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
                if allow_zero: range_str += " or 0"
                init_logger.warning(f"{NEON_YELLOW}Config Validation: Invalid value '{repr(original_val)}' for '{key_path}'. Using default: {repr(default_val)}. Error: {e}. Expected: {'integer' if is_int else 'float'}, Range: {range_str}{RESET}")
                final_val = default_val
                corrected = True

            if corrected:
                current_level[leaf_key] = final_val
                config_needs_saving = True
            return corrected

        init_logger.debug("# Validating configuration parameters...")
        # --- Apply Validations ---
        if not isinstance(updated_config.get("trading_pairs"), list) or \
           not all(isinstance(s, str) and s and '/' in s for s in updated_config.get("trading_pairs", [])):
            init_logger.warning(f"Invalid 'trading_pairs'. Using default {default_config['trading_pairs']}.")
            updated_config["trading_pairs"] = default_config["trading_pairs"]; config_needs_saving = True
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.warning(f"Invalid 'interval'. Using default '{default_config['interval']}'.")
            updated_config["interval"] = default_config["interval"]; config_needs_saving = True
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True)
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "risk_per_trade", Decimal('0'), Decimal('0.5'), is_strict_min=True)
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True)
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True)
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True)
        if not isinstance(updated_config.get("quote_currency"), str) or not updated_config.get("quote_currency"):
             init_logger.warning(f"Invalid 'quote_currency'. Using default '{default_config['quote_currency']}'.")
             updated_config["quote_currency"] = default_config["quote_currency"]; config_needs_saving = True
        if not isinstance(updated_config.get("enable_trading"), bool):
             init_logger.warning(f"Invalid 'enable_trading'. Using default '{default_config['enable_trading']}'.")
             updated_config["enable_trading"] = default_config["enable_trading"]; config_needs_saving = True
        if not isinstance(updated_config.get("use_sandbox"), bool):
             init_logger.warning(f"Invalid 'use_sandbox'. Using default '{default_config['use_sandbox']}'.")
             updated_config["use_sandbox"] = default_config["use_sandbox"]; config_needs_saving = True
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
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1)
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1)
        if updated_config["strategy_params"].get("ob_source") not in ["Wicks", "Body"]:
             init_logger.warning(f"Invalid strategy_params.ob_source. Using default '{DEFAULT_OB_SOURCE}'.")
             updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE; config_needs_saving = True
        if not isinstance(updated_config["strategy_params"].get("ob_extend"), bool):
             init_logger.warning(f"Invalid strategy_params.ob_extend. Using default '{DEFAULT_OB_EXTEND}'.")
             updated_config["strategy_params"]["ob_extend"] = DEFAULT_OB_EXTEND; config_needs_saving = True
        # Protection Params
        if not isinstance(updated_config["protection"].get("enable_trailing_stop"), bool):
             init_logger.warning(f"Invalid protection.enable_trailing_stop. Using default.")
             updated_config["protection"]["enable_trailing_stop"] = default_config["protection"]["enable_trailing_stop"]; config_needs_saving = True
        if not isinstance(updated_config["protection"].get("enable_break_even"), bool):
             init_logger.warning(f"Invalid protection.enable_break_even. Using default.")
             updated_config["protection"]["enable_break_even"] = default_config["protection"]["enable_break_even"]; config_needs_saving = True
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", Decimal('0.0001'), Decimal('0.1'), is_strict_min=True)
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", Decimal('0'), Decimal('0.1'), allow_zero=True)
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", Decimal('0.1'), Decimal('10.0'))
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True)
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", Decimal('0.1'), Decimal('20.0'), is_strict_min=True)
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", Decimal('0'), Decimal('20.0'), allow_zero=True)

        # --- Save Updated Config if Necessary ---
        if config_needs_saving:
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                 init_logger.info(f"{NEON_GREEN}Configuration file '{filepath}' updated with missing/corrected values.{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error saving updated configuration to '{filepath}': {save_err}{RESET}", exc_info=True)

        # Update global QUOTE_CURRENCY
        global QUOTE_CURRENCY
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT")
        init_logger.info(f"Quote currency set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        init_logger.info(f"{Fore.CYAN}# Configuration loading and validation complete.{Style.RESET_ALL}")

        return updated_config # Return the validated and potentially corrected config

    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: Unexpected error during configuration processing: {e}. Using internal defaults.{RESET}", exc_info=True)
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT") # Ensure quote currency is set even on failure
        return default_config # Fallback to internal defaults

# --- Load Global Configuration ---
CONFIG = load_config(CONFIG_FILE)
# QUOTE_CURRENCY is updated inside load_config()

# --- CCXT Exchange Setup (Function Definitions as before) ---
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

# --- CCXT Data Fetching Helpers (Function Definitions as before) ---
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
    if value is None: return None
    try:
        s_val = str(value).strip()
        if not s_val: return None
        d_val = Decimal(s_val)
        if not allow_zero and d_val.is_zero(): return None
        if not allow_negative and d_val < Decimal('0'): return None
        return d_val
    except (InvalidOperation, TypeError, ValueError): return None

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
        price_decimal = Decimal(str(price))
        if price_decimal <= Decimal('0'): return None
        formatted_str = exchange.price_to_precision(symbol, float(price_decimal))
        if Decimal(formatted_str) > Decimal('0'): return formatted_str
        else: return None
    except (InvalidOperation, ValueError, TypeError, KeyError, AttributeError) as e:
        init_logger.warning(f"Error formatting price '{price}' for {symbol}: {e}") # Use init_logger as this might be called early
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

            def safe_decimal_from_ticker(value: Optional[Any], field_name: str) -> Optional[Decimal]:
                return _safe_market_decimal(value, f"ticker.{field_name}", allow_zero=False, allow_negative=False)

            price = safe_decimal_from_ticker(ticker.get('last'), 'last')
            if price: source = "'last' price"
            else:
                bid = safe_decimal_from_ticker(ticker.get('bid'), 'bid')
                ask = safe_decimal_from_ticker(ticker.get('ask'), 'ask')
                if bid and ask: price = (bid + ask) / Decimal('2'); source = f"mid-price (Bid:{bid.normalize()}, Ask:{ask.normalize()})"
                elif ask: price = ask; source = f"'ask' price ({ask.normalize()})"
                elif bid: price = bid; source = f"'bid' price ({bid.normalize()})"

            if price:
                normalized_price = price.normalize()
                lg.debug(f"Current price ({symbol}) obtained via {source}: {normalized_price}")
                return normalized_price
            else:
                last_exception = ValueError(f"No valid positive price found in ticker (last, mid, ask, bid). Ticker data: {ticker}")
                lg.warning(f"Could not find valid current price ({symbol}, Attempt {attempts + 1}). Ticker: {ticker}. Retrying...")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e; lg.warning(f"{NEON_YELLOW}Network error fetching price ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price ({symbol}): {e}. Pausing for {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Authentication error fetching price: {e}. Stopping price fetch.{RESET}"); return None
        except ccxt.ExchangeError as e: last_exception = e; lg.error(f"{NEON_RED}Exchange error fetching price ({symbol}): {e}{RESET}")
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error fetching price ({symbol}): {e}{RESET}", exc_info=True); return None

        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)

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

    if not hasattr(exchange, 'fetch_ohlcv') or not exchange.has.get('fetchOHLCV'):
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV. Cannot fetch klines.")
        return pd.DataFrame()

    min_required = 0
    try:
        sp = CONFIG.get('strategy_params', {})
        min_required = max(sp.get('vt_length', 0)*2, sp.get('vt_atr_period', 0), sp.get('vt_vol_ema_length', 0),
                           sp.get('ph_left', 0)+sp.get('ph_right', 0)+1, sp.get('pl_left', 0)+sp.get('pl_right', 0)+1) + 50
        lg.debug(f"Estimated minimum candles required by strategy: {min_required}")
        if limit < min_required:
            lg.warning(f"{NEON_YELLOW}Requested kline limit ({limit}) < estimated strategy requirement ({min_required}). Accuracy may be affected.{RESET}")
    except Exception as e: lg.warning(f"Could not estimate minimum required candles: {e}")

    category = 'spot'; market_id = symbol
    try:
        market = exchange.market(symbol)
        market_id = market['id']
        category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
        lg.debug(f"Using Bybit category: '{category}' and Market ID: '{market_id}' for kline fetch.")
    except Exception as e: lg.warning(f"Could not determine market category/ID for {symbol}: {e}. Using defaults.")

    all_ohlcv_data: List[List[Union[int, float, str]]] = []
    remaining_limit = limit
    end_timestamp_ms: Optional[int] = None
    max_chunks = math.ceil(limit / BYBIT_API_KLINE_LIMIT) + 2
    chunk_num = 0; total_fetched_count = 0

    while remaining_limit > 0 and chunk_num < max_chunks:
        chunk_num += 1
        fetch_size = min(remaining_limit, BYBIT_API_KLINE_LIMIT)
        lg.debug(f"Fetching kline chunk {chunk_num}/{max_chunks} ({fetch_size} candles) for {symbol}. End TS: {end_timestamp_ms}")
        attempts = 0; last_exception = None; chunk_data = None

        while attempts <= MAX_API_RETRIES:
            try:
                params = {'category': category} if 'bybit' in exchange.id.lower() else {}
                fetch_args: Dict[str, Any] = {'symbol': symbol, 'timeframe': timeframe, 'limit': fetch_size, 'params': params}
                if end_timestamp_ms: fetch_args['until'] = end_timestamp_ms

                chunk_data = exchange.fetch_ohlcv(**fetch_args)
                fetched_count_chunk = len(chunk_data) if chunk_data else 0
                lg.debug(f"API returned {fetched_count_chunk} candles for chunk {chunk_num} (requested {fetch_size}).")

                if chunk_data:
                    if chunk_num == 1: # Validate timestamp lag on first (most recent) chunk
                        try:
                            last_ts = pd.to_datetime(chunk_data[-1][0], unit='ms', utc=True)
                            interval_seconds = exchange.parse_timeframe(timeframe)
                            if interval_seconds:
                                max_allowed_lag = interval_seconds * 2.5
                                actual_lag = (pd.Timestamp.utcnow() - last_ts).total_seconds()
                                if actual_lag > max_allowed_lag:
                                    last_exception = ValueError(f"Kline data potentially stale (Lag: {actual_lag:.1f}s > Max Allowed: {max_allowed_lag:.1f}s).")
                                    lg.warning(f"{NEON_YELLOW}Timestamp lag detected ({symbol}, Chunk 1): {last_exception}. Retrying fetch...{RESET}")
                                    chunk_data = None # Discard stale data
                                else: break # Valid chunk
                            else: break # Cannot validate lag
                        except Exception as ts_err: lg.warning(f"Could not validate timestamp lag ({symbol}, Chunk 1): {ts_err}. Proceeding."); break
                    else: break # Subsequent chunks don't need lag check
                else: remaining_limit = 0; break # No more data available

            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Network error kline chunk {chunk_num} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit kline chunk {chunk_num} ({symbol}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
            except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error fetching klines: {e}. Stopping fetch.{RESET}"); return pd.DataFrame()
            except ccxt.ExchangeError as e:
                last_exception = e; lg.error(f"{NEON_RED}Exchange error kline chunk {chunk_num} ({symbol}): {e}{RESET}")
                err_str = str(e).lower()
                if "invalid timeframe" in err_str or "interval is not supported" in err_str or "symbol invalid" in err_str: lg.critical(f"{NEON_RED}Non-retryable kline error: {e}. Stopping.{RESET}"); return pd.DataFrame()
            except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error kline chunk {chunk_num} ({symbol}): {e}{RESET}", exc_info=True); return pd.DataFrame()

            attempts += 1
            if attempts <= MAX_API_RETRIES and chunk_data is None: time.sleep(RETRY_DELAY_SECONDS * attempts)

        if chunk_data:
            all_ohlcv_data = chunk_data + all_ohlcv_data
            chunk_len = len(chunk_data)
            remaining_limit -= chunk_len
            total_fetched_count += chunk_len
            end_timestamp_ms = chunk_data[0][0] - 1
            if chunk_len < fetch_size: lg.debug(f"Received fewer candles than requested. Assuming end of history."); remaining_limit = 0
        else:
            lg.error(f"{NEON_RED}Failed to fetch kline chunk {chunk_num} for {symbol} after retries. Last error: {last_exception}{RESET}")
            if not all_ohlcv_data: lg.error(f"Failed on first chunk ({symbol}). Cannot proceed."); return pd.DataFrame()
            else: lg.warning(f"Proceeding with {total_fetched_count} candles fetched before error."); break

        if remaining_limit > 0: time.sleep(0.5) # Polite delay between chunks

    if chunk_num >= max_chunks and remaining_limit > 0: lg.warning(f"Stopped fetching klines ({symbol}) at max chunks ({max_chunks}).")
    if not all_ohlcv_data: lg.error(f"No kline data fetched for {symbol} {timeframe}."); return pd.DataFrame()
    lg.info(f"Total klines fetched across requests: {total_fetched_count}")

    seen_timestamps = set(); unique_data = []
    for candle in reversed(all_ohlcv_data):
        ts = candle[0]
        if ts not in seen_timestamps: unique_data.append(candle); seen_timestamps.add(ts)
    unique_data.reverse()
    duplicates_removed = len(all_ohlcv_data) - len(unique_data)
    if duplicates_removed > 0: lg.warning(f"Removed {duplicates_removed} duplicate candles ({symbol}).")
    all_ohlcv_data = unique_data
    all_ohlcv_data.sort(key=lambda x: x[0])
    if len(all_ohlcv_data) > limit: all_ohlcv_data = all_ohlcv_data[-limit:]

    try:
        lg.debug(f"Processing {len(all_ohlcv_data)} final candles into DataFrame ({symbol})...")
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(all_ohlcv_data, columns=cols[:len(all_ohlcv_data[0])])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty: lg.error(f"DataFrame empty after timestamp conversion ({symbol})."); return pd.DataFrame()
        df.set_index('timestamp', inplace=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                df[col] = numeric_series.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
            else: lg.warning(f"Expected column '{col}' not found in fetched data ({symbol}).")

        initial_len = len(df)
        essential_price_cols = ['open', 'high', 'low', 'close']
        df.dropna(subset=essential_price_cols, inplace=True)
        df = df[df['close'] > Decimal('0')]
        if 'volume' in df.columns:
            df.dropna(subset=['volume'], inplace=True)
            df = df[df['volume'] >= Decimal('0')]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0: lg.debug(f"Dropped {rows_dropped} rows ({symbol}) during cleaning.")
        if df.empty: lg.warning(f"Kline DataFrame empty after cleaning ({symbol})."); return pd.DataFrame()
        if not df.index.is_monotonic_increasing: lg.warning(f"Kline index not monotonic ({symbol}), sorting..."); df.sort_index(inplace=True)
        if len(df) > MAX_DF_LEN: lg.debug(f"DataFrame length ({len(df)}) > max ({MAX_DF_LEN}). Trimming ({symbol})."); df = df.iloc[-MAX_DF_LEN:].copy()

        lg.info(f"{NEON_GREEN}Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}{RESET}")
        return df
    except Exception as e:
        lg.error(f"{NEON_RED}Error processing kline data into DataFrame ({symbol}): {e}{RESET}", exc_info=True)
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
    attempts = 0; last_exception = None; market_dict: Optional[Dict] = None

    while attempts <= MAX_API_RETRIES:
        try:
            if not exchange.markets or symbol not in exchange.markets:
                lg.info(f"Market details for '{symbol}' not found. Refreshing market data...")
                try: exchange.load_markets(reload=True); lg.info("Market data refreshed.")
                except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as reload_err: last_exception = reload_err; lg.warning(f"Network error refreshing markets: {reload_err}. Retry {attempts + 1}...")
                except ccxt.AuthenticationError as reload_err: last_exception = reload_err; lg.critical(f"{NEON_RED}Auth error refreshing markets: {reload_err}. Cannot proceed.{RESET}"); return None
                except Exception as reload_err: last_exception = reload_err; lg.error(f"Failed to refresh markets for {symbol}: {reload_err}")

            try: market_dict = exchange.market(symbol)
            except ccxt.BadSymbol: lg.error(f"{NEON_RED}Symbol '{symbol}' invalid on {exchange.id}.{RESET}"); return None
            except Exception as fetch_err: last_exception = fetch_err; lg.warning(f"Error retrieving market dict for '{symbol}': {fetch_err}. Retry {attempts + 1}..."); market_dict = None

            if market_dict:
                lg.debug(f"Market found for '{symbol}'. Parsing details...")
                std_market = market_dict.copy()
                std_market['is_contract'] = std_market.get('contract', False) or std_market.get('type') in ['swap', 'future', 'option']
                std_market['is_linear'] = bool(std_market.get('linear')) and std_market['is_contract']
                std_market['is_inverse'] = bool(std_market.get('inverse')) and std_market['is_contract']
                std_market['contract_type_str'] = "Linear" if std_market['is_linear'] else "Inverse" if std_market['is_inverse'] else "Spot" if std_market.get('spot') else "Unknown"

                precision_info = std_market.get('precision', {}); limits_info = std_market.get('limits', {})
                amount_limits_info = limits_info.get('amount', {}); cost_limits_info = limits_info.get('cost', {})

                std_market['amount_precision_step_decimal'] = _safe_market_decimal(precision_info.get('amount'), 'precision.amount', allow_zero=False)
                std_market['price_precision_step_decimal'] = _safe_market_decimal(precision_info.get('price'), 'precision.price', allow_zero=False)
                std_market['min_amount_decimal'] = _safe_market_decimal(amount_limits_info.get('min'), 'limits.amount.min', allow_zero=True)
                std_market['max_amount_decimal'] = _safe_market_decimal(amount_limits_info.get('max'), 'limits.amount.max', allow_zero=False)
                std_market['min_cost_decimal'] = _safe_market_decimal(cost_limits_info.get('min'), 'limits.cost.min', allow_zero=True)
                std_market['max_cost_decimal'] = _safe_market_decimal(cost_limits_info.get('max'), 'limits.cost.max', allow_zero=False)
                contract_size_val = std_market.get('contractSize', '1')
                std_market['contract_size_decimal'] = _safe_market_decimal(contract_size_val, 'contractSize', allow_zero=False) or Decimal('1')

                if std_market['amount_precision_step_decimal'] is None or std_market['price_precision_step_decimal'] is None:
                    lg.critical(f"{NEON_RED}CRITICAL VALIDATION FAILED:{RESET} Market '{symbol}' missing essential positive precision data (AmountStep/PriceStep). Cannot proceed safely.")
                    lg.error(f"  Parsed Steps: Amount={std_market['amount_precision_step_decimal']}, Price={std_market['price_precision_step_decimal']}")
                    lg.error(f"  Raw Precision Dict: {precision_info}")
                    return None

                amt_step_str = std_market['amount_precision_step_decimal'].normalize(); price_step_str = std_market['price_precision_step_decimal'].normalize()
                min_amt_str = std_market['min_amount_decimal'].normalize() if std_market['min_amount_decimal'] is not None else 'None'
                max_amt_str = std_market['max_amount_decimal'].normalize() if std_market['max_amount_decimal'] is not None else 'None'
                min_cost_str = std_market['min_cost_decimal'].normalize() if std_market['min_cost_decimal'] is not None else 'None'
                max_cost_str = std_market['max_cost_decimal'].normalize() if std_market['max_cost_decimal'] is not None else 'None'
                contract_size_str = std_market['contract_size_decimal'].normalize()
                log_msg = (f"Market Details Parsed ({symbol}): Type={std_market['contract_type_str']}, Active={std_market.get('active', 'N/A')}\n"
                           f"  Precision Steps (Amount/Price): {amt_step_str} / {price_step_str}\n"
                           f"  Limits - Amount (Min/Max): {min_amt_str} / {max_amt_str}\n"
                           f"  Limits - Cost (Min/Max): {min_cost_str} / {max_cost_str}\n"
                           f"  Contract Size: {contract_size_str}")
                lg.debug(log_msg)

                try: final_market_info: MarketInfo = std_market # type: ignore; return final_market_info
                except Exception as cast_err: lg.error(f"Internal error casting market dict to MarketInfo ({symbol}): {cast_err}"); return std_market # type: ignore
            else:
                if attempts < MAX_API_RETRIES: lg.warning(f"Symbol '{symbol}' not found or fetch failed (Attempt {attempts + 1}). Retrying...")
                else: lg.error(f"{NEON_RED}Market '{symbol}' not found on {exchange.id} after retries. Last error: {last_exception}{RESET}"); return None

        except ccxt.BadSymbol as e: lg.error(f"Symbol '{symbol}' invalid on {exchange.id}: {e}"); return None
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error retrieving market info: {e}. Stopping.{RESET}"); return None
        except ccxt.ExchangeError as e:
            last_exception = e; lg.error(f"{NEON_RED}Exchange error retrieving market info ({symbol}): {e}{RESET}")
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for ExchangeError retrieving market info ({symbol}).{RESET}"); return None
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error retrieving market info ({symbol}): {e}{RESET}", exc_info=True); return None

        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts)

    lg.error(f"{NEON_RED}Failed to retrieve market info for '{symbol}' after all attempts. Last error: {last_exception}{RESET}")
    return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available trading balance for a specific currency (e.g., USDT).

    - Handles Bybit V5 account types (UNIFIED, CONTRACT) automatically to find the relevant balance.
    - Parses various potential balance fields ('free', 'availableToWithdraw', 'availableBalance')
      to robustly find the usable balance amount.
    - Includes retry logic for network errors and rate limits.
    - Handles authentication errors critically by re-raising them.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        currency (str): The currency code to fetch the balance for (e.g., "USDT"). Case-sensitive.
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[Decimal]: The available balance as a non-negative Decimal, or None if fetching fails.

    Raises:
        ccxt.AuthenticationError: If authentication fails during the balance fetch attempt.
    """
    lg = logger
    lg.debug(f"Fetching balance for currency: {currency}...")
    attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            balance_str: Optional[str] = None; balance_source_desc: str = "N/A"; found: bool = False; balance_info: Optional[Dict] = None
            account_types_to_check = []
            if 'bybit' in exchange.id.lower(): account_types_to_check = ['UNIFIED', 'CONTRACT']
            account_types_to_check.append('') # Fallback to default

            for acc_type in account_types_to_check:
                try:
                    params = {'accountType': acc_type} if acc_type else {}
                    type_desc = f"Account Type: {acc_type}" if acc_type else "Default Account"
                    lg.debug(f"Fetching balance ({currency}, {type_desc}, Attempt {attempts + 1})...")
                    balance_info = exchange.fetch_balance(params=params)

                    if currency in balance_info and isinstance(balance_info[currency], dict) and balance_info[currency].get('free') is not None:
                        balance_str = str(balance_info[currency]['free']); balance_source_desc = f"{type_desc} ('free' field)"; found = True; break
                    elif 'info' in balance_info and isinstance(balance_info.get('info'), dict) and \
                         'result' in balance_info['info'] and isinstance(balance_info['info'].get('result'), dict) and \
                         isinstance(balance_info['info']['result'].get('list'), list):
                        for account_details in balance_info['info']['result']['list']:
                             account_type_match = (not acc_type or account_details.get('accountType') == acc_type)
                             if account_type_match and isinstance(account_details.get('coin'), list):
                                for coin_data in account_details['coin']:
                                    if coin_data.get('coin') == currency:
                                        balance_val = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                        source_field = 'availableToWithdraw' if coin_data.get('availableToWithdraw') else ('availableBalance' if coin_data.get('availableBalance') else 'walletBalance')
                                        if balance_val is not None:
                                            balance_str = str(balance_val); balance_source_desc = f"Bybit V5 ({account_details.get('accountType')} Account, Field: '{source_field}')"; found = True; break
                                if found: break
                        if found: break
                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    if acc_type and ("account type does not exist" in err_str or "invalid account type" in err_str): lg.debug(f"Account type '{acc_type}' not found. Trying next...")
                    elif acc_type: lg.debug(f"Exchange error fetching balance ({type_desc}): {e}. Trying next..."); last_exception = e
                    else: raise e
                    continue
                except Exception as e: lg.warning(f"Unexpected error fetching balance ({type_desc}): {e}. Trying next..."); last_exception = e; continue

            if found and balance_str is not None:
                balance_decimal = Decimal(balance_str)
                final_balance = max(balance_decimal, Decimal('0'))
                lg.debug(f"Parsed balance ({currency}) from {balance_source_desc}: {final_balance.normalize()}")
                return final_balance
            elif not found and balance_info is not None: raise ccxt.ExchangeError(f"Balance for currency '{currency}' not found in response.")
            elif not found and balance_info is None: raise ccxt.ExchangeError(f"Balance fetch for '{currency}' failed to return data.")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Network error fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit fetching balance ({currency}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error fetching balance: {e}. Stopping.{RESET}"); raise e
        except ccxt.ExchangeError as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Exchange error fetching balance ({currency}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error fetching balance ({currency}): {e}{RESET}", exc_info=True); return None

        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)

    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None

# --- Position & Order Management (Function Definitions as before) ---
def get_open_position(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, logger: logging.Logger) -> Optional[PositionInfo]:
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
        market_info (MarketInfo): The standardized MarketInfo for the symbol.
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[PositionInfo]: A `PositionInfo` TypedDict containing details of the open position if found,
                                otherwise None (no position or fetch failed).
    """
    lg = logger
    attempts = 0; last_exception = None
    market_id: Optional[str] = market_info.get('id')
    category: Optional[str] = market_info.get('contract_type_str', 'Spot').lower() # Default to spot if unknown

    if not market_info.get('is_contract'):
        lg.debug(f"Position check skipped for {symbol}: Spot market.")
        return None
    if not market_id or category not in ['linear', 'inverse']:
        lg.error(f"Cannot check position for {symbol}: Missing market ID or invalid category ('{category}').")
        return None
    lg.debug(f"Using Market ID: '{market_id}', Category: '{category}' for position check.")

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for {symbol} (Attempt {attempts + 1})...")
            positions: List[Dict] = []
            try:
                params = {'category': category, 'symbol': market_id}
                lg.debug(f"Fetching positions with params: {params}")
                if exchange.has.get('fetchPositions'):
                     all_positions = exchange.fetch_positions(params=params)
                     positions = [p for p in all_positions if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id]
                     lg.debug(f"Fetched {len(all_positions)} total positions ({category}), filtered to {len(positions)} matching {symbol}/{market_id}.")
                else: raise ccxt.NotSupported(f"{exchange.id} does not support fetchPositions.")
            except ccxt.ExchangeError as e:
                 no_pos_codes = [110025]
                 no_pos_messages = ["position not found", "no position", "position does not exist", "order not found or too late to cancel"]
                 err_str = str(e).lower()
                 code_str = ""; match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE); code_str = match.group(2) if match else str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
                 code_match = any(str(c) == code_str for c in no_pos_codes) if code_str else False
                 msg_match = any(msg in err_str for msg in no_pos_messages)
                 if code_match or msg_match: lg.info(f"No open position found for {symbol} (Detected via exchange message: {e})."); return None
                 else: raise e

            active_position_raw: Optional[Dict] = None
            size_threshold = Decimal('1e-9') # Default small value
            try:
                amt_step = market_info.get('amount_precision_step_decimal')
                if amt_step and amt_step > 0: size_threshold = amt_step * Decimal('0.01')
            except Exception: pass # Ignore errors getting precision here
            lg.debug(f"Using position size threshold > {size_threshold.normalize()} ({symbol}).")

            for pos in positions:
                size_str_info = str(pos.get('info', {}).get('size', '')).strip()
                size_str_std = str(pos.get('contracts', '')).strip()
                size_str = size_str_info if size_str_info else size_str_std
                if not size_str: continue
                try:
                    size_decimal = Decimal(size_str)
                    if abs(size_decimal) > size_threshold:
                        active_position_raw = pos
                        active_position_raw['size_decimal'] = size_decimal
                        lg.debug(f"Found active position entry ({symbol}): Size={size_decimal.normalize()}")
                        break
                    else: lg.debug(f"Skipping position entry with near-zero size ({symbol}, Size={size_decimal.normalize()}).")
                except (ValueError, InvalidOperation, TypeError) as parse_err: lg.warning(f"Could not parse position size '{size_str}' for {symbol}: {parse_err}. Skipping."); continue

            if active_position_raw:
                std_pos = active_position_raw.copy(); info_field = std_pos.get('info', {})
                side = std_pos.get('side'); parsed_size = std_pos['size_decimal']
                if side not in ['long', 'short']:
                    side_v5 = str(info_field.get('side', '')).strip().lower()
                    if side_v5 == 'buy': side = 'long'
                    elif side_v5 == 'sell': side = 'short'
                    elif parsed_size > size_threshold: side = 'long'
                    elif parsed_size < -size_threshold: side = 'short'
                    else: side = None
                if not side: lg.error(f"Could not determine side for active position ({symbol}). Data: {info_field}"); return None
                std_pos['side'] = side

                std_pos['entryPrice'] = _safe_market_decimal(std_pos.get('entryPrice') or info_field.get('avgPrice') or info_field.get('entryPrice'), 'pos.entryPrice', allow_zero=False)
                std_pos['leverage'] = _safe_market_decimal(std_pos.get('leverage') or info_field.get('leverage'), 'pos.leverage', allow_zero=False)
                std_pos['liquidationPrice'] = _safe_market_decimal(std_pos.get('liquidationPrice') or info_field.get('liqPrice'), 'pos.liqPrice', allow_zero=False)
                std_pos['unrealizedPnl'] = _safe_market_decimal(std_pos.get('unrealizedPnl') or info_field.get('unrealisedPnl') or info_field.get('unrealizedPnl'), 'pos.pnl', allow_zero=True, allow_negative=True)

                def get_protection_field(field_name: str) -> Optional[str]:
                    value = info_field.get(field_name); s_value = str(value).strip() if value is not None else None
                    if not s_value: return None
                    try: return s_value if abs(Decimal(s_value)) > Decimal('1e-12') else None
                    except: return None
                std_pos['stopLossPrice'] = get_protection_field('stopLoss')
                std_pos['takeProfitPrice'] = get_protection_field('takeProfit')
                std_pos['trailingStopLoss'] = get_protection_field('trailingStop')
                std_pos['tslActivationPrice'] = get_protection_field('activePrice')
                std_pos['be_activated'] = False # In-memory flag, reset each run
                std_pos['tsl_activated'] = bool(std_pos['trailingStopLoss']) # Infer initial state from API

                def format_decimal_log(value: Optional[Any]) -> str:
                    dec_val = _safe_market_decimal(value, 'log_format', allow_zero=True, allow_negative=True)
                    return dec_val.normalize() if dec_val is not None else 'N/A'
                ep_str = format_decimal_log(std_pos.get('entryPrice')); size_str = std_pos['size_decimal'].normalize()
                sl_str = std_pos.get('stopLossPrice') or 'N/A'; tp_str = std_pos.get('takeProfitPrice') or 'N/A'
                tsl_dist_str = std_pos.get('trailingStopLoss') or 'N/A'; tsl_act_str = std_pos.get('tslActivationPrice') or 'N/A'
                tsl_log = f"Dist={tsl_dist_str}/Act={tsl_act_str}" if tsl_dist_str != 'N/A' or tsl_act_str != 'N/A' else "N/A"
                pnl_str = format_decimal_log(std_pos.get('unrealizedPnl')); liq_str = format_decimal_log(std_pos.get('liquidationPrice'))
                lg.info(f"{NEON_GREEN}{BRIGHT}Active {side.upper()} Position Found ({symbol}):{RESET} Size={size_str}, Entry={ep_str}, Liq={liq_str}, PnL={pnl_str}, SL={sl_str}, TP={tp_str}, TSL={tsl_log}")

                try: final_position_info: PositionInfo = std_pos # type: ignore; return final_position_info
                except Exception as cast_err: lg.error(f"Internal error casting position dict to PositionInfo ({symbol}): {cast_err}"); return std_pos # type: ignore
            else: lg.info(f"No active position found for {symbol} (or size below threshold)."); return None

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Network error fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit fetching positions ({symbol}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error fetching positions: {e}. Stopping.{RESET}"); return None
        except ccxt.ExchangeError as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Exchange error fetching positions ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error fetching positions ({symbol}): {e}{RESET}", exc_info=True); return None

        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)

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
    if not market_info.get('is_contract', False): lg.info(f"Leverage setting skipped for {symbol}: Not a contract market."); return True
    if not isinstance(leverage, int) or leverage <= 0: lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage '{leverage}'."); return False
    if not hasattr(exchange, 'set_leverage') or not exchange.has.get('setLeverage'): lg.error(f"Exchange {exchange.id} does not support setLeverage."); return False

    market_id = market_info['id']
    attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Attempting to set leverage for {market_id} to {leverage}x (Attempt {attempts + 1})...")
            params = {}
            if 'bybit' in exchange.id.lower():
                 category = market_info.get('contract_type_str', 'Linear').lower()
                 if category not in ['linear', 'inverse']: lg.error(f"Leverage setting failed ({symbol}): Invalid category '{category}'."); return False
                 params = {'category': category, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
                 lg.debug(f"Using Bybit V5 setLeverage params: {params}")

            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params)
            lg.debug(f"set_leverage raw response ({symbol}): {response}")

            ret_code_str: Optional[str] = None; ret_msg: str = "N/A"
            if isinstance(response, dict):
                 info_dict = response.get('info', {})
                 raw_code = info_dict.get('retCode') if info_dict.get('retCode') is not None else response.get('retCode')
                 if raw_code is not None: ret_code_str = str(raw_code)
                 ret_msg = info_dict.get('retMsg', response.get('retMsg', 'Unknown message'))

            if ret_code_str == '0': lg.info(f"{NEON_GREEN}Leverage set successfully for {market_id} to {leverage}x (Code: 0).{RESET}"); return True
            elif ret_code_str == '110045': lg.info(f"{NEON_YELLOW}Leverage for {market_id} already set to {leverage}x (Code: 110045).{RESET}"); return True
            elif ret_code_str is not None and ret_code_str not in ['None', '0']:
                 error_message = f"Bybit API error setting leverage ({symbol}): {ret_msg} (Code: {ret_code_str})"
                 exc = ccxt.ExchangeError(error_message); setattr(exc, 'code', ret_code_str); raise exc
            else: lg.info(f"{NEON_GREEN}Leverage set/confirmed for {market_id} to {leverage}x (No specific error code).{RESET}"); return True

        except ccxt.ExchangeError as e:
            last_exception = e; err_code_str = ""; match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE); err_code_str = match.group(2) if match else str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            err_str_lower = str(e).lower(); lg.error(f"{NEON_RED}Exchange error setting leverage ({market_id}): {e} (Code: {err_code_str}){RESET}")
            if err_code_str == '110045' or "leverage not modified" in err_str_lower: lg.info(f"{NEON_YELLOW}Leverage already set (confirmed via error). Treating as success.{RESET}"); return True
            fatal_codes = ['10001', '10004', '110009', '110013', '110028', '110043', '110044', '110055', '3400045']
            fatal_messages = ["margin mode", "position exists", "risk limit", "parameter error", "insufficient available balance", "invalid leverage value"]
            if err_code_str in fatal_codes or any(msg in err_str_lower for msg in fatal_messages): lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE leverage error ({symbol}). Aborting.{RESET}"); return False
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for ExchangeError setting leverage ({symbol}).{RESET}"); return False
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e; lg.warning(f"{NEON_YELLOW}Network error setting leverage ({market_id}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for NetworkError setting leverage ({symbol}).{RESET}"); return False
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error setting leverage ({symbol}): {e}. Stopping.{RESET}"); return False
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error setting leverage ({market_id}): {e}{RESET}", exc_info=True); return False

        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts)

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
    lg = logger; symbol = market_info['symbol']
    quote_currency = market_info.get('quote', 'QUOTE'); base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False); is_inverse = market_info.get('is_inverse', False)
    size_unit = "Contracts" if is_contract else base_currency

    lg.info(f"{BRIGHT}--- Position Sizing Calculation ({symbol}) ---{RESET}")

    if not isinstance(balance, Decimal) or balance <= Decimal('0'): lg.error(f"Sizing failed ({symbol}): Invalid balance: {balance}."); return None
    try:
        risk_decimal = Decimal(str(risk_per_trade))
        if not (Decimal('0') < risk_decimal <= Decimal('1')): raise ValueError("Risk must be > 0.0 and <= 1.0.")
    except (ValueError, InvalidOperation, TypeError) as e: lg.error(f"Sizing failed ({symbol}): Invalid risk_per_trade '{risk_per_trade}': {e}"); return None
    if not isinstance(initial_stop_loss_price, Decimal) or initial_stop_loss_price <= Decimal('0'): lg.error(f"Sizing failed ({symbol}): Invalid SL price: {initial_stop_loss_price}."); return None
    if not isinstance(entry_price, Decimal) or entry_price <= Decimal('0'): lg.error(f"Sizing failed ({symbol}): Invalid entry price: {entry_price}."); return None
    if initial_stop_loss_price == entry_price: lg.error(f"Sizing failed ({symbol}): SL price cannot equal entry price."); return None

    try:
        amount_step = market_info['amount_precision_step_decimal']
        price_step = market_info['price_precision_step_decimal']
        min_amount = market_info['min_amount_decimal']
        max_amount = market_info['max_amount_decimal']
        min_cost = market_info['min_cost_decimal']
        max_cost = market_info['max_cost_decimal']
        contract_size = market_info['contract_size_decimal']
        if amount_step is None or amount_step <= 0: raise ValueError("Invalid amount step")
        if price_step is None or price_step <= 0: raise ValueError("Invalid price step")
        if contract_size <= Decimal('0'): raise ValueError("Invalid contract size")
        min_amount_eff = min_amount if min_amount is not None else Decimal('0'); max_amount_eff = max_amount if max_amount is not None else Decimal('inf')
        min_cost_eff = min_cost if min_cost is not None else Decimal('0'); max_cost_eff = max_cost if max_cost is not None else Decimal('inf')
        lg.debug(f"  Market Constraints ({symbol}): AmtStep={amount_step.normalize()}, Min/Max Amt={min_amount_eff.normalize()}/{max_amount_eff.normalize()}, Min/Max Cost={min_cost_eff.normalize()}/{max_cost_eff.normalize()}, ContrSize={contract_size.normalize()}")
    except (KeyError, ValueError, TypeError) as e: lg.error(f"Sizing failed ({symbol}): Error accessing market details: {e}"); lg.debug(f" MarketInfo: {market_info}"); return None

    risk_amount_quote = (balance * risk_decimal).quantize(Decimal('1e-8'), ROUND_DOWN)
    stop_loss_distance = abs(entry_price - initial_stop_loss_price)
    if stop_loss_distance <= Decimal('0'): lg.error(f"Sizing failed ({symbol}): SL distance is zero."); return None

    lg.info(f"  Balance: {balance.normalize()} {quote_currency}, Risk: {risk_decimal:.2%} ({risk_amount_quote.normalize()} {quote_currency})")
    lg.info(f"  Entry: {entry_price.normalize()}, SL: {initial_stop_loss_price.normalize()}, SL Dist: {stop_loss_distance.normalize()}")
    lg.info(f"  Contract Type: {market_info['contract_type_str']}")

    calculated_size = Decimal('0')
    try:
        if not is_inverse: # Linear/Spot
            value_change_per_unit = stop_loss_distance * contract_size
            if value_change_per_unit <= Decimal('1e-18'): lg.error(f"Sizing failed ({symbol}, Linear/Spot): Value change per unit near zero."); return None
            calculated_size = risk_amount_quote / value_change_per_unit
            lg.debug(f"  Linear/Spot Size Calc: {risk_amount_quote} / ({stop_loss_distance} * {contract_size}) = {calculated_size}")
        else: # Inverse
            if entry_price <= 0 or initial_stop_loss_price <= 0: lg.error(f"Sizing failed ({symbol}, Inverse): Entry/SL price non-positive."); return None
            inverse_factor = abs( (Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price) )
            if inverse_factor <= Decimal('1e-18'): lg.error(f"Sizing failed ({symbol}, Inverse): Inverse factor near zero."); return None
            risk_per_contract = contract_size * inverse_factor
            if risk_per_contract <= Decimal('1e-18'): lg.error(f"Sizing failed ({symbol}, Inverse): Risk per contract near zero."); return None
            calculated_size = risk_amount_quote / risk_per_contract
            lg.debug(f"  Inverse Size Calc: {risk_amount_quote} / ({contract_size} * {inverse_factor}) = {calculated_size}")
    except (InvalidOperation, OverflowError, ZeroDivisionError) as calc_err: lg.error(f"Sizing failed ({symbol}): Calculation error: {calc_err}."); return None

    if calculated_size <= Decimal('0'): lg.error(f"Sizing failed ({symbol}): Initial calc size non-positive ({calculated_size.normalize()})."); return None
    lg.info(f"  Initial Calculated Size ({symbol}) = {calculated_size.normalize()} {size_unit}")

    adjusted_size = calculated_size

    def estimate_cost(size: Decimal, price: Decimal) -> Optional[Decimal]:
        if not isinstance(size, Decimal) or not isinstance(price, Decimal) or price <= 0 or size <= 0: return None
        try:
             cost = (size * price * contract_size) if not is_inverse else (size * contract_size) / price
             return cost.quantize(Decimal('1e-8'), ROUND_UP)
        except: return None

    if min_amount_eff > 0 and adjusted_size < min_amount_eff:
        lg.warning(f"{NEON_YELLOW}Sizing Update ({symbol}): Calc size {adjusted_size.normalize()} < min amount {min_amount_eff.normalize()}. Adjusting UP.{RESET}")
        adjusted_size = min_amount_eff
    if adjusted_size > max_amount_eff:
        lg.warning(f"{NEON_YELLOW}Sizing Update ({symbol}): Calc size {adjusted_size.normalize()} > max amount {max_amount_eff.normalize()}. Adjusting DOWN.{RESET}")
        adjusted_size = max_amount_eff
    lg.debug(f"  Size after Amount Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")

    cost_adjustment_made = False
    estimated_current_cost = estimate_cost(adjusted_size, entry_price)
    if estimated_current_cost is not None:
        lg.debug(f"  Estimated Cost (after amount limits, {symbol}): {estimated_current_cost.normalize()} {quote_currency}")
        if estimated_current_cost < min_cost_eff:
            lg.warning(f"{NEON_YELLOW}Sizing Update ({symbol}): Est cost {estimated_current_cost.normalize()} < min cost {min_cost_eff.normalize()}. Increasing size.{RESET}")
            cost_adjustment_made = True
            try:
                if not is_inverse: required_size_for_min_cost = min_cost_eff / (entry_price * contract_size)
                else: required_size_for_min_cost = (min_cost_eff * entry_price) / contract_size
                if required_size_for_min_cost <= 0: raise ValueError("Required size for min cost <= 0.")
                lg.info(f"  Size required for min cost ({symbol}): {required_size_for_min_cost.normalize()} {size_unit}")
                if required_size_for_min_cost > max_amount_eff: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Cannot meet min cost ({min_cost_eff.normalize()}) without exceeding max amount ({max_amount_eff.normalize()}).{RESET}"); return None
                adjusted_size = max(min_amount_eff, required_size_for_min_cost)
            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as cost_calc_err: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error calc size for min cost: {cost_calc_err}.{RESET}"); return None
        elif estimated_current_cost > max_cost_eff:
            lg.warning(f"{NEON_YELLOW}Sizing Update ({symbol}): Est cost {estimated_current_cost.normalize()} > max cost {max_cost_eff.normalize()}. Reducing size.{RESET}")
            cost_adjustment_made = True
            try:
                if not is_inverse: max_size_for_max_cost = max_cost_eff / (entry_price * contract_size)
                else: max_size_for_max_cost = (max_cost_eff * entry_price) / contract_size
                if max_size_for_max_cost <= 0: raise ValueError("Max size for max cost <= 0.")
                lg.info(f"  Max size allowed by max cost ({symbol}): {max_size_for_max_cost.normalize()} {size_unit}")
                adjusted_size = max(min_amount_eff, min(adjusted_size, max_size_for_max_cost))
            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as cost_calc_err: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error calc max size for max cost: {cost_calc_err}.{RESET}"); return None
    elif min_cost_eff > 0 or max_cost_eff < Decimal('inf'): lg.warning(f"Could not estimate cost ({symbol}) to check cost limits.")

    if cost_adjustment_made: lg.info(f"  Size after Cost Limits ({symbol}): {adjusted_size.normalize()} {size_unit}")

    final_size = adjusted_size
    try:
        if amount_step <= 0: raise ValueError("Amount step size must be positive.")
        quantized_steps = (adjusted_size / amount_step).quantize(Decimal('1'), ROUND_DOWN)
        final_size = quantized_steps * amount_step
        if final_size != adjusted_size: lg.info(f"Applied amount precision ({symbol}, Step: {amount_step.normalize()}, Rounded DOWN): {adjusted_size.normalize()} -> {final_size.normalize()} {size_unit}")
        else: lg.debug(f"Size {adjusted_size.normalize()} already matches precision step {amount_step.normalize()} ({symbol}).")
    except (InvalidOperation, ValueError, TypeError) as fmt_err: lg.error(f"{NEON_RED}Error applying amount precision ({symbol}): {fmt_err}. Using unrounded: {final_size.normalize()}{RESET}")

    if final_size <= Decimal('0'): lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size <= 0 ({final_size.normalize()}) after adjustments.{RESET}"); return None
    if final_size < min_amount_eff: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} < min amount {min_amount_eff.normalize()} after precision.{RESET}"); return None
    if final_size > max_amount_eff: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final size {final_size.normalize()} > max amount {max_amount_eff.normalize()} after precision.{RESET}"); return None

    final_estimated_cost = estimate_cost(final_size, entry_price)
    if final_estimated_cost is not None:
        lg.debug(f"  Final Estimated Cost ({symbol}, after precision): {final_estimated_cost.normalize()} {quote_currency}")
        if final_estimated_cost < min_cost_eff:
             lg.warning(f"{NEON_YELLOW}Sizing Update ({symbol}): Final est cost {final_estimated_cost.normalize()} < min cost {min_cost_eff.normalize()} after rounding down.{RESET}")
             try:
                 next_step_size = final_size + amount_step; next_step_cost = estimate_cost(next_step_size, entry_price)
                 if next_step_cost is not None:
                     can_bump_up = (next_step_cost >= min_cost_eff) and (next_step_size <= max_amount_eff) and (next_step_cost <= max_cost_eff)
                     if can_bump_up:
                         lg.info(f"{NEON_YELLOW}Bumping final size ({symbol}) up one step to {next_step_size.normalize()} to meet min cost.{RESET}")
                         final_size = next_step_size
                         final_cost_after_bump = estimate_cost(final_size, entry_price)
                         lg.debug(f"  Final Estimated Cost after bump ({symbol}): {final_cost_after_bump.normalize() if final_cost_after_bump else 'N/A'}")
                     else: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Cannot meet min cost. Bumping size violates limits.{RESET}"); return None
                 else: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Could not estimate cost for bumped size check.{RESET}"); return None
             except Exception as bump_err: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Error bumping size for min cost: {bump_err}.{RESET}"); return None
        elif final_estimated_cost > max_cost_eff: lg.error(f"{NEON_RED}Sizing failed ({symbol}): Final est cost {final_estimated_cost.normalize()} > max cost {max_cost_eff.normalize()} after precision.{RESET}"); return None
    elif min_cost_eff > 0: lg.warning(f"Could not perform final cost check ({symbol}) against min cost ({min_cost_eff.normalize()}).")

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
    lg = logger; attempts = 0; last_exception = None
    lg.info(f"Attempting to cancel order ID: {order_id} for symbol {symbol}...")

    market_id = symbol; category = 'spot'; params = {}
    if 'bybit' in exchange.id.lower():
        try:
            market = exchange.market(symbol)
            market_id = market['id']
            category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
            params = {'category': category, 'symbol': market_id}
            lg.debug(f"Using Bybit V5 cancelOrder params: {params}")
        except Exception as e: lg.warning(f"Could not determine category/market_id for cancel order {order_id} ({symbol}): {e}. Using defaults.")

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Cancel order attempt {attempts + 1} for ID {order_id} ({symbol})...")
            exchange.cancel_order(order_id, symbol, params=params)
            lg.info(f"{NEON_GREEN}Successfully cancelled order ID: {order_id} for {symbol}.{RESET}")
            return True
        except ccxt.OrderNotFound: lg.warning(f"{NEON_YELLOW}Order ID {order_id} ({symbol}) not found. Already cancelled/filled? Treating as success.{RESET}"); return True
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: last_exception = e; lg.warning(f"{NEON_YELLOW}Network error cancelling order {order_id} ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 2; lg.warning(f"{NEON_YELLOW}Rate limit cancelling order {order_id} ({symbol}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.ExchangeError as e: last_exception = e; lg.error(f"{NEON_RED}Exchange error cancelling order {order_id} ({symbol}): {e}{RESET}")
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error cancelling order ({symbol}): {e}. Stopping.{RESET}"); return False
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error cancelling order {order_id} ({symbol}): {e}{RESET}", exc_info=True); return False

        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)

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
    side_map = {"BUY": "buy", "SELL": "sell", "EXIT_SHORT": "buy", "EXIT_LONG": "sell"}
    side = side_map.get(trade_signal.upper())

    if side is None: lg.error(f"Invalid trade signal '{trade_signal}' for place_trade ({symbol})."); return None
    if not isinstance(position_size, Decimal) or position_size <= Decimal('0'): lg.error(f"Invalid position size for place_trade ({symbol}): {position_size}."); return None

    order_type = 'market'; is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', 'BASE'); size_unit = "Contracts" if is_contract else base_currency
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"; market_id = market_info['id']

    try:
         amount_float = float(position_size)
         if amount_float <= 1e-15: raise ValueError("Size negligible after float conversion.")
    except Exception as float_err: lg.error(f"Failed to convert size {position_size.normalize()} ({symbol}) to float: {float_err}"); return None

    order_args: Dict[str, Any] = {'symbol': symbol, 'type': order_type, 'side': side, 'amount': amount_float}
    order_params: Dict[str, Any] = {}

    if 'bybit' in exchange.id.lower() and is_contract:
        try:
            category = market_info.get('contract_type_str', 'Linear').lower()
            if category not in ['linear', 'inverse']: raise ValueError(f"Invalid category '{category}'.")
            order_params = {'category': category, 'positionIdx': 0} # Assume One-Way Mode
            if reduce_only: order_params['reduceOnly'] = True; order_params['timeInForce'] = 'IOC'
            lg.debug(f"Using Bybit V5 order params ({symbol}): {order_params}")
        except Exception as e: lg.error(f"Failed to set Bybit V5 params for order ({symbol}): {e}.")

    if params and isinstance(params, dict): order_params.update(params); lg.debug(f"Added custom params ({symbol}): {params}")
    if order_params: order_args['params'] = order_params

    lg.warning(f"{BRIGHT}===> PLACING {action_desc} | {side.upper()} {order_type.upper()} Order | {symbol} | Size: {position_size.normalize()} {size_unit} <==={RESET}")
    if order_params: lg.debug(f"  with Params ({symbol}): {order_params}")

    attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order ({symbol}, Attempt {attempts + 1})...")
            order_result = exchange.create_order(**order_args)
            order_id = order_result.get('id', 'N/A'); status = order_result.get('status', 'N/A')
            avg_price_dec = _safe_market_decimal(order_result.get('average'), 'order.average', allow_zero=True)
            filled_dec = _safe_market_decimal(order_result.get('filled'), 'order.filled', allow_zero=True)
            log_msg = f"{NEON_GREEN}{action_desc} Order Placed Successfully!{RESET} ID: {order_id}, Status: {status}"
            if avg_price_dec: log_msg += f", AvgFillPrice: ~{avg_price_dec.normalize()}"
            if filled_dec is not None: log_msg += f", FilledAmount: {filled_dec.normalize()}"
            lg.info(log_msg); lg.debug(f"Full order result ({symbol}): {order_result}"); return order_result

        except ccxt.InsufficientFunds as e: last_exception = e; lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Insufficient Funds. {e}{RESET}"); try: bal = fetch_balance(exchange, QUOTE_CURRENCY, lg); lg.error(f"  Balance: {bal.normalize() if bal else 'Fetch Failed'}") except: pass; return None
        except ccxt.InvalidOrder as e:
            last_exception = e; lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Invalid Order Parameters. {e}{RESET}"); lg.error(f"  Args Sent: {order_args}")
            err_lower = str(e).lower(); min_amt = market_info.get('min_amount_decimal', 'N/A'); min_cost = market_info.get('min_cost_decimal', 'N/A'); amt_step = market_info.get('amount_precision_step_decimal', 'N/A'); max_amt = market_info.get('max_amount_decimal', 'N/A'); max_cost = market_info.get('max_cost_decimal', 'N/A')
            if any(s in err_lower for s in ["minimum order", "min order", "too small", "lower than limit"]): lg.error(f"  >> Hint: Check size ({position_size.normalize()}) vs MinAmt: {min_amt}, MinCost: {min_cost}).")
            elif any(s in err_lower for s in ["precision", "lot size", "step size", "multiple of"]): lg.error(f"  >> Hint: Check size ({position_size.normalize()}) vs Amount Step: {amt_step}).")
            elif any(s in err_lower for s in ["exceed maximum", "max order", "too large", "greater than limit"]): lg.error(f"  >> Hint: Check size ({position_size.normalize()}) vs MaxAmt: {max_amt}, MaxCost: {max_cost}).")
            elif any(s in err_lower for s in ["reduce only", "reduce-only"]): lg.error(f"  >> Hint: Reduce-only failed. Check position state/size.")
            elif "position idx not match" in err_lower: lg.error(f"  >> Hint: Check Bybit position mode (One-Way vs Hedge).")
            return None
        except ccxt.ExchangeError as e:
            last_exception = e; err_code_str = ""; match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE); err_code_str = match.group(2) if match else str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            lg.error(f"{NEON_RED}Order Failed ({symbol} {action_desc}): Exchange Error. {e} (Code: {err_code_str}){RESET}")
            fatal_order_codes = ['10001', '10004', '110007', '110013', '110014', '110017', '110025', '110040', '30086', '3303001', '3303005', '3400060', '3400088']
            fatal_messages = ["invalid parameter", "precision error", "exceed limit", "risk limit", "invalid symbol", "reduce only check failed", "lot size error", "insufficient available balance", "leverage exceed limit", "trigger liquidation"]
            err_str_lower = str(e).lower()
            if err_code_str in fatal_order_codes or any(msg in err_str_lower for msg in fatal_messages): lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE order error ({symbol}). Aborting.{RESET}"); return None
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for ExchangeError placing order ({symbol}).{RESET}"); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e; lg.warning(f"{NEON_YELLOW}Network error placing order ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for NetworkError placing order ({symbol}).{RESET}"); return None
        except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit placing order ({symbol}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error placing order ({symbol}): {e}. Stopping.{RESET}"); return None
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error placing order ({symbol}): {e}{RESET}", exc_info=True); return None

        attempts += 1
        if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS * attempts)

    lg.error(f"{NEON_RED}Failed to place {action_desc} order ({symbol}) after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None

def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, position_info: PositionInfo, logger: logging.Logger,
                             stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
                             trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool:
    """
    Internal helper: Sets Stop Loss (SL), Take Profit (TP), and/or Trailing Stop Loss (TSL)
    for an existing position using Bybit's V5 private API endpoint `/v5/position/set-trading-stop`.

    **WARNING:** Uses a direct, non-standard CCXT private API call specific to Bybit V5.

    - Handles parameter validation and formatting.
    - Understands Bybit's TSL overriding fixed SL.
    - Allows clearing protection levels by passing Decimal('0').
    - Includes retry logic.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol.
        market_info (MarketInfo): Standardized market info.
        position_info (PositionInfo): Standardized position info.
        logger (logging.Logger): The logger instance.
        stop_loss_price (Optional[Decimal]): Desired fixed SL price or 0 to clear.
        take_profit_price (Optional[Decimal]): Desired fixed TP price or 0 to clear.
        trailing_stop_distance (Optional[Decimal]): Desired TSL distance or 0 to clear.
        tsl_activation_price (Optional[Decimal]): Required if TSL distance > 0.

    Returns:
        bool: True if protection set/updated successfully, False otherwise.
    """
    lg = logger; endpoint = '/v5/position/set-trading-stop'
    lg.debug(f"Preparing {endpoint} call for {symbol}")

    if not market_info.get('is_contract', False): lg.error(f"Protection failed ({symbol}): Not a contract market."); return False
    if not position_info: lg.error(f"Protection failed ({symbol}): Missing position info."); return False

    pos_side = position_info.get('side')
    entry_price_any = position_info.get('entryPrice')
    if pos_side not in ['long', 'short']: lg.error(f"Protection failed ({symbol}): Invalid position side '{pos_side}'."); return False
    try:
        if entry_price_any is None: raise ValueError("Missing entry price.")
        entry_price = Decimal(str(entry_price_any))
        if not entry_price.is_finite() or entry_price <= 0: raise ValueError("Invalid entry price.")
        price_tick = market_info['price_precision_step_decimal']
        if price_tick is None or not price_tick.is_finite() or price_tick <= 0: raise ValueError("Invalid price tick.")
    except (KeyError, ValueError, InvalidOperation, TypeError) as e: lg.error(f"Protection failed ({symbol}): Invalid entry price or market tick: {e}"); return False

    params_to_set: Dict[str, Any] = {}; log_parts: List[str] = [f"{BRIGHT}Preparing protection update ({symbol} {pos_side.upper()} @ Entry: {entry_price.normalize()}):{RESET}"]
    any_protection_requested = False; set_tsl_active = False

    try:
        def format_param(price_decimal: Optional[Union[Decimal, int, float]], param_name: str) -> Optional[str]:
            if price_decimal is None: return None
            try:
                d_price = Decimal(str(price_decimal))
                if d_price.is_zero(): return "0"
                formatted = _format_price(exchange, market_info['symbol'], d_price)
                if formatted: return formatted
                else: lg.error(f"Failed to format {param_name} ({symbol}): Input {d_price.normalize()}"); return None
            except Exception as e: lg.error(f"Error converting {param_name} ({symbol}) value '{price_decimal}': {e}"); return None

        # TSL
        if trailing_stop_distance is not None:
            any_protection_requested = True
            try:
                tsl_dist_dec = Decimal(str(trailing_stop_distance))
                if tsl_dist_dec > 0:
                    min_valid_distance = max(tsl_dist_dec, price_tick)
                    if tsl_activation_price is None: raise ValueError("TSL activation price required.")
                    tsl_act_dec = Decimal(str(tsl_activation_price));
                    if tsl_act_dec <= 0: raise ValueError("TSL activation price must be positive.")
                    is_valid_act = (pos_side == 'long' and tsl_act_dec > entry_price) or (pos_side == 'short' and tsl_act_dec < entry_price)
                    if not is_valid_act: raise ValueError(f"TSL activation {tsl_act_dec.normalize()} invalid for {pos_side} entry {entry_price.normalize()}.")
                    fmt_dist = format_param(min_valid_distance, "TSL Distance"); fmt_act = format_param(tsl_act_dec, "TSL Activation")
                    if fmt_dist and fmt_act: params_to_set['trailingStop'] = fmt_dist; params_to_set['activePrice'] = fmt_act; log_parts.append(f"  - Setting TSL: Dist={fmt_dist}, Act={fmt_act}"); set_tsl_active = True
                    else: lg.error(f"TSL setting failed ({symbol}): Format error (Dist: {fmt_dist}, Act: {fmt_act}).")
                elif tsl_dist_dec.is_zero(): params_to_set['trailingStop'] = "0"; params_to_set['activePrice'] = "0"; log_parts.append("  - Clearing TSL"); set_tsl_active = False
                else: raise ValueError(f"Invalid negative TSL distance: {tsl_dist_dec.normalize()}")
            except (ValueError, InvalidOperation, TypeError) as tsl_err: lg.error(f"TSL param validation failed ({symbol}): {tsl_err}")

        # Fixed SL (only if active TSL not being set)
        if not set_tsl_active:
            if stop_loss_price is not None:
                any_protection_requested = True
                try:
                    sl_dec = Decimal(str(stop_loss_price))
                    if sl_dec > 0:
                        is_valid_sl = (pos_side == 'long' and sl_dec < entry_price) or (pos_side == 'short' and sl_dec > entry_price)
                        if not is_valid_sl: raise ValueError(f"SL price {sl_dec.normalize()} invalid for {pos_side} entry {entry_price.normalize()}.")
                        fmt_sl = format_param(sl_dec, "Stop Loss")
                        if fmt_sl: params_to_set['stopLoss'] = fmt_sl; log_parts.append(f"  - Setting Fixed SL: {fmt_sl}")
                        else: lg.error(f"SL setting failed ({symbol}): Format error {sl_dec.normalize()}.")
                    elif sl_dec.is_zero():
                        if 'stopLoss' not in params_to_set: params_to_set['stopLoss'] = "0"; log_parts.append("  - Clearing Fixed SL")
                except (ValueError, InvalidOperation, TypeError) as sl_err: lg.error(f"SL param validation failed ({symbol}): {sl_err}")
        elif stop_loss_price is not None and Decimal(str(stop_loss_price)) > 0: lg.warning(f"Ignoring fixed SL request ({stop_loss_price}) due to active TSL setting ({symbol}).")

        # Fixed TP
        if take_profit_price is not None:
            any_protection_requested = True
            try:
                tp_dec = Decimal(str(take_profit_price))
                if tp_dec > 0:
                    is_valid_tp = (pos_side == 'long' and tp_dec > entry_price) or (pos_side == 'short' and tp_dec < entry_price)
                    if not is_valid_tp: raise ValueError(f"TP price {tp_dec.normalize()} invalid for {pos_side} entry {entry_price.normalize()}.")
                    fmt_tp = format_param(tp_dec, "Take Profit")
                    if fmt_tp: params_to_set['takeProfit'] = fmt_tp; log_parts.append(f"  - Setting Fixed TP: {fmt_tp}")
                    else: lg.error(f"TP setting failed ({symbol}): Format error {tp_dec.normalize()}.")
                elif tp_dec.is_zero():
                     if 'takeProfit' not in params_to_set: params_to_set['takeProfit'] = "0"; log_parts.append("  - Clearing Fixed TP")
            except (ValueError, InvalidOperation, TypeError) as tp_err: lg.error(f"TP param validation failed ({symbol}): {tp_err}")

    except Exception as validation_err: lg.error(f"Unexpected error during protection validation ({symbol}): {validation_err}", exc_info=True); return False

    if not params_to_set:
        if any_protection_requested: lg.warning(f"{NEON_YELLOW}Protection update skipped ({symbol}): No valid parameters after validation.{RESET}"); return False
        else: lg.debug(f"No protection changes requested ({symbol}). Skipping API call."); return True

    category = market_info.get('contract_type_str', 'Linear').lower(); market_id = market_info['id']
    position_idx = 0 # Assume One-Way Mode
    try:
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None: position_idx = int(pos_idx_val)
        if position_idx != 0: lg.warning(f"Detected positionIdx={position_idx} for {symbol}. Ensure One-Way mode.")
    except: lg.warning(f"Could not parse positionIdx for {symbol}. Using default {position_idx}.")

    final_api_params: Dict[str, Any] = {'category': category, 'symbol': market_id, 'positionIdx': position_idx}
    final_api_params.update(params_to_set)
    final_api_params.update({'tpslMode': 'Full', 'slTriggerBy': 'LastPrice', 'tpTriggerBy': 'LastPrice', 'slOrderType': 'Market', 'tpOrderType': 'Market'})

    lg.info("\n".join(log_parts)); lg.debug(f"  Final API params for {endpoint} ({symbol}): {final_api_params}")

    attempts = 0; last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing private_post {endpoint} ({symbol}, Attempt {attempts + 1})...")
            response = exchange.private_post(endpoint, params=final_api_params)
            lg.debug(f"Raw response from {endpoint} ({symbol}): {response}")

            ret_code = response.get('retCode'); ret_msg = response.get('retMsg', 'Unknown message')
            if ret_code == 0:
                 no_change_msgs = ["not modified", "no need to modify", "parameter not change", "order is not modified", "same as the current"]
                 if any(m in ret_msg.lower() for m in no_change_msgs): lg.info(f"{NEON_YELLOW}Protection parameters already set or no change needed for {symbol} (API Msg: '{ret_msg}').{RESET}")
                 else: lg.info(f"{NEON_GREEN}Protection set/updated successfully for {symbol} (Code: 0).{RESET}")
                 return True
            else:
                 error_message = f"Bybit API error setting protection ({symbol}): {ret_msg} (Code: {ret_code})"
                 lg.error(f"{NEON_RED}{error_message}{RESET}")
                 exc = ccxt.ExchangeError(error_message); setattr(exc, 'code', ret_code); raise exc

        except ccxt.ExchangeError as e:
            last_exception = e; err_code_str = ""; match = re.search(r'(retCode|ret_code)\s*[:=]\s*(\d+)', str(e.args[0] if e.args else ''), re.IGNORECASE); err_code_str = match.group(2) if match else str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            err_str_lower = str(e).lower(); lg.warning(f"{NEON_YELLOW}Exchange error setting protection ({symbol}): {e} (Code: {err_code_str}). Retry {attempts + 1}...{RESET}")
            fatal_protect_codes = ['10001', '10002', '110013', '110036', '110043', '110084', '110085', '110086', '110103', '110104', '110110', '3400045', '3400048', '3400051', '3400052', '3400070', '3400071', '3400072', '3400073']
            fatal_messages = ["invalid parameter", "invalid price", "cannot be higher than", "cannot be lower than", "position status not normal", "precision error", "activation price invalid", "distance invalid", "cannot be the same", "price is out of range", "less than mark price"]
            if err_code_str in fatal_protect_codes or any(msg in err_str_lower for msg in fatal_messages): lg.error(f"{NEON_RED} >> Hint: NON-RETRYABLE protection setting error ({symbol}). Aborting.{RESET}"); return False
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for ExchangeError setting protection ({symbol}).{RESET}"); return False
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e; lg.warning(f"{NEON_YELLOW}Network error setting protection ({symbol}): {e}. Retry {attempts + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries for NetworkError setting protection ({symbol}).{RESET}"); return False
        except ccxt.RateLimitExceeded as e: last_exception = e; wait_time = RETRY_DELAY_SECONDS * 3; lg.warning(f"{NEON_YELLOW}Rate limit setting protection ({symbol}): {e}. Pausing {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: last_exception = e; lg.critical(f"{NEON_RED}Auth error setting protection ({symbol}): {e}. Stopping.{RESET}"); return False
        except Exception as e: last_exception = e; lg.error(f"{NEON_RED}Unexpected error setting protection ({symbol}): {e}{RESET}", exc_info=True); return False

        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts)

    lg.error(f"{NEON_RED}Failed to set protection for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return False

# --- Volumatic Trend + OB Strategy Implementation (Class Definition as before) ---
class VolumaticOBStrategy:
    """
    Encapsulates the logic for calculating the Volumatic Trend and Pivot Order Block strategy indicators.

    Responsibilities:
    - Takes historical OHLCV data as input.
    - Calculates Volumatic Trend indicators: EMA/SWMA trend, ATR, Volatility Bands, Normalized Volume.
    - Identifies Pivot Highs and Pivot Lows.
    - Creates and manages Order Blocks (OBs): Active, Violated, Extended.
    - Prunes active OBs to a configured maximum.
    - Returns structured analysis results (`StrategyAnalysisResults`).

    Note: Uses float for pandas-ta calculations, converts back to Decimal for final results.
    """
    def __init__(self, config: Dict[str, Any], market_info: MarketInfo, logger: logging.Logger):
        self.config = config; self.market_info = market_info; self.logger = logger; self.lg = logger
        self.symbol = market_info.get('symbol', 'UnknownSymbol')
        strategy_cfg = config.get("strategy_params", {})

        try:
            self.vt_length = int(strategy_cfg["vt_length"]); self.vt_atr_period = int(strategy_cfg["vt_atr_period"])
            self.vt_vol_ema_length = int(strategy_cfg["vt_vol_ema_length"]); self.vt_atr_multiplier = Decimal(str(strategy_cfg["vt_atr_multiplier"]))
            self.ob_source = str(strategy_cfg["ob_source"]); self.ph_left = int(strategy_cfg["ph_left"]); self.ph_right = int(strategy_cfg["ph_right"])
            self.pl_left = int(strategy_cfg["pl_left"]); self.pl_right = int(strategy_cfg["pl_right"])
            self.ob_extend = bool(strategy_cfg["ob_extend"]); self.ob_max_boxes = int(strategy_cfg["ob_max_boxes"])

            if not (self.vt_length > 0 and self.vt_atr_period > 0 and self.vt_vol_ema_length > 0 and self.vt_atr_multiplier > 0 and
                    self.ph_left > 0 and self.ph_right > 0 and self.pl_left > 0 and self.pl_right > 0 and self.ob_max_boxes > 0): raise ValueError("Invalid strategy params.")
            if self.ob_source not in ["Wicks", "Body"]: raise ValueError(f"Invalid ob_source '{self.ob_source}'.")
        except Exception as e: self.lg.critical(f"{NEON_RED}FATAL: Failed init VolumaticOBStrategy ({self.symbol}): {e}{RESET}"); raise ValueError(f"Strategy init failed ({self.symbol}): {e}") from e

        self.bull_boxes: List[OrderBlock] = []; self.bear_boxes: List[OrderBlock] = []
        required_for_vt = max(self.vt_length * 2, self.vt_atr_period, self.vt_vol_ema_length)
        required_for_pivots = max(self.ph_left + self.ph_right + 1, self.pl_left + self.pl_right + 1)
        self.min_data_len = max(required_for_vt, required_for_pivots) + 50

        self.lg.info(f"{NEON_CYAN}--- Initializing VolumaticOB Strategy Engine ({self.symbol}) ---{RESET}")
        self.lg.info(f"  VT Params: Len={self.vt_length}, ATR={self.vt_atr_period}, VolEMA={self.vt_vol_ema_length}, ATR Mult={self.vt_atr_multiplier.normalize()}")
        self.lg.info(f"  OB Params: Src='{self.ob_source}', PH(L/R)={self.ph_left}/{self.ph_right}, PL(L/R)={self.pl_left}/{self.pl_right}, Extend={self.ob_extend}, MaxOB={self.ob_max_boxes}")
        self.lg.info(f"  Min Data Recommended: ~{self.min_data_len} candles")
        if self.min_data_len > BYBIT_API_KLINE_LIMIT + 50: self.lg.warning(f"{NEON_YELLOW}CONFIG NOTE ({self.symbol}): Strategy needs {self.min_data_len} candles. Ensure 'fetch_limit' is sufficient.{RESET}")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        if not isinstance(series, pd.Series) or len(series) < 4 or length <= 0: return pd.Series(np.nan, index=series.index, dtype=float)
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.isnull().all(): return pd.Series(np.nan, index=series.index, dtype=float)
        weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
        swma = numeric_series.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)
        return ta.ema(swma, length=length, fillna=np.nan)

    def _find_pivots(self, series: pd.Series, left_bars: int, right_bars: int, is_high: bool) -> pd.Series:
        if not isinstance(series, pd.Series) or series.empty or left_bars < 1 or right_bars < 1: return pd.Series(False, index=series.index, dtype=bool)
        num_series = pd.to_numeric(series, errors='coerce')
        if num_series.isnull().all(): return pd.Series(False, index=series.index, dtype=bool)
        pivot_conditions = num_series.notna()
        for i in range(1, left_bars + 1):
            condition = (num_series > num_series.shift(i)) if is_high else (num_series < num_series.shift(i))
            pivot_conditions &= condition.fillna(False)
        for i in range(1, right_bars + 1):
            condition = (num_series > num_series.shift(-i)) if is_high else (num_series < num_series.shift(-i))
            pivot_conditions &= condition.fillna(False)
        return pivot_conditions.fillna(False)

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        empty_results = StrategyAnalysisResults(dataframe=pd.DataFrame(), last_close=Decimal('NaN'), current_trend_up=None, trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=None, atr=None, upper_band=None, lower_band=None)
        if df_input.empty: self.lg.error(f"Strategy update failed ({self.symbol}): Input DataFrame empty."); return empty_results
        df = df_input.copy()

        if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing:
            self.lg.warning(f"Strategy update ({self.symbol}): Input index not monotonic DatetimeIndex. Sorting...")
            try: df.sort_index(inplace=True); assert df.index.is_monotonic_increasing
            except: self.lg.error(f"Strategy update failed ({self.symbol}): Could not sort index."); return empty_results

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols): self.lg.error(f"Strategy update failed ({self.symbol}): Missing columns: {[c for c in required_cols if c not in df.columns]}."); return empty_results
        if len(df) < self.min_data_len: self.lg.warning(f"Strategy update ({self.symbol}): Insufficient data ({len(df)} < {self.min_data_len}). Results may be inaccurate.")
        self.lg.debug(f"Starting strategy analysis ({self.symbol}) on {len(df)} candles.")

        try:
            df_float = pd.DataFrame(index=df.index)
            for col in required_cols: df_float[col] = pd.to_numeric(df[col], errors='coerce')
            initial_float_len = len(df_float); df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            if len(df_float) < initial_float_len: self.lg.debug(f"Dropped {initial_float_len - len(df_float)} rows ({self.symbol}) during float conversion.")
            if df_float.empty: self.lg.error(f"Strategy update failed ({self.symbol}): DataFrame empty after float conversion."); return empty_results
        except Exception as e: self.lg.error(f"Strategy update failed ({self.symbol}): Error converting to float: {e}", exc_info=True); return empty_results

        try:
            self.lg.debug(f"Calculating indicators ({self.symbol}) using float data...")
            df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length)
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan)
            valid_trend_comp = df_float['ema2'].notna() & df_float['ema1'].shift(1).notna()
            trend_up_series = pd.Series(np.nan, index=df_float.index, dtype=object); trend_up_series.loc[valid_trend_comp] = df_float['ema2'] > df_float['ema1'].shift(1); trend_up_series.ffill(inplace=True)
            df_float['trend_up'] = trend_up_series.astype('boolean')
            df_float['trend_changed'] = ((df_float['trend_up'] != df_float['trend_up'].shift(1)) & df_float['trend_up'].notna() & df_float['trend_up'].shift(1).notna()).fillna(False).astype(bool)
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan); df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill(); df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()
            atr_mult_float = float(self.vt_atr_multiplier)
            valid_band_calc = df_float['ema1_for_bands'].notna() & df_float['atr_for_bands'].notna() & (df_float['atr_for_bands'] > 0)
            df_float['upper_band'] = np.where(valid_band_calc, df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_mult_float), np.nan)
            df_float['lower_band'] = np.where(valid_band_calc, df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_mult_float), np.nan)
            vol_ema = ta.ema(df_float['volume'].fillna(0.0), length=self.vt_vol_ema_length, fillna=np.nan)
            vol_ema_safe = vol_ema.replace(0, np.nan).fillna(method='bfill').fillna(1e-9)
            df_float['vol_norm'] = (df_float['volume'].fillna(0.0) / vol_ema_safe) * 100.0
            df_float['vol_norm_int'] = df_float['vol_norm'].fillna(0.0).clip(0.0, 200.0).astype(int)
            high_series = df_float['high'] if self.ob_source == "Wicks" else df_float[['open', 'close']].max(axis=1)
            low_series = df_float['low'] if self.ob_source == "Wicks" else df_float[['open', 'close']].min(axis=1)
            df_float['is_ph'] = self._find_pivots(high_series, self.ph_left, self.ph_right, is_high=True)
            df_float['is_pl'] = self._find_pivots(low_series, self.pl_left, self.pl_right, is_high=False)
            self.lg.debug(f"Indicator calculations complete ({self.symbol}).")
        except Exception as e: self.lg.error(f"Strategy update failed ({self.symbol}): Indicator calc error: {e}", exc_info=True); empty_results['dataframe'] = df; return empty_results

        try:
            self.lg.debug(f"Converting indicators back to Decimal ({self.symbol})...")
            indicator_cols_decimal = ['atr', 'ema1', 'ema2', 'upper_band', 'lower_band', 'vol_norm']
            indicator_cols_int = ['vol_norm_int']
            indicator_cols_bool = ['trend_up', 'trend_changed', 'is_ph', 'is_pl']
            for col in indicator_cols_decimal:
                if col in df_float.columns: source_series = df_float[col].reindex(df.index); df[col] = source_series.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
            for col in indicator_cols_int:
                if col in df_float.columns: source_series = df_float[col].reindex(df.index); df[col] = source_series.fillna(0).astype(int)
            for col in indicator_cols_bool:
                 if col in df_float.columns:
                    source_series = df_float[col].reindex(df.index)
                    df[col] = source_series.astype('boolean') if col == 'trend_up' else source_series.fillna(False).astype(bool)
        except Exception as e: self.lg.error(f"Strategy update failed ({self.symbol}): Error converting indicators back: {e}", exc_info=True); empty_results['dataframe'] = df; return empty_results

        initial_len_final = len(df)
        essential_cols = ['close', 'atr', 'trend_up', 'is_ph', 'is_pl']; df.dropna(subset=essential_cols, inplace=True)
        if 'atr' in df.columns: df = df[df['atr'] > Decimal('0')]
        if len(df) < initial_len_final: self.lg.debug(f"Dropped {initial_len_final - len(df)} initial rows ({self.symbol}) due to missing indicators.")
        if df.empty: self.lg.warning(f"Strategy update ({self.symbol}): DataFrame empty after final cleaning."); empty_results['dataframe'] = df; return empty_results
        self.lg.debug(f"Indicators finalized ({self.symbol}). Processing Order Blocks...")

        try:
            new_ob_count = 0; violated_ob_count = 0; extended_ob_count = 0
            last_candle_ts = df.index[-1]
            new_bull_candidates: List[OrderBlock] = []; new_bear_candidates: List[OrderBlock] = []
            existing_ob_ids = {ob['id'] for ob in self.bull_boxes + self.bear_boxes}

            for timestamp, candle in df.iterrows():
                if candle.get('is_ph'):
                    ob_id = f"B_{timestamp.strftime('%y%m%d%H%M%S')}"
                    if ob_id not in existing_ob_ids:
                        ob_top = candle['high'] if self.ob_source == "Wicks" else max(candle['open'], candle['close'])
                        ob_bottom = candle['open'] if self.ob_source == "Wicks" else min(candle['open'], candle['close'])
                        if isinstance(ob_top, Decimal) and isinstance(ob_bottom, Decimal) and ob_top > ob_bottom:
                            new_bear_candidates.append(OrderBlock(id=ob_id, type='bear', timestamp=timestamp, top=ob_top, bottom=ob_bottom, active=True, violated=False, violation_ts=None, extended_to_ts=timestamp))
                            new_ob_count += 1; existing_ob_ids.add(ob_id)
                if candle.get('is_pl'):
                    ob_id = f"L_{timestamp.strftime('%y%m%d%H%M%S')}"
                    if ob_id not in existing_ob_ids:
                        ob_top = candle['open'] if self.ob_source == "Wicks" else max(candle['open'], candle['close'])
                        ob_bottom = candle['low'] if self.ob_source == "Wicks" else min(candle['open'], candle['close'])
                        if isinstance(ob_top, Decimal) and isinstance(ob_bottom, Decimal) and ob_top > ob_bottom:
                            new_bull_candidates.append(OrderBlock(id=ob_id, type='bull', timestamp=timestamp, top=ob_top, bottom=ob_bottom, active=True, violated=False, violation_ts=None, extended_to_ts=timestamp))
                            new_ob_count += 1; existing_ob_ids.add(ob_id)

            self.bull_boxes.extend(new_bull_candidates); self.bear_boxes.extend(new_bear_candidates)
            if new_ob_count > 0: self.lg.debug(f"Identified {new_ob_count} new Order Blocks ({self.symbol}).")

            all_boxes = self.bull_boxes + self.bear_boxes; active_boxes_after_update: List[OrderBlock] = []
            for box in all_boxes:
                 if not box['active']: continue
                 relevant_candles = df[df.index > box['timestamp']]; box_violated = False
                 for ts, candle in relevant_candles.iterrows():
                      close_price = candle.get('close')
                      if isinstance(close_price, Decimal) and close_price.is_finite():
                           violation = (box['type'] == 'bull' and close_price < box['bottom']) or (box['type'] == 'bear' and close_price > box['top'])
                           if violation: box['active'] = False; box['violated'] = True; box['violation_ts'] = ts; violated_ob_count += 1; self.lg.debug(f"{box['type'].capitalize()} OB {box['id']} ({self.symbol}) VIOLATED at {ts.strftime('%H:%M')} by close {close_price.normalize()}"); box_violated = True; break
                           elif self.ob_extend: box['extended_to_ts'] = ts; extended_ob_count += 1
                 if not box_violated and box['active']:
                      if self.ob_extend: box['extended_to_ts'] = last_candle_ts
                      active_boxes_after_update.append(box)

            if violated_ob_count > 0: self.lg.debug(f"Marked {violated_ob_count} OBs as violated ({self.symbol}).")
            self.bull_boxes = [b for b in active_boxes_after_update if b['type'] == 'bull']
            self.bear_boxes = [b for b in active_boxes_after_update if b['type'] == 'bear']
            self.bull_boxes = sorted(self.bull_boxes, key=lambda b: b['timestamp'], reverse=True)[:self.ob_max_boxes]
            self.bear_boxes = sorted(self.bear_boxes, key=lambda b: b['timestamp'], reverse=True)[:self.ob_max_boxes]
            self.lg.debug(f"Pruned active OBs ({self.symbol}). Kept: Bulls={len(self.bull_boxes)}, Bears={len(self.bear_boxes)}")
        except Exception as e: self.lg.error(f"Strategy update failed ({self.symbol}): Error processing OBs: {e}", exc_info=True)

        last_candle_final = df.iloc[-1] if not df.empty else None
        def safe_decimal_from_candle(col_name: str, pos: bool = False) -> Optional[Decimal]:
            if last_candle_final is None: return None; val = last_candle_final.get(col_name)
            if isinstance(val, Decimal) and val.is_finite(): return val if not pos or val > 0 else None
            return None
        def safe_bool_from_candle(col_name: str) -> Optional[bool]:
            if last_candle_final is None: return None; val = last_candle_final.get(col_name); return bool(val) if pd.notna(val) else None
        def safe_int_from_candle(col_name: str) -> Optional[int]:
             if last_candle_final is None: return None; val = last_candle_final.get(col_name); return int(val) if pd.notna(val) else None

        analysis_results = StrategyAnalysisResults(
            dataframe=df, last_close=safe_decimal_from_candle('close') or Decimal('NaN'),
            current_trend_up=safe_bool_from_candle('trend_up'), trend_just_changed=bool(safe_bool_from_candle('trend_changed')),
            active_bull_boxes=self.bull_boxes, active_bear_boxes=self.bear_boxes,
            vol_norm_int=safe_int_from_candle('vol_norm_int'), atr=safe_decimal_from_candle('atr', pos=True),
            upper_band=safe_decimal_from_candle('upper_band'), lower_band=safe_decimal_from_candle('lower_band')
        )

        trend_str = f"{NEON_GREEN}UP{RESET}" if analysis_results['current_trend_up'] is True else f"{NEON_RED}DOWN{RESET}" if analysis_results['current_trend_up'] is False else f"{NEON_YELLOW}Undetermined{RESET}"
        atr_str = f"{analysis_results['atr'].normalize()}" if analysis_results['atr'] else "N/A"
        time_str = last_candle_final.name.strftime('%Y-%m-%d %H:%M:%S %Z') if last_candle_final is not None else "N/A"
        self.lg.debug(f"--- Strategy Analysis Results ({self.symbol} @ {time_str}) ---")
        self.lg.debug(f"  Last Close: {analysis_results['last_close'].normalize() if analysis_results['last_close'].is_finite() else 'NaN'}")
        self.lg.debug(f"  Trend: {trend_str} (Changed Last: {analysis_results['trend_just_changed']})")
        self.lg.debug(f"  ATR: {atr_str}, Vol Norm: {analysis_results['vol_norm_int']}")
        self.lg.debug(f"  Active OBs (Bull/Bear): {len(analysis_results['active_bull_boxes'])} / {len(analysis_results['active_bear_boxes'])}")
        self.lg.debug(f"---------------------------------------------")

        return analysis_results

# --- Signal Generation based on Strategy Results (Refactored into Class) ---
class SignalGenerator:
    """
    Generates trading signals ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD")
    and initial SL/TP levels based on strategy analysis results and current position state.
    """
    def __init__(self, config: Dict[str, Any], market_info: MarketInfo, logger: logging.Logger):
        """
        Initializes the Signal Generator.

        Args:
            config (Dict[str, Any]): Main configuration dictionary.
            market_info (MarketInfo): Standardized market info for the symbol.
            logger (logging.Logger): Logger instance.

        Raises:
            ValueError: If critical configuration parameters are missing or invalid.
        """
        self.config = config
        self.market_info = market_info
        self.logger = logger
        self.lg = logger
        strategy_cfg = config.get("strategy_params", {})
        protection_cfg = config.get("protection", {})

        try:
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg["ob_entry_proximity_factor"]))
            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg["ob_exit_proximity_factor"]))
            self.initial_tp_atr_multiple = Decimal(str(protection_cfg["initial_take_profit_atr_multiple"]))
            self.initial_sl_atr_multiple = Decimal(str(protection_cfg["initial_stop_loss_atr_multiple"]))
            # Get price precision for SL/TP calculation
            self.price_tick = market_info['price_precision_step_decimal']
            if self.price_tick is None or not self.price_tick.is_finite() or self.price_tick <= 0:
                 raise ValueError("Market price precision step (tick size) is invalid.")

            if not (self.ob_entry_proximity_factor >= 1): raise ValueError("ob_entry_proximity_factor must be >= 1.0")
            if not (self.ob_exit_proximity_factor >= 1): raise ValueError("ob_exit_proximity_factor must be >= 1.0")
            if not (self.initial_tp_atr_multiple >= 0): raise ValueError("initial_take_profit_atr_multiple must be >= 0")
            if not (self.initial_sl_atr_multiple > 0): raise ValueError("initial_stop_loss_atr_multiple must be > 0")

            self.lg.info(f"{NEON_CYAN}--- Initializing Signal Generator ({market_info['symbol']}) ---{RESET}")
            self.lg.info(f"  OB Entry Prox: {self.ob_entry_proximity_factor.normalize()}, OB Exit Prox: {self.ob_exit_proximity_factor.normalize()}")
            self.lg.info(f"  Initial SL Mult: {self.initial_sl_atr_multiple.normalize()}, Initial TP Mult: {self.initial_tp_atr_multiple.normalize()}")
            self.lg.info(f"  Price Tick (for SL/TP): {self.price_tick.normalize()}")
            self.lg.info(f"-------------------------------------------")

        except (KeyError, ValueError, InvalidOperation, TypeError) as e:
             self.lg.critical(f"{NEON_RED}FATAL: Error initializing SignalGenerator ({market_info['symbol']}): {e}.{RESET}", exc_info=True)
             raise ValueError(f"SignalGenerator init failed ({market_info['symbol']}): {e}") from e

    def _calculate_initial_sl_tp(self, entry_price: Decimal, side: str, atr: Decimal) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculates initial SL and TP prices based on ATR multiples and market precision."""
        if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0 or \
           not isinstance(atr, Decimal) or not atr.is_finite() or atr <= 0 or side not in ['long', 'short']:
            self.lg.error(f"Invalid inputs for SL/TP calculation: entry={entry_price}, side={side}, atr={atr}")
            return None, None

        # Calculate raw SL price
        sl_dist = atr * self.initial_sl_atr_multiple
        raw_sl_price = entry_price - sl_dist if side == 'long' else entry_price + sl_dist

        # Calculate raw TP price (if enabled)
        raw_tp_price: Optional[Decimal] = None
        if self.initial_tp_atr_multiple > 0:
            tp_dist = atr * self.initial_tp_atr_multiple
            raw_tp_price = entry_price + tp_dist if side == 'long' else entry_price - tp_dist

        # Quantize prices using the utility function
        final_sl = quantize_price(raw_sl_price, self.price_tick, side, is_tp=False)
        final_tp = quantize_price(raw_tp_price, self.price_tick, side, is_tp=True) if raw_tp_price else None

        # Sanity check: ensure SL/TP are still valid after quantization
        if final_sl <= 0: final_sl = None # Should not happen if entry/ATR are positive
        if final_tp is not None and final_tp <= 0: final_tp = None
        # Ensure SL is on loss side and TP is on profit side relative to entry
        if side == 'long':
            if final_sl and final_sl >= entry_price: self.lg.warning(f"Quantized SL {final_sl} >= entry {entry_price} for long. Setting SL=None."); final_sl = None
            if final_tp and final_tp <= entry_price: self.lg.warning(f"Quantized TP {final_tp} <= entry {entry_price} for long. Setting TP=None."); final_tp = None
        elif side == 'short':
            if final_sl and final_sl <= entry_price: self.lg.warning(f"Quantized SL {final_sl} <= entry {entry_price} for short. Setting SL=None."); final_sl = None
            if final_tp and final_tp >= entry_price: self.lg.warning(f"Quantized TP {final_tp} >= entry {entry_price} for short. Setting TP=None."); final_tp = None

        self.lg.debug(f"Calculated Initial SL: {final_sl.normalize() if final_sl else 'None'}, TP: {final_tp.normalize() if final_tp else 'None'} (Raw SL={raw_sl_price.normalize()}, Raw TP={raw_tp_price.normalize() if raw_tp_price else 'N/A'})")
        return final_sl, final_tp

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: Optional[PositionInfo], symbol: str) -> SignalResult:
        """
        Determines the trading signal based on strategy analysis results and the current position state.

        Args:
            analysis_results: Results from VolumaticOBStrategy.update().
            open_position: Standardized PositionInfo dict or None.
            symbol: Trading symbol for logging.

        Returns:
            SignalResult: TypedDict containing the signal, reason, and initial SL/TP if applicable.
        """
        # Default result
        result = SignalResult(signal="HOLD", reason="No condition met", initial_sl=None, initial_tp=None)

        # --- Extract and Validate Essential Data ---
        last_close = analysis_results['last_close']
        trend_up = analysis_results['current_trend_up']
        trend_changed = analysis_results['trend_just_changed']
        atr = analysis_results['atr']
        active_bull_boxes = analysis_results['active_bull_boxes']
        active_bear_boxes = analysis_results['active_bear_boxes']

        if not isinstance(last_close, Decimal) or not last_close.is_finite() or last_close <= 0:
            result['reason'] = "Invalid last close price"
            self.lg.warning(f"Signal ({symbol}): {result['reason']} ({last_close}). Holding.")
            return result
        if trend_up is None: # Trend must be determined
            result['reason'] = "Trend undetermined"
            self.lg.info(f"Signal ({symbol}): {result['reason']}. Holding.")
            return result
        # ATR is needed for potential entry SL/TP calculation
        if atr is None or not atr.is_finite() or atr <= 0:
             # Allow holding/exiting without ATR, but cannot enter
             self.lg.warning(f"Signal ({symbol}): Invalid ATR ({atr}). Cannot calculate entry SL/TP. Only Exit/Hold possible.")
             # Proceed to check exit/hold, but entry will be blocked later if atr is needed

        # --- Position Exists: Check for Exit Signals ---
        if open_position:
            pos_side = open_position.get('side')
            if pos_side not in ['long', 'short']:
                result['reason'] = f"Position exists but side is invalid ({pos_side})"
                self.lg.error(f"Signal ({symbol}): {result['reason']}. Holding.")
                return result

            self.lg.debug(f"Signal ({symbol}): Checking exit conditions for existing {pos_side.upper()} position.")

            # Exit Condition 1: Trend Reversal on Last Candle
            if trend_changed:
                if pos_side == 'long' and not trend_up:
                    result['signal'] = "EXIT_LONG"; result['reason'] = "Trend flipped DOWN"
                    self.lg.info(f"{BRIGHT}{NEON_YELLOW}>>> EXIT LONG Signal ({symbol}): {result['reason']} <<< {RESET}")
                    return result
                elif pos_side == 'short' and trend_up:
                    result['signal'] = "EXIT_SHORT"; result['reason'] = "Trend flipped UP"
                    self.lg.info(f"{BRIGHT}{NEON_YELLOW}>>> EXIT SHORT Signal ({symbol}): {result['reason']} <<< {RESET}")
                    return result

            # Exit Condition 2: Price violates proximity to opposing OB
            try:
                if pos_side == 'long' and active_bear_boxes:
                    # Find nearest active bear box (top price closest to, but above, last close)
                    nearest_bear_box = min(
                        (box for box in active_bear_boxes if box['top'] >= last_close),
                        key=lambda box: box['top'] - last_close, default=None
                    )
                    if nearest_bear_box:
                        exit_threshold = nearest_bear_box['top'] / self.ob_exit_proximity_factor
                        self.lg.debug(f"  Long Exit Check ({symbol}): LastClose={last_close.normalize()}, Nearest Bear OB Top={nearest_bear_box['top'].normalize()}, Exit Threshold (>={exit_threshold.normalize()})")
                        if last_close >= exit_threshold:
                            result['signal'] = "EXIT_LONG"; result['reason'] = f"Price too close to Bear OB {nearest_bear_box['id']}"
                            self.lg.info(f"{BRIGHT}{NEON_YELLOW}>>> EXIT LONG Signal ({symbol}): {result['reason']} <<< {RESET}")
                            return result

                elif pos_side == 'short' and active_bull_boxes:
                    # Find nearest active bull box (bottom price closest to, but below, last close)
                    nearest_bull_box = min(
                        (box for box in active_bull_boxes if box['bottom'] <= last_close),
                        key=lambda box: last_close - box['bottom'], default=None
                    )
                    if nearest_bull_box:
                        exit_threshold = nearest_bull_box['bottom'] * self.ob_exit_proximity_factor
                        self.lg.debug(f"  Short Exit Check ({symbol}): LastClose={last_close.normalize()}, Nearest Bull OB Bottom={nearest_bull_box['bottom'].normalize()}, Exit Threshold (<={exit_threshold.normalize()})")
                        if last_close <= exit_threshold:
                            result['signal'] = "EXIT_SHORT"; result['reason'] = f"Price too close to Bull OB {nearest_bull_box['id']}"
                            self.lg.info(f"{BRIGHT}{NEON_YELLOW}>>> EXIT SHORT Signal ({symbol}): {result['reason']} <<< {RESET}")
                            return result
            except Exception as exit_check_err:
                 self.lg.error(f"Error checking OB exit proximity ({symbol}): {exit_check_err}", exc_info=True)

            # If no exit conditions met
            result['reason'] = f"No exit condition met for {pos_side} position"
            self.lg.debug(f"Signal ({symbol}): {result['reason']}. Holding.")
            return result # Default is HOLD

        # --- No Position Exists: Check for Entry Signals ---
        else:
            self.lg.debug(f"Signal ({symbol}): No open position. Checking entry conditions.")
            # Check ATR validity again, as it's essential for entry SL/TP
            if atr is None:
                 result['reason'] = "ATR invalid, cannot calculate entry SL/TP"
                 self.lg.warning(f"Signal ({symbol}): {result['reason']}. Holding.")
                 return result

            try:
                # Check BUY Entry
                if trend_up and active_bull_boxes:
                    # Find nearest active bull box (bottom closest to, but below, last close)
                    nearest_bull_box = min(
                        (box for box in active_bull_boxes if box['bottom'] <= last_close),
                        key=lambda box: last_close - box['bottom'], default=None
                    )
                    if nearest_bull_box:
                        entry_threshold = nearest_bull_box['top'] * self.ob_entry_proximity_factor
                        self.lg.debug(f"  BUY Entry Check ({symbol}): Trend=UP, LastClose={last_close.normalize()}, Nearest Bull OB Top={nearest_bull_box['top'].normalize()}, Entry Threshold (<={entry_threshold.normalize()})")
                        if last_close <= entry_threshold:
                            sl, tp = self._calculate_initial_sl_tp(last_close, 'long', atr) # Use last_close as potential entry price
                            if sl: # Require valid SL for entry
                                result['signal'] = "BUY"; result['reason'] = f"Trend UP, price near Bull OB {nearest_bull_box['id']}"
                                result['initial_sl'] = sl; result['initial_tp'] = tp
                                self.lg.info(f"{BRIGHT}{NEON_GREEN}>>> BUY Signal ({symbol}): {result['reason']} (SL={sl.normalize() if sl else 'N/A'}, TP={tp.normalize() if tp else 'N/A'}) <<< {RESET}")
                                return result
                            else: self.lg.warning(f"BUY Signal ({symbol}) suppressed: Could not calculate valid initial SL.")
                        # else: self.lg.debug(f"  BUY Condition ({symbol}): Price {last_close.normalize()} > Bull OB Top {nearest_bull_box['top'].normalize()} * Factor {self.ob_entry_proximity_factor.normalize()}")
                    # else: self.lg.debug(f"  BUY Condition ({symbol}): No suitable active Bull OB found below current price.")

                # Check SELL Entry
                elif not trend_up and active_bear_boxes:
                    # Find nearest active bear box (top closest to, but above, last close)
                    nearest_bear_box = min(
                        (box for box in active_bear_boxes if box['top'] >= last_close),
                        key=lambda box: box['top'] - last_close, default=None
                    )
                    if nearest_bear_box:
                        entry_threshold = nearest_bear_box['bottom'] / self.ob_entry_proximity_factor
                        self.lg.debug(f"  SELL Entry Check ({symbol}): Trend=DOWN, LastClose={last_close.normalize()}, Nearest Bear OB Bottom={nearest_bear_box['bottom'].normalize()}, Entry Threshold (>={entry_threshold.normalize()})")
                        if last_close >= entry_threshold:
                            sl, tp = self._calculate_initial_sl_tp(last_close, 'short', atr) # Use last_close as potential entry price
                            if sl: # Require valid SL for entry
                                result['signal'] = "SELL"; result['reason'] = f"Trend DOWN, price near Bear OB {nearest_bear_box['id']}"
                                result['initial_sl'] = sl; result['initial_tp'] = tp
                                self.lg.info(f"{BRIGHT}{NEON_RED}>>> SELL Signal ({symbol}): {result['reason']} (SL={sl.normalize() if sl else 'N/A'}, TP={tp.normalize() if tp else 'N/A'}) <<< {RESET}")
                                return result
                            else: self.lg.warning(f"SELL Signal ({symbol}) suppressed: Could not calculate valid initial SL.")
                        # else: self.lg.debug(f"  SELL Condition ({symbol}): Price {last_close.normalize()} < Bear OB Bottom {nearest_bear_box['bottom'].normalize()} / Factor {self.ob_entry_proximity_factor.normalize()}")
                    # else: self.lg.debug(f"  SELL Condition ({symbol}): No suitable active Bear OB found above current price.")

            except Exception as entry_check_err:
                self.lg.error(f"Error checking OB entry proximity ({symbol}): {entry_check_err}", exc_info=True)

            # If no entry conditions met
            result['reason'] = "Trend/OB conditions not met for entry"
            self.lg.debug(f"Signal ({symbol}): {result['reason']}. Holding.")
            return result # Default is HOLD


# --- Main Execution ---
def _handle_shutdown_signal(signum, frame):
    """Sets the shutdown flag when SIGINT (Ctrl+C) or SIGTERM is received."""
    global _shutdown_requested
    signal_name = signal.Signals(signum).name
    init_logger.warning(f"\n{NEON_RED}{BRIGHT}Shutdown signal ({signal_name}) received. Requesting graceful exit...{RESET}")
    _shutdown_requested = True

def main():
    """Main function to run the trading bot."""
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _handle_shutdown_signal)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, _handle_shutdown_signal) # Handle termination signal (e.g., from systemd or Docker)

    init_logger.info(f"{Fore.MAGENTA}{BRIGHT}--- Starting Pyrmethus Bot v{BOT_VERSION} ---{Style.RESET_ALL}")

    # --- Initialize Exchange ---
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"{NEON_RED}FATAL: Failed to initialize exchange connection. Exiting.{RESET}")
        sys.exit(1)

    # --- Initialize Strategy and Signal Generators for each symbol ---
    strategy_engines = {}
    signal_generators = {}
    active_trading_pairs = [] # List of symbols successfully initialized
    for symbol in CONFIG['trading_pairs']:
        symbol_logger = setup_logger(symbol) # Get or create logger for the symbol
        symbol_logger.info(f"Initializing strategy and signal generator for {symbol}...")
        market_info = get_market_info(exchange, symbol, symbol_logger)
        if not market_info:
            symbol_logger.error(f"Failed to get valid market info for {symbol}. Skipping this symbol.")
            continue # Skip to next symbol if market info fails

        try:
            strategy_engines[symbol] = VolumaticOBStrategy(CONFIG, market_info, symbol_logger)
            signal_generators[symbol] = SignalGenerator(CONFIG, market_info, symbol_logger)
            active_trading_pairs.append(symbol)
            symbol_logger.info(f"Initialization complete for {symbol}.")
        except ValueError as init_err: # Catch initialization errors from classes
            symbol_logger.error(f"Failed to initialize strategy/signal generator for {symbol}: {init_err}. Skipping this symbol.")
            continue # Skip symbol if strategy/signal generator fails init

    if not active_trading_pairs:
        init_logger.critical(f"{NEON_RED}FATAL: No trading pairs were successfully initialized. Check configuration and market availability. Exiting.{RESET}")
        sys.exit(1)

    init_logger.info(f"{NEON_GREEN}Initialization complete for active trading pairs: {active_trading_pairs}{RESET}")
    if not CONFIG.get('enable_trading', False):
        init_logger.warning(f"{NEON_YELLOW}--- TRADING IS DISABLED (enable_trading = false in config.json) ---{RESET}")
        init_logger.warning(f"{NEON_YELLOW}--- Bot will fetch data and generate signals, but WILL NOT place orders. ---{RESET}")

    # --- Main Trading Loop ---
    init_logger.info(f"{Fore.CYAN}### Starting Main Trading Loop ###{Style.RESET_ALL}")
    loop_count = 0
    # Dictionary to store in-memory position state flags (BE/TSL activated) per symbol
    # NOTE: This state is NOT persistent across restarts.
    position_states: Dict[str, Dict[str, bool]] = {sym: {'be_activated': False, 'tsl_activated': False} for sym in active_trading_pairs}

    while not _shutdown_requested:
        loop_count += 1
        init_logger.debug(f"--- Starting Loop Cycle #{loop_count} ---")

        for symbol in active_trading_pairs:
            if _shutdown_requested: break # Check flag again before processing each symbol

            symbol_logger = get_logger_for_symbol(symbol)
            symbol_logger.info(f"--- Processing Symbol: {symbol} (Cycle #{loop_count}) ---")

            try:
                # --- 1. Get Market Info (refresh periodically? Maybe not every cycle) ---
                # Re-fetch market info less frequently unless needed for dynamic changes
                # For now, assume initial market_info is sufficient per cycle
                market_info = get_market_info(exchange, symbol, symbol_logger)
                if not market_info:
                    symbol_logger.warning(f"Could not refresh market info for {symbol} in loop cycle. Skipping this cycle.")
                    continue

                # --- 2. Fetch Kline Data ---
                timeframe_key = CONFIG.get('interval', '5')
                timeframe_ccxt = CCXT_INTERVAL_MAP.get(timeframe_key, '5m') # Default to 5m if invalid
                fetch_limit = CONFIG.get('fetch_limit', DEFAULT_FETCH_LIMIT)
                df_raw = fetch_klines_ccxt(exchange, symbol, timeframe_ccxt, fetch_limit, symbol_logger)
                if df_raw.empty:
                    symbol_logger.warning(f"Kline data fetch returned empty DataFrame for {symbol}. Skipping this cycle.")
                    continue

                # --- 3. Strategy Analysis ---
                strategy = strategy_engines[symbol]
                analysis = strategy.update(df_raw)
                if not analysis or analysis['dataframe'].empty or analysis['last_close'].is_nan():
                    symbol_logger.warning(f"Strategy analysis failed or produced invalid results for {symbol}. Skipping this cycle.")
                    continue

                # --- 4. Check Current Position ---
                current_position = get_open_position(exchange, symbol, market_info, symbol_logger)

                # --- 5. Position Management (If Position Exists) ---
                if current_position:
                    # Retrieve or initialize the in-memory state for this symbol
                    symbol_state = position_states.setdefault(symbol, {'be_activated': False, 'tsl_activated': False})

                    # Update state flags based on current API response (important if bot restarted)
                    symbol_state['tsl_activated'] = bool(current_position.get('trailingStopLoss'))
                    # Note: We can't reliably know from API if BE was *already* activated (just that an SL is set).
                    # The 'be_activated' flag tracks if *this bot instance* set BE.

                    # Pass the current state to the management function
                    manage_existing_position(exchange, symbol, market_info, current_position, analysis, symbol_state, symbol_logger)
                    # Update the main dictionary with potentially modified state from manage_existing_position
                    position_states[symbol] = symbol_state
                else:
                    # If no position, reset the in-memory state for the symbol
                    if symbol in position_states:
                        if position_states[symbol]['be_activated'] or position_states[symbol]['tsl_activated']:
                             symbol_logger.debug(f"Resetting in-memory BE/TSL state for {symbol} as position is closed.")
                        position_states[symbol] = {'be_activated': False, 'tsl_activated': False}

                # --- 6. Generate Trading Signal ---
                generator = signal_generators[symbol]
                # Pass the potentially updated position info (after management checks might refresh it)
                # Re-fetch position *after* potential management actions if needed, or pass the existing one
                # For simplicity now, use the position fetched before management. Management only sets SL/TP/TSL.
                signal_info = generator.generate_signal(analysis, current_position, symbol)

                # --- 7. Execute Trade Action ---
                if CONFIG.get('enable_trading', False):
                    execute_trade_action(exchange, symbol, market_info, current_position, signal_info, analysis, symbol_logger)
                elif signal_info['signal'] != "HOLD":
                    symbol_logger.warning(f"{NEON_YELLOW}TRADING DISABLED:{RESET} Signal '{signal_info['signal']}' generated for {symbol} but not executed. Reason: {signal_info['reason']}")

            except ccxt.AuthenticationError as e:
                symbol_logger.critical(f"{NEON_RED}Authentication Error during main loop for {symbol}: {e}. Stopping bot.{RESET}")
                _shutdown_requested = True # Trigger shutdown
                break # Exit inner loop
            except Exception as e:
                symbol_logger.error(f"{NEON_RED}Critical error in main loop for {symbol}: {e}{RESET}", exc_info=True)
                # Decide whether to continue to next symbol or break/shutdown based on error severity
                # For now, continue to the next symbol after logging the error

            # --- Delay between symbols ---
            if not _shutdown_requested:
                 symbol_delay = CONFIG.get('loop_delay_seconds', LOOP_DELAY_SECONDS)
                 symbol_logger.debug(f"Waiting {symbol_delay}s before next symbol or cycle end.")
                 # Use a loop with smaller sleeps to check shutdown flag more frequently
                 for _ in range(symbol_delay):
                     if _shutdown_requested: break
                     time.sleep(1)
            if _shutdown_requested: break # Exit outer loop if shutdown requested

        # --- End of Cycle ---
        if _shutdown_requested:
             init_logger.info(f"{NEON_YELLOW}Shutdown requested. Exiting main loop.{RESET}")
             break

        init_logger.debug(f"--- Finished Loop Cycle #{loop_count} ---")
        # Optional: Add a longer delay here if needed after processing all symbols

    # --- Bot Shutdown ---
    init_logger.info(f"{Fore.MAGENTA}{BRIGHT}--- Pyrmethus Bot Shutting Down ---{Style.RESET_ALL}")
    # Add any cleanup tasks here if necessary (e.g., close connections, save state)
    init_logger.info("Bot stopped.")
    sys.exit(0)


def manage_existing_position(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, position_info: PositionInfo,
                             analysis_results: StrategyAnalysisResults, position_state: Dict[str, bool], logger: logging.Logger):
    """
    Manages protections (Break-Even, Trailing Stop Loss) for an existing open position.
    Updates the position_state dictionary in place.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol.
        market_info: Standardized market info.
        position_info: Standardized position info of the open position.
        analysis_results: Results from the strategy analysis.
        position_state: In-memory dictionary holding {'be_activated': bool, 'tsl_activated': bool}. Modified in place.
        logger: Logger instance for the symbol.
    """
    lg = logger
    lg.debug(f"Managing existing {position_info['side']} position for {symbol}...")

    # --- Extract necessary data ---
    pos_side = position_info['side']
    entry_price_any = position_info.get('entryPrice')
    atr = analysis_results['atr']
    last_close = analysis_results['last_close']
    be_activated = position_state['be_activated']
    tsl_activated = position_state['tsl_activated'] # Reflects API state + bot's in-memory setting

    # Validate inputs needed for management
    if not isinstance(entry_price_any, Decimal) or not entry_price_any.is_finite() or entry_price_any <= 0:
        lg.warning(f"Position management ({symbol}): Invalid entry price ({entry_price_any}). Cannot manage.")
        return
    entry_price = entry_price_any
    if not isinstance(atr, Decimal) or not atr.is_finite() or atr <= 0:
        lg.warning(f"Position management ({symbol}): Invalid ATR ({atr}). Cannot manage BE/TSL.")
        return
    if not isinstance(last_close, Decimal) or not last_close.is_finite() or last_close <= 0:
        lg.warning(f"Position management ({symbol}): Invalid last close price ({last_close}). Cannot manage.")
        return

    protection_cfg = CONFIG.get('protection', {})
    price_tick = market_info['price_precision_step_decimal']
    if price_tick is None or price_tick <= 0:
        lg.warning(f"Position management ({symbol}): Invalid price tick ({price_tick}). Cannot accurately set BE offset.")
        return # Cannot proceed without valid price tick

    # --- 1. Break-Even (BE) Management ---
    enable_be = protection_cfg.get('enable_break_even', False)
    if enable_be and not be_activated: # Only trigger BE once per position instance (in-memory)
        be_trigger_atr_multiple = Decimal(str(protection_cfg.get('break_even_trigger_atr_multiple', 1.0)))
        be_offset_ticks = int(protection_cfg.get('break_even_offset_ticks', 2))

        # Calculate profit in ATR multiples
        profit_distance = abs(last_close - entry_price)
        profit_in_atr = profit_distance / atr if atr > 0 else Decimal('0')

        lg.debug(f"  BE Check ({symbol}): ProfitATR={profit_in_atr.normalize()} vs TriggerATR={be_trigger_atr_multiple.normalize()}, PriceTick={price_tick.normalize()}, OffsetTicks={be_offset_ticks}")

        # Check if profit target for BE is reached
        if profit_in_atr >= be_trigger_atr_multiple:
            lg.info(f"{NEON_YELLOW}Break-Even Triggered ({symbol}): Profit {profit_in_atr.normalize()} ATR >= Trigger {be_trigger_atr_multiple.normalize()} ATR.{RESET}")

            # Calculate BE stop loss price (entry price +/- offset ticks)
            offset_amount = price_tick * be_offset_ticks
            be_sl_price = entry_price + offset_amount if pos_side == 'long' else entry_price - offset_amount

            # Quantize the BE price just in case offset calculation wasn't exact multiple
            be_sl_price = quantize_price(be_sl_price, price_tick, pos_side, is_tp=False)

            # Check if the calculated BE SL price is valid (e.g., doesn't cross entry incorrectly)
            valid_be_sl = (pos_side == 'long' and be_sl_price > entry_price - price_tick) or \
                          (pos_side == 'short' and be_sl_price < entry_price + price_tick) # Allow BE at or slightly beyond entry

            if valid_be_sl and be_sl_price > 0:
                 lg.warning(f"{BRIGHT}---> Setting Break-Even Stop Loss for {symbol} at {be_sl_price.normalize()} <---{RESET}")
                 # Call the API function to set ONLY the stop loss
                 # Pass 0 for TP and TSL to avoid clearing them if they exist (set-trading-stop modifies all specified params)
                 # We need the current TP to avoid clearing it accidentally. Fetch fresh position info? Or assume TP doesn't change?
                 # Safest: Fetch current TP from position_info if available, otherwise don't set TP.
                 current_tp_str = position_info.get('takeProfitPrice')
                 current_tp_dec = Decimal(current_tp_str) if current_tp_str and _safe_market_decimal(current_tp_str, 'current_tp') else None

                 # Call API to set the new SL, preserving existing TP if possible
                 success = _set_position_protection(
                     exchange=exchange, symbol=symbol, market_info=market_info, position_info=position_info, logger=lg,
                     stop_loss_price=be_sl_price,
                     take_profit_price=current_tp_dec, # Pass current TP to avoid clearing it
                     trailing_stop_distance=Decimal('0'), # Ensure TSL is cleared/not set
                     tsl_activation_price=Decimal('0')
                 )
                 if success:
                     position_state['be_activated'] = True # Mark BE as activated in memory for this run
                     lg.info(f"{NEON_GREEN}Break-Even SL successfully set for {symbol}.{RESET}")
                     # Update position_info in memory? Or rely on next cycle's fetch? For now, rely on next fetch.
                 else:
                     lg.error(f"{NEON_RED}Failed to set Break-Even SL for {symbol} via API.{RESET}")
            else:
                lg.error(f"Break-Even SL calculation resulted in invalid price ({symbol}): {be_sl_price.normalize()}. Cannot set BE.")
        # else: lg.debug(f"  BE condition not met for {symbol}.")

    # --- 2. Trailing Stop Loss (TSL) Management ---
    # Note: TSL activation relies on API call, this only triggers the initial API call.
    enable_tsl = protection_cfg.get('enable_trailing_stop', False)
    # Check if TSL is enabled in config AND TSL has not been activated yet (by API or this bot instance)
    # AND BE has not been activated (typically TSL is used instead of BE, or before BE)
    # Refined logic: Activate TSL if enabled, not yet active, and profit threshold met. Don't check BE state here.
    if enable_tsl and not tsl_activated:
        tsl_activation_perc = Decimal(str(protection_cfg.get('trailing_stop_activation_percentage', 0.003)))
        tsl_callback_rate = Decimal(str(protection_cfg.get('trailing_stop_callback_rate', 0.005)))

        # Calculate current profit percentage
        profit_perc = (last_close - entry_price) / entry_price if pos_side == 'long' else (entry_price - last_close) / entry_price
        lg.debug(f"  TSL Check ({symbol}): Profit%={profit_perc:.4%} vs Activation%={tsl_activation_perc:.4%}, CallbackRate={tsl_callback_rate:.4%}")

        # Check if profit percentage reaches the activation threshold
        if profit_perc >= tsl_activation_perc:
            lg.info(f"{NEON_YELLOW}Trailing Stop Loss Activation Triggered ({symbol}): Profit {profit_perc:.4%} >= Activation {tsl_activation_perc:.4%}.{RESET}")

            # Calculate TSL distance in price units based on the *activation price* (current price)
            # Bybit `trailingStop` param is the distance, `activePrice` triggers it.
            tsl_distance = last_close * tsl_callback_rate # Distance based on current price * callback rate
            # Ensure distance is at least one tick
            tsl_distance = max(tsl_distance, price_tick)

            # Set activation price slightly beyond current price in profit direction to ensure immediate activation if possible
            # Or simply use current price? Bybit docs are a bit ambiguous. Using current price seems safer.
            activation_price = last_close

            # Quantize distance and activation price
            # Distance needs careful quantization - Bybit expects price distance. Quantize to price tick?
            # Let's quantize distance to price tick precision for safety.
            quantized_tsl_distance = quantize_price(tsl_distance, price_tick, pos_side, is_tp=False) # Use SL rounding for distance
            quantized_activation_price = quantize_price(activation_price, price_tick, pos_side, is_tp=True) # Use TP rounding for activation price

            if quantized_tsl_distance > 0 and quantized_activation_price > 0:
                 lg.warning(f"{BRIGHT}---> Activating Trailing Stop Loss for {symbol} with Distance={quantized_tsl_distance.normalize()}, Activation={quantized_activation_price.normalize()} <---{RESET}")
                 # Call the API function to set TSL distance and activation price
                 # Preserve existing TP if possible
                 current_tp_str = position_info.get('takeProfitPrice')
                 current_tp_dec = Decimal(current_tp_str) if current_tp_str and _safe_market_decimal(current_tp_str, 'current_tp') else None

                 # NOTE: Setting active TSL via API will override any existing fixed SL.
                 success = _set_position_protection(
                     exchange=exchange, symbol=symbol, market_info=market_info, position_info=position_info, logger=lg,
                     trailing_stop_distance=quantized_tsl_distance,
                     tsl_activation_price=quantized_activation_price,
                     take_profit_price=current_tp_dec, # Preserve TP
                     stop_loss_price=None # Explicitly don't send fixed SL as TSL overrides it
                 )
                 if success:
                     position_state['tsl_activated'] = True # Mark TSL as activated in memory for this run
                     lg.info(f"{NEON_GREEN}Trailing Stop Loss successfully activated for {symbol} via API.{RESET}")
                 else:
                     lg.error(f"{NEON_RED}Failed to activate Trailing Stop Loss for {symbol} via API.{RESET}")
            else:
                lg.error(f"Trailing Stop Loss calculation resulted in invalid distance/activation price ({symbol}): Dist={quantized_tsl_distance}, Act={quantized_activation_price}. Cannot activate TSL.")
        # else: lg.debug(f"  TSL activation condition not met for {symbol}.")

    # else: lg.debug(f"  TSL already active or disabled for {symbol}.")


def execute_trade_action(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo,
                         current_position: Optional[PositionInfo], signal_info: SignalResult,
                         analysis_results: StrategyAnalysisResults, logger: logging.Logger):
    """
    Executes the trade action based on the generated signal.

    - Handles BUY, SELL, EXIT_LONG, EXIT_SHORT signals.
    - Calculates position size for new entries.
    - Sets initial SL/TP for new entries.
    - Places market orders using `place_trade`.
    - Confirms position entry/exit after placing orders.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol.
        market_info: Standardized market info.
        current_position: Standardized position info (or None).
        signal_info: Result from SignalGenerator.
        analysis_results: Results from strategy analysis.
        logger: Logger instance for the symbol.
    """
    lg = logger
    signal = signal_info['signal']

    # --- Handle Exit Signals ---
    if signal in ["EXIT_LONG", "EXIT_SHORT"]:
        if not current_position:
            lg.warning(f"Received {signal} signal for {symbol}, but no open position found. Ignoring.")
            return
        pos_side = current_position['side']
        pos_size_dec = current_position['size_decimal'] # Already Decimal, positive for long, negative for short

        # Ensure signal matches position side
        if (signal == "EXIT_LONG" and pos_side != 'long') or \
           (signal == "EXIT_SHORT" and pos_side != 'short'):
            lg.error(f"Signal mismatch: Received {signal} but current position is {pos_side} for {symbol}. Ignoring exit.")
            return

        # Calculate exit size (absolute value of current position size)
        exit_size = abs(pos_size_dec)
        if exit_size <= 0:
            lg.error(f"Cannot {signal} for {symbol}: Current position size is zero or invalid ({pos_size_dec}).")
            return

        lg.warning(f"{BRIGHT}>>> Executing {signal} for {symbol} <<<")
        lg.info(f"  Reason: {signal_info['reason']}")
        lg.info(f"  Closing {pos_side} position of size: {exit_size.normalize()}")

        # Place market order to close the position (reduceOnly=True)
        order_result = place_trade(
            exchange=exchange, symbol=symbol, trade_signal=signal, position_size=exit_size,
            market_info=market_info, logger=lg, reduce_only=True
        )

        if order_result:
            lg.info(f"Exit order placed successfully for {symbol}. ID: {order_result.get('id', 'N/A')}")
            # Optional: Wait and confirm position is closed or reduced
            # time.sleep(CONFIG.get('position_confirm_delay_seconds', POSITION_CONFIRM_DELAY_SECONDS))
            # confirmed_position = get_open_position(exchange, symbol, market_info, lg)
            # if confirmed_position: lg.warning(f"Position still open after exit order ({symbol}). Size: {confirmed_position.get('size_decimal', 'N/A')}")
            # else: lg.info(f"{NEON_GREEN}Position successfully closed/reduced for {symbol}.{RESET}")
        else:
            lg.error(f"{NEON_RED}Failed to place {signal} order for {symbol}.{RESET}")

    # --- Handle Entry Signals ---
    elif signal in ["BUY", "SELL"]:
        if current_position:
            lg.warning(f"Received {signal} signal for {symbol}, but a position already exists. Ignoring entry.")
            # TODO: Add logic here if scaling into positions is desired in the future.
            return

        # --- Check if entry is allowed (Trading enabled) ---
        if not CONFIG.get('enable_trading', False):
             lg.warning(f"{NEON_YELLOW}TRADING DISABLED:{RESET} Suppressing {signal} entry for {symbol}.")
             return

        # --- Calculate Position Size ---
        balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if balance is None:
            lg.error(f"Cannot place {signal} order for {symbol}: Failed to fetch balance.")
            return
        if balance <= Decimal('0'):
            lg.error(f"Cannot place {signal} order for {symbol}: Insufficient balance ({balance.normalize()} {QUOTE_CURRENCY}).")
            return

        risk_per_trade = CONFIG.get('risk_per_trade', 0.01)
        entry_price = analysis_results['last_close'] # Use last close as estimated entry for market order
        initial_sl = signal_info['initial_sl']

        if not initial_sl: # Must have a valid SL calculated by signal generator
            lg.error(f"Cannot place {signal} order for {symbol}: Initial Stop Loss calculation failed.")
            return

        position_size = calculate_position_size(
            balance=balance, risk_per_trade=risk_per_trade,
            initial_stop_loss_price=initial_sl, entry_price=entry_price,
            market_info=market_info, exchange=exchange, logger=lg
        )

        if position_size is None or position_size <= Decimal('0'):
            lg.error(f"Cannot place {signal} order for {symbol}: Position sizing failed or resulted in zero/negative size.")
            return

        # --- Set Leverage (Optional) ---
        leverage = CONFIG.get('leverage', 0)
        if market_info.get('is_contract') and leverage > 0:
            lg.info(f"Setting leverage to {leverage}x for {symbol} before entry...")
            if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                lg.warning(f"Failed to set leverage for {symbol}. Proceeding with entry attempt, but leverage might be incorrect.")
                # Continue even if leverage setting fails? Or abort? For now, continue with warning.

        # --- Place Entry Order ---
        lg.warning(f"{BRIGHT}>>> Executing {signal} Entry for {symbol} <<<")
        lg.info(f"  Reason: {signal_info['reason']}")
        lg.info(f"  Calculated Size: {position_size.normalize()}")
        lg.info(f"  Initial SL: {signal_info['initial_sl'].normalize() if signal_info['initial_sl'] else 'N/A'}")
        lg.info(f"  Initial TP: {signal_info['initial_tp'].normalize() if signal_info['initial_tp'] else 'N/A'}")

        order_result = place_trade(
            exchange=exchange, symbol=symbol, trade_signal=signal, position_size=position_size,
            market_info=market_info, logger=lg, reduce_only=False
        )

        # --- Post-Entry Actions (Set SL/TP) ---
        if order_result:
            lg.info(f"Entry order placed successfully for {symbol}. ID: {order_result.get('id', 'N/A')}")

            # Wait briefly for position to reflect on the exchange
            confirm_delay = CONFIG.get('position_confirm_delay_seconds', POSITION_CONFIRM_DELAY_SECONDS)
            lg.debug(f"Waiting {confirm_delay}s to confirm position and set SL/TP for {symbol}...")
            time.sleep(confirm_delay)

            # Fetch the newly opened position details
            new_position = get_open_position(exchange, symbol, market_info, lg)

            if new_position:
                lg.info(f"Confirmed {new_position['side']} position opened for {symbol}. Size: {new_position['size_decimal'].normalize()}, Entry: {new_position.get('entryPrice', 'N/A')}")

                # Set initial SL and TP using the prices calculated by the signal generator
                initial_tp = signal_info['initial_tp']
                lg.info(f"Setting initial protection for {symbol}: SL={initial_sl.normalize()}, TP={initial_tp.normalize() if initial_tp else 'None'}")
                sl_tp_set_success = _set_position_protection(
                    exchange=exchange, symbol=symbol, market_info=market_info, position_info=new_position, logger=lg,
                    stop_loss_price=initial_sl,
                    take_profit_price=initial_tp # Pass None if TP disabled or calc failed
                )
                if sl_tp_set_success:
                    lg.info(f"{NEON_GREEN}Initial SL/TP set successfully for {symbol}.{RESET}")
                else:
                    lg.error(f"{NEON_RED}Failed to set initial SL/TP for {symbol} after entry.{RESET}")
                    # Consider closing the position immediately if SL/TP setting fails? Risky. Log error for now.
            else:
                lg.error(f"{NEON_RED}Failed to confirm open position for {symbol} after placing entry order.{RESET}")
                # Order might have failed silently or filled unexpectedly? Manual check required.
        else:
            lg.error(f"{NEON_RED}Failed to place {signal} entry order for {symbol}.{RESET}")

    # --- Handle HOLD Signal ---
    elif signal == "HOLD":
        lg.info(f"Signal ({symbol}): HOLD. Reason: {signal_info['reason']}")
        # No action needed
    else:
        lg.error(f"Unknown signal '{signal}' received for {symbol}. Ignoring.")


# --- Run the Bot ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # This is handled by the signal handler, but catch here just in case
        init_logger.info("KeyboardInterrupt caught in __main__. Exiting.")
        sys.exit(0)
    except Exception as global_err:
        # Catch any unexpected errors that weren't handled elsewhere
        init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL UNHANDLED EXCEPTION:{RESET} {global_err}", exc_info=True)
        sys.exit(1)

