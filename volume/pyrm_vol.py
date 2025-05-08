```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pyrmethus_volumatic_bot.py
# Unified and Enhanced Trading Bot incorporating Volumatic Trend, Pivot Order Blocks,
# and advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Merged and improved from multiple source files.
# Version 1.6.0: Completed core trading logic structure, CCXT helpers, position management,
#                 robust error handling, notifications, and graceful shutdown based on
#                 analysis of provided code snippets. Placeholder strategy included.

"""
Pyrmethus Volumatic Bot: A Python Trading Bot for Bybit V5

This bot implements a trading strategy based on the combination of:
1.  **Volumatic Trend:** An EMA/SWMA crossover system with ATR-based bands,
    incorporating normalized volume analysis. (Placeholder logic included - REQUIRES IMPLEMENTATION)
2.  **Pivot Order Blocks (OBs):** Identifying potential support/resistance zones
    based on pivot highs and lows derived from candle wicks or bodies. (Placeholder logic included - REQUIRES IMPLEMENTATION)

This version synthesizes features and robustness from previous iterations, including:
-   Robust configuration loading from both .env (secrets) and config.json (parameters).
-   Detailed configuration validation with automatic correction to defaults.
-   Flexible notification options (Termux SMS and Email) with improved error handling.
-   Enhanced logging with colorama, rotation, sensitive data redaction, and timezone support.
-   Comprehensive API interaction handling with retries and specific error logging for CCXT.
-   Accurate Decimal usage for all financial calculations.
-   Structured data types using TypedDicts for clarity.
-   Implementation of native Bybit V5 Stop Loss, Take Profit, and Trailing Stop Loss where possible
    via CCXT order placement/editing (leveraging exchange-specific parameters).
-   Logic for managing Break-Even stop adjustments.
-   Support for multiple trading pairs (processed sequentially per cycle).
-   Graceful shutdown on interruption or critical errors.
-   Basic in-memory state management for Break-Even and Trailing Stop activation per symbol.

Disclaimer:
- **EXTREME RISK**: Trading cryptocurrencies, especially futures contracts with leverage and automated systems, involves substantial risk of financial loss. This script is provided for EDUCATIONAL PURPOSES ONLY. You could lose your entire investment and potentially more. Use this software entirely at your own risk. The authors and contributors assume NO responsibility for any trading losses.
- **NATIVE SL/TP/TSL RELIANCE**: The bot's protective stop mechanisms rely heavily on Bybit's exchange-native order execution. Their performance is subject to exchange conditions, potential slippage during volatile periods, API reliability, order book liquidity, and specific exchange rules (e.g., trigger prices like Mark, Index, Last). These orders are NOT GUARANTEED to execute at the precise trigger price specified.
- **PARAMETER SENSITIVITY & OPTIMIZATION**: The performance of this bot is HIGHLY dependent on the chosen strategy parameters (indicator settings, risk levels, SL/TP/TSL percentages, filter thresholds). These parameters require extensive backtesting, optimization, and forward testing on a TESTNET environment before considering any live deployment. Default parameters are unlikely to be profitable and serve only as examples.
- **API RATE LIMITS & BANS**: Excessive API requests can lead to temporary or permanent bans from the exchange. Monitor API usage and adjust script timing (loop_delay_seconds) accordingly. CCXT's built-in rate limiter is enabled but may not prevent all issues under heavy load or rapid market conditions.
- **SLIPPAGE**: Market orders, used for entry and potentially for SL/TP/TSL execution by the exchange, are susceptible to slippage. This means the actual execution price may differ from the price observed when the order was placed, especially during high volatility or low liquidity.
- **TEST THOROUGHLY**: **DO NOT RUN THIS SCRIPT WITH REAL FUNDS WITHOUT EXTENSIVE AND SUCCESSFUL TESTING ON A TESTNET OR DEMO ACCOUNT.** Ensure you fully understand every part of the code, its logic (especially the placeholder strategy sections), and its potential risks before any live deployment. The strategy logic provided is a PLACEHOLDER and requires significant development and testing.
"""

# --- Standard Library Imports ---
import hashlib # Used for HMAC signature if needed (CCXT handles this usually)
import hmac    # Used for HMAC signature if needed (CCXT handles this usually)
import json
import logging
import math
import os
import re
import signal
import sys
import time
import subprocess # Used for Termux SMS
import shutil     # Used to check for termux-sms-send command availability
import smtplib    # Used for Email notifications
import traceback  # Used for logging detailed exceptions
from email.mime.text import MIMEText # Used for Email notifications
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

# --- Timezone Handling ---
# Attempt to import the standard library's zoneinfo (Python 3.9+)
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    # Ensure tzdata is installed for non-UTC timezones with zoneinfo
    try:
        # Attempt to load a common non-UTC zone to check if tzdata is available
        _ = ZoneInfo("UTC") # Check UTC first as it's always available
        _ = ZoneInfo("America/New_York") # Then check a common non-UTC zone
        _ZONEINFO_AVAILABLE = True
    except ZoneInfoNotFoundError:
        _ZONEINFO_AVAILABLE = False
        # Use print here as logger might not be ready
        print(f"\033[93mWarning: 'zoneinfo' is available, but 'tzdata' package seems missing or corrupt.\033[0m") # Yellow text
        print(f"\033[93m         `pip install tzdata` is recommended for reliable timezone support.\033[0m")
    except Exception as tz_init_err:
         _ZONEINFO_AVAILABLE = False
         # Catch any other unexpected errors during ZoneInfo initialization
         print(f"\033[93mWarning: Error initializing test timezone with 'zoneinfo': {tz_init_err}\033[0m")
except ImportError:
    _ZONEINFO_AVAILABLE = False
    # Fallback for older Python versions or if zoneinfo itself is not installed
    print(f"\033[93mWarning: 'zoneinfo' module not found (requires Python 3.9+). Falling back to basic UTC implementation.\033[0m")
    print(f"\033[93m         For accurate local time logging, upgrade Python or use a backport library (`pip install backports.zoneinfo`).\033[0m")

# Basic UTC fallback implementation mimicking the zoneinfo.ZoneInfo interface if zoneinfo is not available
if not _ZONEINFO_AVAILABLE:
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
                print(f"\033[93mWarning: Fallback ZoneInfo initialized with key '{key}', but will always use UTC.\033[0m")
            # Store the requested key for representation, though internally we always use UTC
            self._requested_key = key

        def __call__(self, dt: Optional[datetime] = None) -> Optional[datetime]:
            """Attaches UTC timezone info to a datetime object. Returns None if input is None."""
            if dt is None:
                return None
            # If naive, replace tzinfo with UTC. If already aware, convert to UTC.
            return dt.astimezone(timezone.utc)

        def fromutc(self, dt: datetime) -> datetime:
            """Converts a UTC datetime to this timezone (which is always UTC in the fallback)."""
            if not isinstance(dt, datetime):
                raise TypeError("fromutc() requires a datetime argument")
            if dt.tzinfo is None:
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
import numpy as np # Numerical operations
import pandas as pd # Data manipulation and analysis
import pandas_ta as ta  # Technical Analysis library (requires pandas, numpy)

# API & Networking
import requests         # For HTTP requests (often used implicitly by ccxt)
import ccxt             # Crypto Exchange Trading Library

# Utilities
from colorama import Fore, Style, init as colorama_init # Colored console output
from dotenv import load_dotenv                        # Load environment variables from .env file

# --- Initial Setup ---
# Set Decimal precision globally for accurate financial calculations
# 28 digits is often sufficient for most crypto pairs, but verify based on specific needs.
getcontext().prec = 28
# Initialize Colorama for cross-platform colored output (autoreset ensures color ends after each print)
colorama_init(autoreset=True)
# Load environment variables from a .env file if it exists in the current directory or parent directories.
load_dotenv()

# --- Global Configuration & Secrets (Loaded Later) ---
# API Credentials will be loaded from .env file
API_KEY: Optional[str] = os.getenv("BYBIT_API_KEY")
API_SECRET: Optional[str] = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Use print directly here as logger might not be fully initialized yet.
    print(f"{Fore.RED}{Style.BRIGHT}FATAL ERROR: BYBIT_API_KEY and/or BYBIT_API_SECRET environment variables are missing.{Style.RESET_ALL}")
    print(f"{Fore.RED}Please ensure they are set in your system environment or in a '.env' file in the bot's directory.{Style.RESET_ALL}")
    print(f"{Fore.RED}The bot cannot authenticate with the exchange and will now exit.{Style.RESET_ALL}")
    sys.exit(1) # Exit immediately if credentials are missing

# Notification Settings (Loaded from .env file)
SMTP_SERVER: Optional[str] = os.getenv("SMTP_SERVER")
SMTP_PORT_STR: Optional[str] = os.getenv("SMTP_PORT")
SMTP_PORT: int = 587 # Default SMTP port
if SMTP_PORT_STR and SMTP_PORT_STR.isdigit():
    SMTP_PORT = int(SMTP_PORT_STR)
elif SMTP_PORT_STR:
    print(f"{Fore.YELLOW}Warning: Invalid SMTP_PORT value '{SMTP_PORT_STR}' in .env file. Using default port {SMTP_PORT}.{Style.RESET_ALL}")
SMTP_USER: Optional[str] = os.getenv("SMTP_USER")
SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")
NOTIFICATION_EMAIL_RECIPIENT: Optional[str] = os.getenv("NOTIFICATION_EMAIL_RECIPIENT") # Renamed from NOTIFICATION_EMAIL
TERMUX_SMS_RECIPIENT: Optional[str] = os.getenv("TERMUX_SMS_RECIPIENT") # Renamed from SMS_RECIPIENT_NUMBER

# --- Constants ---
BOT_VERSION = "1.6.0"

# --- Configuration File & Logging ---
CONFIG_FILE: str = "config.json"    # Name of the configuration file
LOG_DIRECTORY: str = "bot_logs"     # Directory to store log files

# --- Timezone Configuration ---
DEFAULT_TIMEZONE_STR: str = "UTC" # Default timezone if not specified elsewhere (Safer to default to UTC)
# Prioritize TIMEZONE from .env, fallback to the default defined above.
TIMEZONE_STR: str = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)
try:
    # Attempt to load user-specified timezone, fallback to UTC if tzdata is not installed or invalid
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

# --- API & Timing Constants (Defaults, overridden by config.json) ---
MAX_API_RETRIES: int = 5           # Max number of retries for most failed API calls (excluding rate limits)
RETRY_DELAY_SECONDS: int = 5       # Initial delay in seconds between retries (often increased exponentially)
POSITION_CONFIRM_DELAY_SECONDS: int = 10 # Delay after placing order to confirm position status (allows exchange processing time)
LOOP_DELAY_SECONDS: int = 30       # Base delay in seconds between main loop cycles (per symbol) - Increased slightly for stability
BYBIT_API_KLINE_LIMIT: int = 1000  # Max klines per Bybit V5 API request (important constraint for fetch_klines_ccxt)

# --- Data & Strategy Constants ---
# Bybit API requires specific strings for intervals in some calls.
# config.json uses the keys from VALID_INTERVALS, CCXT calls use values from CCXT_INTERVAL_MAP.
VALID_INTERVALS: List[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP: Dict[str, str] = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
DEFAULT_FETCH_LIMIT: int = 750     # Default number of klines to fetch historically if not specified in config
MAX_DF_LEN: int = 2000             # Maximum length of DataFrame to keep in memory (helps prevent memory bloat over time)
MIN_KLINES_FOR_STRATEGY: int = 250 # Minimum number of klines required to calculate strategy indicators reliably (e.g., VT_ATR_PERIOD + VT_LENGTH + safety margin)

# Default Strategy Parameters (Volumatic Trend + Order Blocks). Overridden by config.json
DEFAULT_VT_LENGTH: int = 40
DEFAULT_VT_ATR_PERIOD: int = 200
DEFAULT_VT_VOL_EMA_LENGTH: int = 950 # Placeholder: Lookback for Volume EMA/SWMA
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0 # Placeholder: ATR multiplier for VT bands
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0 # Placeholder: Parameter mentioned in original code, usage unclear in placeholder logic.

# Default Order Block (OB) parameters. Overridden by config.json
DEFAULT_OB_SOURCE: str = "Wicks"    # Source for defining OB price levels: "Wicks" (High/Low) or "Body" (Open/Close)
DEFAULT_PH_LEFT: int = 10           # Pivot High lookback periods
DEFAULT_PH_RIGHT: int = 10          # Pivot High lookforward periods
DEFAULT_PL_LEFT: int = 10           # Pivot Low lookback periods
DEFAULT_PL_RIGHT: int = 10          # Pivot Low lookforward periods
DEFAULT_OB_EXTEND: bool = True      # Whether to visually extend OB boxes until violated
DEFAULT_OB_MAX_BOXES: int = 50      # Maximum number of active Order Blocks to track per side (bull/bear)
DEFAULT_OB_ENTRY_PROXIMITY_FACTOR: float = 1.005 # Price must be within this factor of OB edge for entry
DEFAULT_OB_EXIT_PROXIMITY_FACTOR: float = 1.001 # Price must be within this factor of OB edge for exit/invalidation

# Default Protection Parameters. Overridden by config.json
DEFAULT_INITIAL_STOP_LOSS_ATR_MULTIPLE: float = 1.8 # Initial SL distance = ATR * this multiple
DEFAULT_INITIAL_TAKE_PROFIT_ATR_MULTIPLE: float = 0.7 # Initial TP distance = ATR * this multiple (0 to disable)
DEFAULT_ENABLE_BREAK_EVEN: bool = True          # Enable moving SL to break-even?
DEFAULT_BREAK_EVEN_TRIGGER_ATR_MULTIPLE: float = 1.0 # Move SL to BE when price moves ATR * multiple in profit
DEFAULT_BREAK_EVEN_OFFSET_TICKS: int = 2        # Offset SL from entry by this many price ticks for BE
DEFAULT_ENABLE_TRAILING_STOP: bool = True       # Enable Trailing Stop Loss?
DEFAULT_TRAILING_STOP_CALLBACK_RATE: float = 0.005 # TSL callback/distance (e.g., 0.005 = 0.5%)
DEFAULT_TRAILING_STOP_ACTIVATION_PERCENTAGE: float = 0.003 # Activate TSL when price moves this % from entry

# Dynamically loaded from config: QUOTE_CURRENCY (e.g., "USDT")
QUOTE_CURRENCY: str = "USDT" # Placeholder, will be updated by load_config()

# Minimum order value (USD) for Bybit Perpetual contracts. Used in quantity calculation sanity check.
# This is a common constraint on Bybit V5 for USDT perpetuals.
MIN_ORDER_VALUE_USDT: Decimal = Decimal("1.0")

# Threshold for considering a position size or price difference as zero/negligible
POSITION_QTY_EPSILON: Decimal = Decimal("1e-9") # Small value to compare quantities against zero
PRICE_EPSILON: Decimal = Decimal("1e-9") # Small value for comparing prices

# --- UI Constants (Colorama Foregrounds and Styles) ---
# Define constants for frequently used colors and styles for console output.
NEON_GREEN: str = Fore.LIGHTGREEN_EX
NEON_BLUE: str = Fore.CYAN
NEON_PURPLE: str = Fore.MAGENTA
NEON_YELLOW: str = Fore.YELLOW
NEON_RED: str = Fore.LIGHTRED_EX
NEON_CYAN: str = Fore.CYAN         # Duplicate of NEON_BLUE, kept for potential compatibility
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
_termux_sms_command_exists: Optional[bool] = None # Cache the result of checking termux-sms-send existence

# --- Type Definitions (Enhanced with Docstrings and Detail) ---
class OrderBlock(TypedDict):
    """Represents an identified Order Block (OB) zone."""
    id: str                 # Unique identifier (e.g., "BULL_1678886400000_Wicks")
    type: str               # "BULL" (demand zone) or "BEAR" (supply zone)
    timestamp: pd.Timestamp # Timestamp of the candle defining the block (UTC, pandas Timestamp for convenience)
    top: Decimal            # Top price level (highest point) of the block
    bottom: Decimal         # Bottom price level (lowest point) of the block
    active: bool            # Is the block currently considered active (not violated by fetched data)?
    violated: bool          # Has the block been violated by subsequent price action in the fetched data?
    violation_ts: Optional[pd.Timestamp] # Timestamp when the violation occurred (UTC, pandas Timestamp)

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
    vol_norm_int: Optional[int] # Placeholder: Normalized volume indicator value (e.g., 0-100), if used by strategy
    atr: Optional[Decimal]    # Current Average True Range value from the last candle (last value in the ATR series)
    upper_band: Optional[Decimal] # Placeholder: Upper band value from the last candle (e.g., from Volumatic Trend)
    lower_band: Optional[Decimal] # Placeholder: Lower band value from the last candle (e.g., from Volumatic Trend)

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
    contracts: Optional[Any] # Number of contracts (legacy/float). Use size_decimal derived field instead for precision.
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
    size_decimal: Decimal        # Position size as Decimal (positive, derived from 'contracts' or 'info')
    entryPrice_decimal: Optional[Decimal] # Entry price as Decimal
    markPrice_decimal: Optional[Decimal] # Mark price as Decimal
    liquidationPrice_decimal: Optional[Decimal] # Liquidation price as Decimal
    leverage_decimal: Optional[Decimal] # Leverage as Decimal
    unrealizedPnl_decimal: Optional[Decimal] # Unrealized PnL as Decimal
    notional_decimal: Optional[Decimal] # Notional value (position value in quote currency) as Decimal
    collateral_decimal: Optional[Decimal] # Collateral (margin used) as Decimal
    initialMargin_decimal: Optional[Decimal] # Initial margin as Decimal
    maintenanceMargin_decimal: Optional[Decimal] # Maintenance margin as Decimal
    # --- Protection Order Status (Extracted from 'info' or root, strings often returned by Exchange API) ---
    # Note: These are the RAW string values returned by the API, need parsing to Decimal.
    # CCXT unified fields may not capture all exchange nuances, rely on 'info' if needed.
    stopLossPrice_raw: Optional[str] # Current stop loss price set on the exchange (as string, often '0' or '0.0' if not set)
    takeProfitPrice_raw: Optional[str] # Current take profit price set on the exchange (as string, often '0' or '0.0' if not set)
    # Bybit V5 'trailingStop' in position info is the current trigger price once active, not the distance/callback rate.
    # The 'activePrice' is the activation price. We store both raw and parsed.
    trailingStopPrice_raw: Optional[str] # Current trailing stop trigger price (as string, e.g., Bybit V5 'trailingStop')
    tslActivationPrice_raw: Optional[str] # Trailing stop activation price (as string, e.g., Bybit V5 'activePrice')
    # --- Parsed Protection Order Status (Decimal) ---
    stopLossPrice_dec: Optional[Decimal]
    takeProfitPrice_dec: Optional[Decimal]
    trailingStopPrice_dec: Optional[Decimal] # TSL trigger price
    tslActivationPrice_dec: Optional[Decimal] # TSL activation price
    # --- Bot State Tracking (Managed internally by the bot logic, reflects *bot's* knowledge/actions on this position instance) ---
    # These are added dynamically when fetched and linked to bot state.
    be_activated: bool           # Has the break-even logic been triggered and successfully executed *by the bot* for this position instance?
    tsl_activated: bool          # Has the trailing stop loss been activated *by the bot* (based on price check) or detected as active on the exchange?

class SignalResult(TypedDict):
    """Represents the outcome of the signal generation process based on strategy analysis."""
    signal: str              # The generated signal: "BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT", "NONE" (no actionable signal)
    reason: str              # A human-readable explanation for why the signal was generated
    initial_sl_price: Optional[Decimal] # Calculated initial stop loss price if the signal is for a new entry ("BUY" or "SELL")
    initial_tp_price: Optional[Decimal] # Calculated initial take profit price if the signal is for a new entry ("BUY" or "SELL")

# --- Logging Setup ---
class SensitiveFormatter(logging.Formatter):
    """
    Custom log formatter that redacts sensitive API keys, secrets, and recipient info
    from log messages. Inherits from `logging.Formatter` and overrides the `format` method.
    """
    _api_key_placeholder = "***API_KEY_REDACTED***"
    _api_secret_placeholder = "***API_SECRET_REDACTED***"
    _email_recipient_placeholder = "***EMAIL_RECIPIENT_REDACTED***"
    _sms_recipient_placeholder = "***SMS_RECIPIENT_REDACTED***"
    _smtp_password_placeholder = "***SMTP_PASSWORD_REDACTED***"

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record using the parent class, then redacts sensitive strings.

        Args:
            record: The logging.LogRecord to format.

        Returns:
            The formatted log message string with sensitive data redacted.
        """
        formatted_msg = super().format(record)
        redacted_msg = formatted_msg

        # Perform redaction only if the sensitive variables are set and are strings
        try:
            # Redact API Key
            if API_KEY and isinstance(API_KEY, str) and len(API_KEY) > 4:
                redacted_msg = redacted_msg.replace(API_KEY, self._api_key_placeholder)
            # Redact API Secret
            if API_SECRET and isinstance(API_SECRET, str) and len(API_SECRET) > 4:
                redacted_msg = redacted_msg.replace(API_SECRET, self._api_secret_placeholder)
            # Redact Email Recipient (if used in logs, e.g., in send_notification)
            if NOTIFICATION_EMAIL_RECIPIENT and isinstance(NOTIFICATION_EMAIL_RECIPIENT, str):
                 redacted_msg = redacted_msg.replace(NOTIFICATION_EMAIL_RECIPIENT, self._email_recipient_placeholder)
            # Redact SMS Recipient (if used in logs, e.g., in send_notification)
            if TERMUX_SMS_RECIPIENT and isinstance(TERMUX_SMS_RECIPIENT, str):
                 redacted_msg = redacted_msg.replace(TERMUX_SMS_RECIPIENT, self._sms_recipient_placeholder)
            # Redact SMTP Password (unlikely to be in standard logs, but defensive)
            if SMTP_PASSWORD and isinstance(SMTP_PASSWORD, str) and len(SMTP_PASSWORD) > 4: # Check length too
                 redacted_msg = redacted_msg.replace(SMTP_PASSWORD, self._smtp_password_placeholder)

        except Exception as e:
            # Prevent crashing the application if redaction fails unexpectedly.
            # Use print as logging itself might be the source of the issue.
            print(f"WARNING: Error during log message redaction: {e}", file=sys.stderr)
            return formatted_msg # Return original in case of error
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
        super().__init__(fmt=self._log_format, datefmt=self._date_format, **kwargs)
        # Set the time converter to use the globally configured TIMEZONE for local time display.
        # This lambda converts the timestamp (usually Unix epoch float) to a timetuple in the local zone.
        self.converter = lambda timestamp: datetime.fromtimestamp(timestamp, tz=TIMEZONE).timetuple()

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
        name: The name for the logger (e.g., 'init', 'main', 'BTC_USDT').
              Used in log messages and to generate the log filename.

    Returns:
        A configured logging.Logger instance ready for use.
    """
    # Sanitize the logger name for safe use in filenames and hierarchical logging
    # Replace /, :, and spaces with underscores. Limit length for safety.
    safe_filename_part = re.sub(r'[^\w\-.]', '_', name)[:50] # Limit length
    # Use dot notation for logger names to support potential hierarchical logging features.
    logger_name = f"pyrmethus.{safe_filename_part}"
    # Construct the full path for the log file.
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")

    logger = logging.getLogger(logger_name)

    # Prevent adding handlers multiple times if the logger was already configured
    if logger.hasHandlers():
        # If handlers exist, assume configuration is stable and return the existing logger.
        # Optional: Could re-check and update console log level here if needed dynamically.
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
            # Use print here, as logger might not be fully configured yet.
            print(f"{NEON_YELLOW}Warning: Invalid CONSOLE_LOG_LEVEL '{console_log_level_str}'. Defaulting to INFO.{RESET}")
            log_level = logging.INFO
        sh.setLevel(log_level) # Set the minimum level for console output.

        # Use the custom NeonConsoleFormatter for colors, local time, and redaction.
        console_formatter = NeonConsoleFormatter()
        sh.setFormatter(console_formatter)
        logger.addHandler(sh)
    except Exception as e:
        # Log setup errors should be visible immediately.
        print(f"{NEON_RED}{BRIGHT}Error setting up console logger: {e}{RESET}")

    # Prevent messages logged by this logger from propagating up to the root logger,
    # which might have its own handlers (e.g., default StreamHandler) causing duplicate output.
    logger.propagate = False

    # Using logger instance to log setup status, check if handlers are actually added
    if logger.handlers:
         # Check if fh and sh were successfully created and added
         file_handler_level = next((h.level for h in logger.handlers if isinstance(h, RotatingFileHandler)), 'N/A')
         console_handler_level = next((h.level for h in logger.handlers if isinstance(h, logging.StreamHandler)), 'N/A')
         # Use debug level for this initialization message
         logger.debug(f"Logger '{logger_name}' initialized. File Handler Level: {logging.getLevelName(file_handler_level) if isinstance(file_handler_level, int) else file_handler_level}, Console Handler Level: {logging.getLevelName(console_handler_level) if isinstance(console_handler_level, int) else console_handler_level}")
    else:
         # Use print if logger failed completely
         print(f"{NEON_RED}{BRIGHT}Warning: Logger '{logger_name}' initialized but no handlers were successfully added.{RESET}")

    return logger

# --- Initial Logger Setup ---
# Create a logger instance specifically for the initialization phase.
init_logger = setup_logger("init")
init_logger.info(f"{Fore.MAGENTA}{BRIGHT}===== Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing ====={Style.RESET_ALL}")
init_logger.info(f"Using Timezone for Console Logs: {TIMEZONE_STR} ({TIMEZONE})")
init_logger.debug(f"Decimal Precision Set To: {getcontext().prec}")
# Remind user about dependencies, helpful for troubleshooting setup issues.
init_logger.debug("Ensure required packages are installed: pandas, pandas_ta, numpy, ccxt, requests, python-dotenv, colorama, tzdata (recommended), smtplib (built-in), email (built-in)")

# --- Notification Setup ---
def send_notification(subject: str, body: str, logger: logging.Logger, notification_type: Optional[str] = None) -> bool:
    """
    Sends a notification via email or Termux SMS if configured and enabled.

    If `notification_type` is None, it attempts to use the type specified in the config.
    If notification type in config is invalid or not set, it logs a warning and does nothing.

    Args:
        subject (str): The subject line for the notification (used for email and prefixed to SMS).
        body (str): The main content of the notification.
        logger (logging.Logger): Logger instance to use for logging notification status/errors.
        notification_type (Optional[str]): Override config type ('email' or 'sms'). If None, uses config.

    Returns:
        bool: True if the notification was attempted successfully (not necessarily received), False otherwise.
    """
    lg = logger # Alias for convenience

    # Check master notification enable/disable from config (assuming CONFIG is loaded globally)
    if 'CONFIG' not in globals() or not CONFIG.get("notifications", {}).get("enable_notifications", False):
        lg.debug(f"Notifications are disabled by config. Skipping notification: '{subject}'")
        return False

    # Determine the type of notification to send
    notify_type: str
    if notification_type is None:
        configured_type = CONFIG.get("notifications", {}).get("notification_type", "").lower() # Default to empty string if missing
        if configured_type not in ["email", "sms"]:
             lg.warning(f"{NEON_YELLOW}Invalid 'notification_type' '{configured_type}' specified in config. Must be 'email' or 'sms'. Notification not sent.{RESET}")
             return False
        notify_type = configured_type
    else:
        notify_type = notification_type.lower()
        if notify_type not in ["email", "sms"]:
             lg.error(f"{NEON_RED}Invalid notification_type argument passed to send_notification: '{notification_type}'. Must be 'email' or 'sms'. Notification not sent.{RESET}")
             return False

    # --- Email Notification ---
    if notify_type == "email":
        # Check if email settings are complete from .env
        if not all([SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_EMAIL_RECIPIENT]):
            lg.warning("Email notification selected but .env settings are incomplete (SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_EMAIL_RECIPIENT env vars). Email not sent.")
            return False

        try:
            msg = MIMEText(body)
            msg['Subject'] = f"[Pyrmethus Bot] {subject}" # Prefix subject for clarity
            msg['From'] = SMTP_USER
            msg['To'] = NOTIFICATION_EMAIL_RECIPIENT

            lg.debug(f"Attempting to send email notification to {NOTIFICATION_EMAIL_RECIPIENT} via {SMTP_SERVER}:{SMTP_PORT}...")
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.ehlo() # Greet server
                server.starttls() # Upgrade connection to secure
                server.ehlo() # Greet again after TLS
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, NOTIFICATION_EMAIL_RECIPIENT, msg.as_string())
            lg.info(f"{NEON_GREEN}Successfully sent email notification: '{subject}'{RESET}")
            return True
        except smtplib.SMTPAuthenticationError as e:
             lg.error(f"{NEON_RED}Failed to send email notification: Authentication error (check SMTP_USER/SMTP_PASSWORD). Error: {e}{RESET}")
             return False
        except smtplib.SMTPServerDisconnected:
             lg.error(f"{NEON_RED}Failed to send email notification: Server disconnected unexpectedly.{RESET}")
             return False
        except smtplib.SMTPException as e:
             lg.error(f"{NEON_RED}Failed to send email notification: SMTP error: {e}{RESET}")
             return False
        except Exception as e:
            lg.error(f"{NEON_RED}Failed to send email notification due to an unexpected error: {e}{RESET}")
            lg.debug(traceback.format_exc())
            return False

    # --- Termux SMS Notification ---
    elif notify_type == "sms":
         # Check if Termux SMS is configured and command is available
         global _termux_sms_command_exists
         if TERMUX_SMS_RECIPIENT is None:
             lg.warning("SMS notification selected but TERMUX_SMS_RECIPIENT environment variable is not set. SMS not sent.")
             return False

         # Check for command existence only once per script run
         if _termux_sms_command_exists is None:
             termux_command_path = shutil.which('termux-sms-send')
             _termux_sms_command_exists = termux_command_path is not None
             if not _termux_sms_command_exists:
                  lg.warning(f"{NEON_YELLOW}SMS notification selected, but 'termux-sms-send' command not found in PATH. Ensure Termux:API is installed (`pkg install termux-api`) and PATH is correct.{RESET}")
             else:
                  lg.debug(f"Found 'termux-sms-send' command at: {termux_command_path}")

         if not _termux_sms_command_exists:
             return False # Don't proceed if command is missing

         # Prepare the command. Message should be the last argument(s).
         # Prefix message for clarity.
         sms_message = f"[Pyrmethus Bot] {subject}: {body}"
         command: List[str] = ['termux-sms-send', '-n', TERMUX_SMS_RECIPIENT, sms_message]

         # Use a timeout from config if available, default if not.
         sms_timeout = CONFIG.get('notifications', {}).get('sms_timeout_seconds', 30)

         try:
             lg.debug(f"Dispatching SMS notification to {TERMUX_SMS_RECIPIENT} (Timeout: {sms_timeout}s)...")
             # Execute the command via subprocess with timeout and output capture
             result = subprocess.run(
                 command,
                 capture_output=True,
                 text=True,          # Decode stdout/stderr as text
                 check=False,        # Don't raise exception on non-zero exit code
                 timeout=sms_timeout
             )

             if result.returncode == 0:
                 lg.info(f"{NEON_GREEN}SMS notification dispatched successfully.{RESET}")
                 if result.stdout: lg.debug(f"SMS Send stdout: {result.stdout.strip()}")
                 return True
             else:
                 # Log error details from stderr if available
                 error_details = result.stderr.strip() if result.stderr else "No stderr output"
                 lg.error(f"{NEON_RED}SMS notification failed. Return Code: {result.returncode}, Stderr: {error_details}{RESET}")
                 if result.stdout: lg.error(f"SMS Send stdout (on error): {result.stdout.strip()}")
                 return False
         except FileNotFoundError:
             # This shouldn't happen due to the check above, but handle defensively
             lg.error(f"{NEON_RED}SMS failed: 'termux-sms-send' command vanished unexpectedly? Ensure Termux:API is installed.{RESET}")
             _termux_sms_command_exists = False # Update cache
             return False
         except subprocess.TimeoutExpired:
             lg.error(f"{NEON_RED}SMS failed: Command timed out after {sms_timeout}s.{RESET}")
             return False
         except Exception as e:
             lg.error(f"{NEON_RED}SMS failed: Unexpected error during dispatch: {e}{RESET}")
             lg.debug(traceback.format_exc())
             return False

    else:
        # This branch should not be reached if notify_type is validated correctly
        lg.error(f"{NEON_RED}Internal Error: Reached invalid notification_type branch for '{notify_type}'.{RESET}")
        return False

# --- Configuration Loading & Validation ---

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
    updated_config = config.copy() # Create a shallow copy to avoid modifying the original dict in place during recursion
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            # Key is missing, add it with the default value
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config Update: Added missing key '{full_key_path}' with default value: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # If both default and loaded values are dicts, recurse into nested dict
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                # If nested dict was changed, update the parent dict and mark as changed
                updated_config[key] = nested_config
                changed = True
        # Optional: Could add type mismatch check here, but validation below handles it more robustly.
    return updated_config, changed

def _validate_and_correct_numeric(cfg_level: Dict, default_level: Dict, leaf_key: str, key_path: str,
                             min_val: Union[int, float, Decimal], max_val: Union[int, float, Decimal],
                             is_strict_min: bool = False, is_int: bool = False, allow_zero: bool = False) -> bool:
    """
    Validates a numeric config value at `key_path` (e.g., "protection.leverage") within a dictionary level.

    Checks type (int/float/str numeric), range [min_val, max_val] or (min_val, max_val] if strict.
    Uses the default value from `default_level` and logs a warning/info if correction needed.
    Updates the `cfg_level` dictionary in place if correction is made.
    Uses Decimal for robust comparisons.

    Args:
        cfg_level (Dict): The config dictionary level being validated (updated in place).
        default_level (Dict): The default config dictionary level for fallback values.
        leaf_key (str): The key within cfg_level and default_level to validate.
        key_path (str): The full dot-separated path to the key (e.g., "protection.leverage") for logging.
        min_val (Union[int, float, Decimal]): Minimum allowed value (inclusive unless is_strict_min).
        max_val (Union[int, float, Decimal]): Maximum allowed value (inclusive).
        is_strict_min (bool): If True, value must be strictly greater than min_val.
        is_int (bool): If True, value must be an integer.
        allow_zero (bool): If True, zero is allowed even if outside min/max range.

    Returns:
        bool: True if a correction was made, False otherwise.
    """
    original_val = cfg_level.get(leaf_key)
    default_val = default_level.get(leaf_key) # Get default for fallback on error

    corrected = False
    final_val = original_val # Start with the original value
    target_type_str = 'integer' if is_int else 'float' # For logging

    try:
        # 1. Reject Boolean Type Explicitly (often loaded from JSON as bool)
        if isinstance(original_val, bool):
             raise TypeError("Boolean type is not valid for numeric configuration.")

        # 2. Attempt Conversion to Decimal for Robust Validation
        # Convert to string first to handle floats accurately and numeric strings like "1.0" or " 5 ".
        try:
            str_val = str(original_val).strip()
            if str_val == "": # Handle empty strings explicitly
                 raise ValueError("Empty string cannot be converted to a number.")
            num_val = Decimal(str_val)
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

        if is_int:
            # Check if the Decimal value has a fractional part (is not an exact integer)
            if num_val % 1 != 0:
                needs_type_correction = True
                # Truncate towards zero (standard int() behavior)
                final_val_dec = num_val.to_integral_value(rounding=ROUND_DOWN)
                final_val = int(final_val_dec)
                init_logger.info(f"{NEON_YELLOW}Config Update: Truncated fractional part for integer key '{key_path}' from {repr(original_val)} to {repr(final_val)}.{RESET}")
                # Re-check range after truncation, as truncation could push it out of bounds
                min_check_passed_trunc = (final_val_dec > min_dec) if is_strict_min else (final_val_dec >= min_dec)
                range_check_passed_trunc = min_check_passed_trunc and (final_val_dec <= max_dec)
                if not range_check_passed_trunc and not (allow_zero and final_val_dec.is_zero()):
                    range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
                    allowed_str = f"{range_str}{' or 0' if allow_zero else ''}"
                    raise ValueError(f"Value truncated to {final_val}, which is outside the allowed range {allowed_str}.")
            # Check if the original type wasn't already int (e.g., it was 10.0 loaded as float, or "10" loaded as str)
            elif not isinstance(original_val, int):
                 needs_type_correction = True
                 final_val = int(num_val) # Convert the whole number Decimal to int
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type for integer key '{key_path}' from {type(original_val).__name__} to int (value: {repr(final_val)}).{RESET}")
            else:
                 # Already an integer conceptually and stored as int
                 final_val = int(num_val) # Ensure it's definitely int type

        else: # Expecting float/Decimal
            # If original was float or int, check if the Decimal conversion changed value significantly (precision issues)
            final_val_dec = num_val # Use the validated Decimal value
            if isinstance(original_val, (float, int)):
                 converted_float = float(final_val_dec) # Convert validated Decimal to float
                 # Use a small tolerance for float comparison to avoid flagging tiny representation differences
                 if abs(float(original_val) - converted_float) > float(PRICE_EPSILON): # Use epsilon for float compare
                      needs_type_correction = True
                      final_val = converted_float
                      init_logger.info(f"{NEON_YELLOW}Config Update: Adjusted float/int value for '{key_path}' due to precision from {repr(original_val)} to {repr(final_val)}.{RESET}")
                 else:
                      final_val = converted_float # Keep the float representation if close enough
            # If original was a string or something else, convert Decimal to float
            elif not isinstance(original_val, float): # Check specifically against float
                 needs_type_correction = True
                 final_val = float(final_val_dec) # Convert validated Decimal to float
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type for float key '{key_path}' from {type(original_val).__name__} to float (value: {repr(final_val)}).{RESET}")
            else: # Already a float
                 final_val = float(final_val_dec) # Ensure it's float type


        # Mark as corrected if type was changed OR value was truncated/adjusted
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

# Wrapper validation function for boolean keys (exists and is bool, attempts str/int conversion)
def _validate_boolean(cfg_level: Dict[str, Any], default_level: Dict[str, Any], leaf_key: str, key_path: str) -> bool:
    """
    Validates a boolean config value within a dictionary level.
    - Checks if the value is a boolean.
    - Attempts to interpret common string ("true", "false") or integer (1, 0) values as booleans.
    - If validation fails, logs a warning and replaces the value with the corresponding default.
    - Logs informational messages for conversions.

    Args:
        cfg_level: The current dictionary level of the loaded config being validated (updated in place).
        default_level: The corresponding dictionary level in the default config structure.
        leaf_key: The specific key (string) to validate within the current level.
        key_path: The full dot-notation path to the key (e.g., 'enable_trading') for logging.

    Returns:
        True if the value was corrected or replaced with the default, False otherwise.
    """
    original_val = cfg_level.get(leaf_key)
    default_val = default_level.get(leaf_key)
    corrected = False
    final_val = original_val # Start with the original value

    if isinstance(original_val, bool):
        # Already a boolean, no correction needed
        return False

    # Attempt to interpret common string representations ("true", "false", "1", "0", etc.)
    corrected_val = None
    if isinstance(original_val, str):
        val_lower = original_val.lower().strip()
        if val_lower in ['true', 'yes', '1', 'on']: corrected_val = True
        elif val_lower in ['false', 'no', '0', 'off']: corrected_val = False
    # Handle numeric 1/0 as well
    elif isinstance(original_val, int) and original_val in [0, 1]:
        corrected_val = bool(original_val)

    if corrected_val is not None:
         # Successfully interpreted as boolean, update the value and log
         init_logger.info(f"{NEON_YELLOW}Config Update: Corrected boolean-like value for '{key_path}' from {repr(original_val)} to {repr(corrected_val)}.{RESET}")
         final_val = corrected_val
         corrected = True
    else:
         # Cannot interpret the value as boolean, use the default and log a warning
         init_logger.warning(f"{NEON_YELLOW}Config Validation Warning: Invalid value for boolean key '{key_path}': {repr(original_val)}. Expected true/false or equivalent. Using default: {repr(default_val)}.{RESET}")
         final_val = default_val
         corrected = True

    # Update the configuration dictionary only if a correction or replacement was made
    if corrected:
        cfg_level[leaf_key] = final_val

    return corrected

# Wrapper validation function for string choice keys
def _validate_string_choice(cfg_level: Dict[str, Any], default_level: Dict[str, Any], leaf_key: str, key_path: str, valid_choices: List[str], case_sensitive: bool = False) -> bool:
     """
     Validates a string value within a dictionary level against a list of allowed choices.
     - Checks if the value is a string and is present in the `valid_choices` list.
     - Supports case-insensitive matching by default.
     - If validation fails, logs a warning and replaces the value with the corresponding default.
     - Logs informational messages if the case was corrected during validation.

     Args:
         cfg_level: The current dictionary level of the loaded config being validated (updated in place).
         default_level: The corresponding dictionary level in the default config structure.
         leaf_key: The specific key (string) to validate within the current level.
         key_path: The full dot-notation path to the key (e.g., 'interval') for logging.
         valid_choices: A list of strings representing the only acceptable values.
         case_sensitive: If True, checks require an exact case match. If False (default), uses case-insensitive matching.

     Returns:
         True if the value was corrected or replaced with the default, False otherwise.
     """
     original_val = cfg_level.get(leaf_key)
     default_val = default_level.get(leaf_key)
     corrected = False
     final_val = original_val # Start with the original value

     # Check if the value is a string and is in the allowed choices (case-insensitive by default)
     found_match = False
     corrected_match_value = None # Store the correctly cased version if found case-insensitively

     if isinstance(original_val, str):
         for choice in valid_choices:
              if case_sensitive:
                  if original_val == choice:
                      found_match = True
                      corrected_match_value = choice # Use the exact match
                      break
              else:
                  # Ensure original_val is treated as string before lower()
                  if str(original_val).lower() == str(choice).lower():
                      found_match = True
                      corrected_match_value = choice # Use the canonical casing from `choices` list
                      break

     if not found_match: # Not a valid choice (or wrong type)
         init_logger.warning(f"{NEON_YELLOW}Config Validation Warning: Invalid value for '{key_path}': {repr(original_val)}. Must be one of {valid_choices} ({'case-sensitive' if case_sensitive else 'case-insensitive'}). Using default: {repr(default_val)}.{RESET}")
         final_val = default_val
         corrected = True
     elif corrected_match_value != original_val: # Valid choice, but potentially wrong case (if case-insensitive) or type (e.g. number instead of string '1')
         init_logger.info(f"{NEON_YELLOW}Config Update: Corrected case/value for '{key_path}' from '{original_val}' to '{corrected_match_value}'.{RESET}")
         final_val = corrected_match_value
         corrected = True
     # If found_match is True AND corrected_match_value == original_val, no correction needed

     # Update the configuration dictionary only if a correction or replacement was made
     if corrected:
         cfg_level[leaf_key] = final_val

     return corrected

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads, validates, and potentially updates configuration from a JSON file.

    Steps:
    1. Defines the default configuration structure and values.
    2. Checks if the specified config file exists. If not, creates it with default values.
    3. Loads the configuration from the JSON file. Handles JSON decoding errors by attempting
       to back up the corrupted file and recreating a default one.
    4. Ensures all necessary keys from the default structure are present in the loaded config,
       adding missing keys with their default values.
    5. Performs detailed validation and type/range correction for each parameter using helper
       functions (_validate_and_correct_numeric, _validate_boolean, _validate_string_choice).
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
        "max_concurrent_positions": 1,          # Maximum number of positions allowed open simultaneously across all pairs (Current logic assumes 1 per symbol).

        # == Risk & Sizing ==
        "risk_per_trade": 0.01,                 # Fraction of available balance to risk per trade (e.g., 0.01 = 1%). Must be > 0.0 and <= 1.0.
        "leverage": 20,                         # Desired leverage for contract trading (integer). 0 or 1 typically means spot/no leverage. Exchange limits apply.

        # == API & Timing ==
        "retry_delay": RETRY_DELAY_SECONDS,             # Base delay in seconds between API retry attempts (integer)
        "loop_delay_seconds": LOOP_DELAY_SECONDS,       # Delay in seconds between processing cycles for each symbol (integer)
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Delay after placing order to confirm position status (integer)

        # == Data Fetching ==
        "fetch_limit": DEFAULT_FETCH_LIMIT,             # Default number of historical klines to fetch (integer)
        "min_klines_for_strategy": MIN_KLINES_FOR_STRATEGY, # Minimum klines needed to run strategy (integer > 0)
        "orderbook_limit": 25,                          # (Currently Unused) Limit for order book depth fetching (integer, if feature implemented later)

        # == Strategy Parameters (Volumatic Trend + Order Blocks) ==
        # TODO: Replace placeholder parameters and logic with the actual strategy implementation.
        "strategy_params": {
            # -- Volumatic Trend (VT) - Placeholders --
            "vt_length": DEFAULT_VT_LENGTH,             # Lookback period for VT calculation (integer > 0)
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,     # Lookback period for ATR calculation within VT (integer > 0)
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH, # Placeholder: Lookback for Volume EMA/SWMA (integer > 0)
            "vt_atr_multiplier": DEFAULT_VT_ATR_MULTIPLIER, # Placeholder: ATR multiplier for VT bands (float > 0)
            "vt_step_atr_multiplier": DEFAULT_VT_STEP_ATR_MULTIPLIER, # Unused placeholder (float)

            # -- Order Blocks (OB) - Placeholders --
            "ob_source": DEFAULT_OB_SOURCE,             # Candle part to define OB price range: "Wicks" or "Body" (string)
            "ph_left": DEFAULT_PH_LEFT,                 # Lookback periods for Pivot High detection (integer > 0)
            "ph_right": DEFAULT_PH_RIGHT,               # Pivot High lookforward periods (integer > 0)
            "pl_left": DEFAULT_PL_LEFT,                 # Pivot Low lookback periods (integer > 0)
            "pl_right": DEFAULT_PL_RIGHT,               # Pivot Low lookforward periods (integer > 0)
            "ob_extend": DEFAULT_OB_EXTEND,             # Extend OB visualization until violated? (boolean)
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,       # Maximum number of active Order Blocks to track per side (integer > 0)
            "ob_entry_proximity_factor": DEFAULT_OB_ENTRY_PROXIMITY_FACTOR, # Price must be within this factor of OB edge for entry (float >= 1.0)
            "ob_exit_proximity_factor": DEFAULT_OB_EXIT_PROXIMITY_FACTOR,  # Price must be within this factor of OB edge for exit/invalidation (float >= 1.0)
        },

        # == Protection Parameters (Stop Loss, Take Profit, Trailing, Break Even) ==
        "protection": {
            # -- Initial SL/TP (often ATR-based) --
            "initial_stop_loss_atr_multiple": DEFAULT_INITIAL_STOP_LOSS_ATR_MULTIPLE, # Initial SL distance = ATR * this multiple (float > 0)
            "initial_take_profit_atr_multiple": DEFAULT_INITIAL_TAKE_PROFIT_ATR_MULTIPLE, # Initial TP distance = ATR * this multiple (float >= 0, 0 means no initial TP)

            # -- Break Even (BE) --
            "enable_break_even": DEFAULT_ENABLE_BREAK_EVEN,         # Enable moving SL to break-even? (boolean)
            "break_even_trigger_atr_multiple": DEFAULT_BREAK_EVEN_TRIGGER_ATR_MULTIPLE, # Move SL to BE when price moves ATR * multiple in profit (float > 0)
            "break_even_offset_ticks": DEFAULT_BREAK_EVEN_OFFSET_TICKS, # Offset SL from entry by this many price ticks for BE (integer >= 0)

            # -- Trailing Stop Loss (TSL) --
            "enable_trailing_stop": DEFAULT_ENABLE_TRAILING_STOP,               # Enable Trailing Stop Loss? (boolean)
            "trailing_stop_callback_rate": DEFAULT_TRAILING_STOP_CALLBACK_RATE, # TSL callback/distance (float > 0). E.g., 0.005 = 0.5%.
            "trailing_stop_activation_percentage": DEFAULT_TRAILING_STOP_ACTIVATION_PERCENTAGE, # Activate TSL when price moves this % from entry (float >= 0).
        },

        # == Notifications ==
        "notifications": {
            "enable_notifications": True, # Master switch for notifications
            "notification_type": "email", # 'email' or 'sms'. Must match one of these.
            "sms_timeout_seconds": 30     # Timeout for termux-sms-send command (integer seconds)
        },

        # == Backtesting (Placeholder) ==
        "backtesting": {
            "enabled": False,
            "start_date": "2023-01-01", # Format YYYY-MM-DD
            "end_date": "2023-12-31",   # Format YYYY-MM-DD
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
            QUOTE_CURRENCY = str(default_config.get("quote_currency", "USDT")).strip().upper()
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config # Return defaults directly as the file is now correct

        except IOError as e:
            init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Could not create config file '{filepath}': {e}.{RESET}")
            init_logger.critical(f"{NEON_RED}Please check directory permissions. Using internal defaults as fallback.{RESET}")
            # Fallback to using internal defaults in memory if file creation fails
            QUOTE_CURRENCY = str(default_config.get("quote_currency", "USDT")).strip().upper()
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
             init_logger.warning(f"{NEON_YELLOW}Could not back up corrupted config file '{filepath}': {backup_err}{RESET}")

        # Attempt to recreate the default file
        try:
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully recreated default config file: {filepath}{RESET}")
            QUOTE_CURRENCY = str(default_config.get("quote_currency", "USDT")).strip().upper()
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config # Return the defaults
        except IOError as e_create:
            init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Error recreating config file after corruption: {e_create}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = str(default_config.get("quote_currency", "USDT")).strip().upper()
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return default_config # Fallback to internal defaults
    except Exception as e:
        # Catch any other unexpected errors during file loading or initial parsing
        init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Unexpected error loading config file '{filepath}': {e}{RESET}", exc_info=True)
        init_logger.critical(f"{NEON_RED}Using internal defaults as fallback.{RESET}")
        QUOTE_CURRENCY = str(default_config.get("quote_currency", "USDT")).strip().upper()
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

        # Helper function to safely navigate nested dictionaries for validation/correction
        def get_nested_levels(cfg: Dict, path: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
            """Gets the dictionary level and leaf key for validation, handling potential path errors."""
            keys = path.split('.')
            current_cfg_level = cfg
            current_def_level = default_config # Access the default config structure
            try:
                # Iterate through parent keys to reach the level containing the target key
                for key in keys[:-1]:
                    # Check existence and type in BOTH loaded and default config
                    if key not in current_cfg_level or not isinstance(current_cfg_level[key], dict):
                         # This indicates a structure mismatch or key missing despite ensure_keys, log error
                         init_logger.error(f"{NEON_RED}Config validation structure error: Path segment '{key}' not found or not a dictionary in loaded config at '{path}'. Using default subsection.{RESET}")
                         # Attempt to replace the entire subsection with default if possible
                         if key in current_def_level and isinstance(current_def_level[key], dict):
                              current_cfg_level[key] = default_config[key] # Reset subsection to default
                              nonlocal config_needs_saving
                              config_needs_saving = True
                              current_cfg_level = current_cfg_level[key] # Continue with the default subsection
                              current_def_level = current_def_level[key]
                              continue # Skip to next key in path
                         else:
                              return None, None, None # Cannot recover
                    if key not in current_def_level or not isinstance(current_def_level[key], dict):
                         # This indicates an error in the *default_config* structure definition itself
                         init_logger.critical(f"{NEON_RED}FATAL: Internal Default Config structure error: Path segment '{key}' not found or not a dictionary in default config at '{path}'. Cannot validate.{RESET}")
                         return None, None, None # Cannot proceed

                    current_cfg_level = current_cfg_level[key]
                    current_def_level = current_def_level[key]

                leaf_key = keys[-1]
                # Ensure the final key exists in the default structure (should always if default_config is correct)
                if leaf_key not in current_def_level:
                    init_logger.critical(f"{NEON_RED}FATAL: Internal Default Config structure error: Leaf key '{leaf_key}' not found in default config structure for path '{path}'. Cannot validate.{RESET}")
                    return None, None, None

                # Check if leaf key is missing in loaded config level *after* ensure_keys.
                # This case should ideally not happen if ensure_keys runs correctly, but handle defensively.
                if leaf_key not in current_cfg_level:
                    init_logger.warning(f"{NEON_YELLOW}Config validation: Key '{key_path}' unexpectedly missing after ensure_keys. Using default value: {repr(current_def_level[leaf_key])}.{RESET}")
                    current_cfg_level[leaf_key] = current_def_level[leaf_key]
                    nonlocal config_needs_saving
                    config_needs_saving = True # Needs saving if added here

                return current_cfg_level, current_def_level, leaf_key
            except Exception as e:
                init_logger.critical(f"{NEON_RED}FATAL: Unexpected error traversing config path '{path}' for validation: {e}. Cannot validate.{RESET}", exc_info=True)
                return None, None, None # Indicate failure to access the level

        # Wrapper validation function for numeric values
        def validate_numeric(cfg: Dict, key_path: str, min_val, max_val, is_strict_min=False, is_int=False, allow_zero=False):
            """Applies numeric validation using helpers and marks config for saving on changes."""
            nonlocal config_needs_saving # Allow modification of the outer scope variable
            cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
            if cfg_level is None or def_level is None or leaf_key is None:
                # Error already logged by get_nested_levels, or structure reset to default
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

            corrected = _validate_boolean(cfg_level, def_level, leaf_key, key_path)
            if corrected:
                config_needs_saving = True

        # Wrapper validation function for string choices
        def validate_string_choice(cfg: Dict, key_path: str, choices: List[str], case_sensitive: bool = False):
             """Validates string value against a list of allowed choices, marking config for saving on changes."""
             nonlocal config_needs_saving
             cfg_level, def_level, leaf_key = get_nested_levels(cfg, key_path)
             if cfg_level is None or def_level is None or leaf_key is None: return # Error handled

             corrected = _validate_string_choice(cfg_level, def_level, leaf_key, key_path, choices, case_sensitive)
             if corrected:
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

        # Validate quote_currency (must be non-empty string, convert to uppercase)
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
        # Use updated value or default fallback
        QUOTE_CURRENCY = str(updated_config.get("quote_currency", "USDT")).strip().upper()
        init_logger.info(f"Quote currency set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")

        validate_numeric(updated_config, "max_concurrent_positions", 1, 100, is_int=True) # Range 1 to 100 concurrent positions

        # == Risk & Sizing ==
        # Risk per trade must be > 0% and <= 100% (Decimal comparison)
        validate_numeric(updated_config, "risk_per_trade", Decimal('0'), Decimal('1'), is_strict_min=True)
        # Leverage >= 0 (0 or 1 means disable setting leverage). Max 200 as a sanity limit.
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True)

        # == API & Timing ==
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True) # 1-60 seconds base retry delay
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True) # 1 second to 1 hour loop delay
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 120, is_int=True) # 1-120 seconds confirm delay

        # == Data Fetching ==
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True) # Min 50 klines, max MAX_DF_LEN
        # min_klines_for_strategy must be positive and less than or equal to fetch_limit
        validate_numeric(updated_config, "min_klines_for_strategy", 1, updated_config.get("fetch_limit", DEFAULT_FETCH_LIMIT), is_int=True, is_strict_min=True)
        validate_numeric(updated_config, "orderbook_limit", 1, 100, is_int=True) # 1-100 order book depth levels

        # == Strategy Params ==
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 1000, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True) # ATR period <= max data length
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True) # Vol EMA period <= max data length
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0) # ATR multiplier range 0.1 to 20.0
        # vt_step_atr_multiplier is currently unused, minimal validation
        validate_numeric(updated_config, "strategy_params.vt_step_atr_multiplier", 0.1, 20.0)

        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True) # Pivot lookback 1-100
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True) # Pivot lookforward 1-100
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True) # Pivot lookback 1-100
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True) # Pivot lookforward 1-100
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 500, is_int=True) # Max OBs 1-500
        # Proximity factors must be >= 1.0 (use Decimal for comparison)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", Decimal('1.0'), Decimal('1.1')) # e.g., 1.005 = 0.5% proximity
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", Decimal('1.0'), Decimal('1.1')) # e.g., 1.001 = 0.1% proximity
        validate_string_choice(updated_config, "strategy_params.ob_source", ["Wicks", "Body"], case_sensitive=False) # Case-insensitive check
        validate_boolean(updated_config, "strategy_params.ob_extend")

        # == Protection Params ==
        # Initial SL distance must be > 0 ATRs
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", Decimal('0'), Decimal('100.0'), is_strict_min=True)
        # Initial TP distance can be 0 (disabled) or positive
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", Decimal('0'), Decimal('100.0'), allow_zero=True)

        validate_boolean(updated_config, "protection.enable_break_even")
        # BE trigger must be > 0 ATRs in profit
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", Decimal('0'), Decimal('10.0'), is_strict_min=True)
        # BE offset ticks can be 0 or positive
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True)

        validate_boolean(updated_config, "protection.enable_trailing_stop")
        # TSL callback must be > 0 (e.g., 0-50% range)
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", Decimal('0'), Decimal('0.5'), is_strict_min=True)
        # TSL activation can be 0% (immediate once triggered) or positive (e.g., 0-50% range)
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", Decimal('0'), Decimal('0.5'), allow_zero=True)


        # == Notifications ==
        validate_boolean(updated_config, "notifications.enable_notifications")
        validate_string_choice(updated_config, "notifications.notification_type", ["email", "sms"], case_sensitive=False)
        validate_numeric(updated_config, "notifications.sms_timeout_seconds", 5, 120, is_int=True) # 5-120 seconds timeout


        # == Backtesting (Placeholder) ==
        validate_boolean(updated_config, "backtesting.enabled")
        # Basic date format check YYYY-MM-DD
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        bp = updated_config.get("backtesting", {})
        def_bp = default_config.get("backtesting", {}) # Use default nested dict for validation
        for date_key in ["start_date", "end_date"]:
            date_val = bp.get(date_key)
            def_date_val = def_bp.get(date_key)
            if not isinstance(date_val, str) or not date_pattern.match(date_val):
                 init_logger.warning(f"{NEON_YELLOW}Config Warning: 'backtesting.{date_key}' ('{date_val}') is invalid. Expected YYYY-MM-DD format. Using default: '{def_date_val}'{RESET}")
                 bp[date_key] = def_date_val
                 config_needs_saving = True
            # Optional: Add logic to check start_date < end_date if needed (e.g., using datetime.strptime)


        init_logger.debug("Configuration parameter validation complete.")

        # --- Step 4: Save Updated Config if Necessary ---
        if config_needs_saving:
             init_logger.info(f"{NEON_YELLOW}Configuration updated with defaults or corrections. Saving changes to '{filepath}'...{RESET}")
             try:
                 # Convert any lingering Decimal objects to float for JSON serialization
                 # This is important as json.dump doesn't handle Decimal natively.
                 def convert_decimals_to_float(obj: Any) -> Any:
                     if isinstance(obj, Decimal):
                         return float(obj)
                     if isinstance(obj, dict):
                         return {k: convert_decimals_to_float(v) for k, v in obj.items()}
                     if isinstance(obj, list):
                         return [convert_decimals_to_float(elem) for elem in obj]
                     return obj

                 output_config = convert_decimals_to_float(updated_config)

                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(output_config, f_write, indent=4, ensure_ascii=False)
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
        QUOTE_CURRENCY = str(default_config.get("quote_currency", "USDT")).strip().upper()
        init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        return default_config # Fallback to internal defaults

# --- Load Global Configuration ---
# This loads the configuration, performs validation, and updates the global QUOTE_CURRENCY
CONFIG = load_config(CONFIG_FILE)


# --- CCXT Helper Functions (Enhanced & Refined) ---

def _safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0"), allow_none: bool = False) -> Optional[Decimal]:
    """
    Safely converts a value to a Decimal, handling None, pandas NA, empty strings,
    non-finite numbers, and potential errors.

    Args:
        value: The value to convert (can be string, float, int, Decimal, None, pandas NA, etc.).
        default: The Decimal value to return if conversion fails and allow_none is False.
        allow_none: If True and input value is None/NA/empty/conversion fails, returns None instead of default.

    Returns:
        The converted Decimal value, the default, or None if allow_none is True and input is invalid/None.
    """
    # Use pandas.isna to handle numpy.nan, None, and pandas.NA efficiently
    if pd.isna(value):
        return None if allow_none else default

    # Explicitly check for None type as well, as pd.isna(None) is True
    if value is None:
        return None if allow_none else default

    try:
        # Convert to string first for consistent handling of floats and potential leading/trailing spaces
        s_val = str(value).strip()
        if not s_val: # Handle empty strings explicitly after stripping
             # An empty string is generally not a valid number
             return None if allow_none else default

        # Attempt conversion to Decimal
        d_val = Decimal(s_val)

        # Check for non-finite values (NaN, Infinity) which are generally invalid for financial calculations
        if not d_val.is_finite():
             # init_logger.debug(f"Non-finite Decimal value encountered: {d_val}") # Optional debug log
             return None if allow_none else default

        # If conversion is successful and finite, return the Decimal object
        return d_val
    except (InvalidOperation, TypeError, ValueError):
        # Catch errors during Decimal conversion (e.g., non-numeric strings)
        return None if allow_none else default


def _format_price(exchange: ccxt.Exchange, symbol: str, price: Optional[Decimal]) -> Optional[str]:
    """
    Formats a price value according to the market's price precision rules using ccxt.
    Returns the formatted string or None if the input price is None or invalid.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        price: The price as an Optional Decimal.

    Returns:
        The formatted price string, or None if input is invalid or market info is missing.
    """
    if price is None:
        return None
    # Ensure price is finite before formatting (allow zero for certain cases like SL/TP removal)
    if not isinstance(price, Decimal) or not price.is_finite():
         setup_logger(f"ccxt.format.{symbol}").warning(f"Attempted to format invalid price: {price}")
         return None
    # Price formatting should generally handle positive and potentially zero values if exchange allows
    # Negative prices are usually invalid.
    if price < Decimal('0'):
         setup_logger(f"ccxt.format.{symbol}").warning(f"Attempted to format negative price: {price}")
         return None

    try:
        # Use ccxt's built-in precision formatting
        # Note: ccxt.price_to_precision might expect float/string, but often handles Decimal too.
        # Convert to string for safer input to ccxt functions if issues arise.
        formatted_price = exchange.price_to_precision(symbol, str(price))
        return formatted_price
    except ccxt.ExchangeError as e:
        # More specific logging for exchange errors during formatting (e.g., market not found)
        setup_logger(f"ccxt.format.{symbol}").error(f"Exchange error formatting price {price} for {symbol}: {e}")
        return None
    except Exception as e:
        # Log other formatting errors at debug level
        setup_logger(f"ccxt.format.{symbol}").debug(f"Failed to format price {price} for {symbol}: {e}")
        return None

def _format_amount(exchange: ccxt.Exchange, symbol: str, amount: Optional[Decimal]) -> Optional[str]:
    """
    Formats an amount/quantity value according to the market's amount precision rules using ccxt.
    Returns the formatted string or None if the input amount is None or invalid.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        amount: The amount as an Optional Decimal.

    Returns:
        The formatted amount string, or None if input is invalid or market info is missing.
    """
    if amount is None:
        return None
    # Ensure amount is finite and non-negative
    if not isinstance(amount, Decimal) or amount < Decimal('0') or not amount.is_finite():
         setup_logger(f"ccxt.format.{symbol}").warning(f"Attempted to format invalid amount: {amount}")
         return None
    try:
        # Use ccxt's built-in precision formatting
        # Convert to string for safer input if needed.
        formatted_amount = exchange.amount_to_precision(symbol, str(amount))
        return formatted_amount
    except ccxt.ExchangeError as e:
        setup_logger(f"ccxt.format.{symbol}").error(f"Exchange error formatting amount {amount} for {symbol}: {e}")
        return None
    except Exception as e:
        # Log other formatting errors at debug level
        setup_logger(f"ccxt.format.{symbol}").debug(f"Failed to format amount {amount} for {symbol}: {e}")
        return None

def _parse_decimal(value: Any, key_name: str, logger: logging.Logger, allow_none: bool = False, default: Decimal = Decimal("0.0")) -> Optional[Decimal]:
    """
    Safely parses a raw value from an API response (dict) into a Decimal, logging errors.
    This is a specific wrapper around _safe_decimal_conversion for API parsing contexts.

    Args:
        value: The raw value from the API response dictionary.
        key_name: The name of the key this value came from (for logging).
        logger: The logger instance for the specific symbol/context.
        allow_none: If True, returns None if value is None/NA/empty/invalid. If False, returns default.
        default: The default Decimal value to return if allow_none is False and parsing fails.

    Returns:
        The parsed Decimal, or None/default as specified.
        Logs a debug message if parsing fails but a default/None is returned.
    """
    d_val = _safe_decimal_conversion(value, default=default, allow_none=True) # Always allow_none internally first

    if d_val is None:
        # Parsing failed (None, NA, empty string, invalid format, non-finite)
        if allow_none:
            # Log only if the original value wasn't obviously None/empty
            if value is not None and str(value).strip() != "":
                 logger.debug(f"Parsed '{key_name}': Value {repr(value)} is invalid, returning None as allowed.")
            return None
        else:
            # Log only if the original value wasn't obviously None/empty
            if value is not None and str(value).strip() != "":
                 logger.debug(f"Parsed '{key_name}': Value {repr(value)} is invalid, returning default {default.normalize()}.")
            return default
    # else: # Successfully parsed a finite Decimal
    #     logger.debug(f"Parsed '{key_name}': Value {repr(value)} converted to Decimal {d_val.normalize()}.")
    return d_val # Return the successfully parsed Decimal


def _handle_ccxt_exception(e: Exception, logger: logging.Logger, action: str, symbol: Optional[str] = None, retry_attempt: int = 0) -> bool:
    """
    Logs CCXT exceptions with specific messages and determines if a retry is warranted.
    Implements exponential backoff for retries.

    Args:
        e (Exception): The exception caught.
        logger (logging.Logger): The logger instance for the operation.
        action (str): A description of the action being attempted (e.g., "fetching klines", "creating order").
        symbol (Optional[str]): The symbol involved in the action, if applicable.
        retry_attempt (int): The current retry attempt number (0 for initial attempt).

    Returns:
        bool: True if the error is potentially transient and retrying is advised, False otherwise.
    """
    symbol_str = f" for {symbol}" if symbol else ""
    log_prefix = f"Error {action}{symbol_str} (Attempt {retry_attempt + 1}/{MAX_API_RETRIES + 1}): "
    retry_recommendation = False # Assume no retry by default
    wait_time = 0 # Seconds to wait before next attempt

    # Use isinstance for exception types rather than checking string content where possible
    if isinstance(e, (ccxt.DDoSProtection, ccxt.RateLimitExceeded)):
        # Rate limits or DDoS protection responses suggest temporary overload
        logger.warning(f"{NEON_YELLOW}{log_prefix}Rate limit or DDoS protection triggered: {e}{RESET}")
        # Check if `e.retryAfter` is available for a hint on how long to wait (in milliseconds)
        retry_after_ms = getattr(e, 'retryAfter', None)
        if retry_after_ms and isinstance(retry_after_ms, (int, float)) and retry_after_ms > 0:
             wait_time = max(retry_after_ms / 1000.0, RETRY_DELAY_SECONDS) # Convert ms to s, ensure minimum delay
             wait_time = min(wait_time, 600) # Cap wait time (e.g., 10 minutes max)
             logger.info(f"Waiting instructed {wait_time:.2f}s before retrying...")
        else:
            # Fallback to exponential backoff if no specific retry time is given
            wait_time = RETRY_DELAY_SECONDS * (2 ** retry_attempt)
            wait_time = min(wait_time, 600) # Cap exponential backoff
            logger.info(f"Waiting default backoff {wait_time:.2f}s before retrying...")
        retry_recommendation = True # These are usually temporary

    elif isinstance(e, (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException)):
        # General network issues, timeouts, connection errors
        logger.warning(f"{NEON_YELLOW}{log_prefix}Network or timeout error: {e}{RESET}")
        wait_time = RETRY_DELAY_SECONDS * (2 ** retry_attempt) # Exponential backoff
        wait_time = min(wait_time, 600) # Cap
        logger.info(f"Waiting {wait_time:.2f}s before retrying...")
        retry_recommendation = True # These are often transient

    elif isinstance(e, (ccxt.ExchangeNotAvailable, ccxt.OnMaintenance)):
        # Exchange is down for maintenance or temporarily offline
        logger.warning(f"{NEON_YELLOW}{log_prefix}Exchange temporarily unavailable or in maintenance: {e}{RESET}")
        # Wait a longer, fixed amount for exchange-wide issues
        wait_time = max(RETRY_DELAY_SECONDS * 5, 60) # Wait at least 60s or 5 * base delay
        logger.info(f"Waiting {wait_time:.2f}s before retrying...")
        retry_recommendation = True # Worth retrying after a pause

    elif isinstance(e, ccxt.AuthenticationError):
        # API Key/Secret invalid, IP not whitelisted, etc. - Fatal error
        logger.critical(f"{NEON_RED}{BRIGHT}{log_prefix}Authentication Error: {e}{RESET}")
        logger.critical(f"{NEON_RED}Please check your API credentials and IP whitelist. Exiting.{RESET}")
        # Optionally trigger immediate shutdown
        global _shutdown_requested
        _shutdown_requested = True
        retry_recommendation = False # Authentication errors are not retryable in this context

    elif isinstance(e, ccxt.InsufficientFunds):
        # Attempted to place an order but balance was too low.
        logger.error(f"{NEON_RED}{log_prefix}Insufficient funds: {e}{RESET}")
        # This implies an issue with position sizing or unexpected balance change.
        # Retrying the *same* order is unlikely to work immediately.
        retry_recommendation = False # The *order placement* itself is not retryable without addressing funds issue
        # Optionally send a notification about insufficient funds
        send_notification(f"Funds Error: {symbol}", f"Insufficient funds for {action} on {symbol}. Error: {e}", logger, notification_type='email') # Prioritize email for critical errors

    elif isinstance(e, ccxt.InvalidOrder):
        # Order parameters (price, amount, type) are invalid according to the exchange.
        logger.error(f"{NEON_RED}{log_prefix}Invalid order parameters: {e}{RESET}")
        # This likely indicates a calculation error, incorrect precision, or violation of limits. Not retryable without fixing logic/parameters.
        # Log details including any parameters that were attempted
        logger.debug(f"Invalid order details/context: {getattr(e, 'context', 'N/A')}") # CCXT may attach context
        retry_recommendation = False # The *order placement* is not retryable without fixing parameters

    elif isinstance(e, ccxt.OrderNotFound):
         # Attempted to cancel or modify an order that doesn't exist or was already filled/cancelled.
         logger.warning(f"{NEON_YELLOW}{log_prefix}Order not found: {e}{RESET}")
         retry_recommendation = False # Order doesn't exist, retrying won't help find it. Assume it was handled otherwise.

    elif isinstance(e, ccxt.CancelPending):
         # The order cancellation is pending. Retrying cancellation might be needed or just wait.
         logger.info(f"{log_prefix}Cancel pending: {e}. Waiting briefly before potential retry...")
         wait_time = RETRY_DELAY_SECONDS # Small delay before re-attempting cancellation check/cancel
         retry_recommendation = True # Keep retrying cancel attempts or checks

    elif isinstance(e, ccxt.ArgumentsRequired):
        # Missing required arguments for an API call.
        logger.critical(f"{NEON_RED}{BRIGHT}{log_prefix}Missing arguments for API call: {e}{RESET}")
        logger.critical(f"{NEON_RED}This indicates a bug in the bot's code. Exiting.{RESET}")
        global _shutdown_requested
        _shutdown_requested = True
        retry_recommendation = False # Bug requires code fix

    elif isinstance(e, ccxt.ExchangeError):
        # Catch other generic exchange errors that weren't specifically handled above
        logger.warning(f"{NEON_YELLOW}{log_prefix}A generic Exchange error occurred: {e}{RESET}")
        # Check for specific messages that might indicate temporary issues (e.g., "System busy", "Try again later")
        error_msg = str(e).lower()
        if "busy" in error_msg or "try again" in error_msg or "timeout" in error_msg:
             wait_time = RETRY_DELAY_SECONDS * (2 ** retry_attempt) # Exponential backoff
             wait_time = min(wait_time, 600) # Cap
             logger.info(f"Generic exchange error seems temporary. Waiting {wait_time:.2f}s before retrying...")
             retry_recommendation = True
        else:
             # Assume other ExchangeErrors are potentially fatal or require investigation
             logger.error(f"{NEON_RED}{log_prefix}Unhandled ExchangeError: {e}. Not retrying this action.{RESET}")
             retry_recommendation = False # Be cautious with unknown ExchangeErrors

    else:
        # Catch all other unexpected CCXT exceptions or non-CCXT exceptions during API interaction
        logger.error(f"{NEON_RED}{log_prefix}An unexpected error occurred during API interaction: {e}{RESET}", exc_info=True) # Log traceback for unexpected errors
        # Decide default retry behavior for unknown errors - generally cautious
        wait_time = RETRY_DELAY_SECONDS * (retry_attempt + 1) # Linear backoff for unknown
        wait_time = min(wait_time, 300) # Cap wait time for unknown errors
        logger.info(f"Waiting {wait_time:.2f}s before retrying (for unexpected error type)...")
        # Only retry up to max attempts for unknown errors
        retry_recommendation = (retry_attempt < MAX_API_RETRIES)

    # Perform the wait if a retry is recommended
    if retry_recommendation and wait_time > 0:
        # Sleep while periodically checking for shutdown request
        wait_end_time = time.time() + wait_time
        while time.time() < wait_end_time:
             if _shutdown_requested:
                  logger.info("Shutdown requested during retry delay. Aborting retry.")
                  return False # Abort retry if shutdown requested
             time.sleep(min(1, wait_end_time - time.time())) # Sleep for 1s or remaining time

    # Return whether to retry ONLY if max retries haven't been exceeded
    return retry_recommendation and (retry_attempt < MAX_API_RETRIES)


# --- Exchange Initialization ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes and returns the CCXT exchange instance with error handling.

    Args:
        logger: The logger instance to use for initialization messages.

    Returns:
        A configured ccxt.Exchange instance, or None if initialization fails.
    """
    lg = logger
    lg.info("Initializing CCXT exchange instance...")
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Enable CCXT's built-in rate limiter
            'options': {
                'adjustForTimeDifference': True, # Adjust for potential time skew
                # Bybit V5 specific options (example, check CCXT/Bybit docs)
                'defaultType': 'swap', # Or 'future', 'spot' depending on primary use
                # 'code': QUOTE_CURRENCY, # May be needed for Unified Margin balance fetching
                # 'recvWindow': 10000, # Optional: Increase receive window if needed
            },
        }

        # Select Bybit V5 class
        exchange = ccxt.bybit(exchange_options)

        # Set sandbox mode if configured
        if CONFIG.get("use_sandbox", True):
            lg.warning(f"{NEON_YELLOW}Sandbox mode enabled. Using Bybit Testnet.{RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}Sandbox mode DISABLED. Using Bybit LIVE environment! REAL FUNDS AT RISK!{RESET}")

        # Load markets data (crucial for precision, limits, etc.)
        lg.info("Loading markets from exchange...")
        exchange.load_markets()
        lg.info(f"{NEON_GREEN}Markets loaded successfully. Found {len(exchange.markets)} markets.{RESET}")

        # Optional: Test API connection (e.g., fetch server time)
        lg.info("Testing API connection...")
        server_time = exchange.fetch_time()
        local_time = int(time.time() * 1000)
        time_diff = abs(server_time - local_time)
        lg.info(f"Exchange time: {server_time}, Local time: {local_time}, Difference: {time_diff} ms")
        if time_diff > 5000: # Warn if time difference is significant (e.g., > 5 seconds)
             lg.warning(f"{NEON_YELLOW}Significant time difference ({time_diff} ms) detected between local machine and exchange server. Check system clock synchronization (NTP).{RESET}")

        lg.info(f"{NEON_GREEN}CCXT Exchange instance initialized successfully.{RESET}")
        return exchange

    except ccxt.AuthenticationError as e:
        lg.critical(f"{NEON_RED}{BRIGHT}Authentication Error during exchange initialization: {e}{RESET}")
        lg.critical(f"{NEON_RED}Please check API Key/Secret and IP Whitelist settings. Cannot continue.{RESET}")
        return None
    except ccxt.NetworkError as e:
        lg.critical(f"{NEON_RED}{BRIGHT}Network Error during exchange initialization: {e}{RESET}")
        lg.critical(f"{NEON_RED}Check internet connection and exchange status (e.g., {exchange.urls.get('www') if 'exchange' in locals() else 'N/A'}). Cannot continue.{RESET}")
        return None
    except ccxt.ExchangeError as e:
        lg.critical(f"{NEON_RED}{BRIGHT}Exchange Error during exchange initialization: {e}{RESET}")
        lg.critical(f"{NEON_RED}Could not load markets or connect properly. Cannot continue.{RESET}")
        return None
    except Exception as e:
        lg.critical(f"{NEON_RED}{BRIGHT}An unexpected error occurred during exchange initialization: {e}{RESET}", exc_info=True)
        return None


# --- Exchange Data Fetching Functions (Using CCXT Helpers) ---

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Fetches historical kline data for a symbol using CCXT with retry logic.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        timeframe: The kline interval in *config* format (e.g., '1', '5', '60').
        limit: The number of historical klines to fetch. Max limit is BYBIT_API_KLINE_LIMIT.
        logger: The logger instance for the symbol.

    Returns:
        A pandas DataFrame containing the kline data (indexed by timestamp, UTC),
        or None if fetching fails after retries. Columns are Decimal type.
    """
    lg = logger
    ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
    if not ccxt_timeframe:
        lg.error(f"Invalid timeframe '{timeframe}' provided. Cannot fetch klines.")
        return None

    klines_df = None
    action_desc = f"fetching klines ({ccxt_timeframe}, limit {limit})"
    for attempt in range(MAX_API_RETRIES + 1):
        if _shutdown_requested:
             lg.info(f"Shutdown requested during kline fetch for {symbol}. Aborting.")
             return None
        try:
            lg.debug(f"{action_desc} for {symbol}, attempt {attempt + 1}/{MAX_API_RETRIES + 1}")
            # Ensure limit does not exceed exchange max
            fetch_limit = min(limit, BYBIT_API_KLINE_LIMIT)
            # Fetch the `fetch_limit` most recent candles. CCXT handles fetching latest candles by default.
            # No need to calculate `since` parameter here for fetching latest data.
            raw_klines = exchange.fetch_ohlcv(symbol, ccxt_timeframe, limit=fetch_limit)

            if raw_klines and isinstance(raw_klines, list) and len(raw_klines) > 0:
                # Convert raw list of lists into a pandas DataFrame
                # Columns: timestamp, open, high, low, close, volume
                # Timestamps from CCXT are in milliseconds UTC
                df = pd.DataFrame(raw_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                # Convert timestamp (milliseconds, UTC) to datetime and set as index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.set_index('timestamp')

                # Convert price and volume columns to Decimal for precision using the robust helper
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    # Use apply with _parse_decimal. If None is returned, it indicates an issue.
                    # Default to Decimal('0.0') if allow_none=False (safer for calculations).
                    # Log errors within _parse_decimal if values are problematic.
                    df[col] = df[col].apply(lambda x: _parse_decimal(x, col, lg, allow_none=False, default=Decimal('NaN'))) # Use NaN initially

                # Drop rows with any NaN values introduced by conversion errors
                original_len = len(df)
                df.dropna(inplace=True)
                if len(df) < original_len:
                     lg.warning(f"Dropped {original_len - len(df)} rows with invalid numeric data during kline processing for {symbol}.")

                # Check if DataFrame is still valid after processing
                if df.empty:
                     lg.warning(f"Kline data became empty after processing for {symbol}.")
                     klines_df = None # Treat as fetch failure
                else:
                     # Sort by timestamp just in case (though CCXT usually returns sorted)
                     df.sort_index(inplace=True)
                     lg.debug(f"Successfully fetched and processed {len(df)} klines for {symbol} ({ccxt_timeframe}).")
                     klines_df = df
                     break # Exit retry loop on success
            elif raw_klines is not None: # Empty list returned
                 lg.warning(f"Fetched empty kline data list for {symbol} ({ccxt_timeframe}, limit {fetch_limit}). Attempt {attempt + 1}.")
                 klines_df = pd.DataFrame() # Return empty DataFrame, not None
                 break # Consider empty list a valid (but possibly problematic) result, don't retry endlessly
            else: # None returned
                 lg.warning(f"Kline fetch returned None for {symbol} ({ccxt_timeframe}, limit {fetch_limit}). Attempt {attempt + 1}.")
                 klines_df = None # Treat as fetch failure for retry logic

        except Exception as e:
            # Let the helper handle logging and determine if retry is needed
            retry = _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
            if not retry:
                # If _handle_ccxt_exception returns False, it's a fatal error or max retries reached for transient errors
                return None # Return None on failure after retries or fatal error
            # If retry is True, the helper has already waited, continue the loop

    if klines_df is None:
        lg.error(f"{NEON_RED}Failed to fetch usable kline data for {symbol} ({ccxt_timeframe}, limit {limit}) after {MAX_API_RETRIES + 1} attempts.{RESET}")
        return None
    elif klines_df.empty:
         lg.warning(f"Returning empty DataFrame for {symbol} klines after processing.")
         return klines_df # Return empty DF if fetch succeeded but data was empty/invalid

    # Optional: Trim DataFrame to MAX_DF_LEN if it exceeds the limit
    if len(klines_df) > MAX_DF_LEN:
        lg.debug(f"Trimming DataFrame for {symbol} from {len(klines_df)} to {MAX_DF_LEN} rows.")
        klines_df = klines_df.tail(MAX_DF_LEN)

    return klines_df


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency using CCXT with retry logic.

    Args:
        exchange: The CCXT exchange instance.
        currency: The currency code (e.g., 'USDT', 'BTC'). Case-insensitive.
        logger: The logger instance for the operation.

    Returns:
        The available ('free') balance as a Decimal, or None if fetching fails after retries
        or if the currency is not found or has zero/invalid balance.
    """
    lg = logger
    target_currency = currency.upper() # Normalize currency code
    balance_value: Optional[Decimal] = None
    action_desc = f"fetching balance for {target_currency}"

    for attempt in range(MAX_API_RETRIES + 1):
        if _shutdown_requested:
             lg.info(f"Shutdown requested during balance fetch for {target_currency}. Aborting.")
             return None
        try:
            lg.debug(f"{action_desc}, attempt {attempt + 1}/{MAX_API_RETRIES + 1}")
            # Use fetch_balance to get all account balances
            balances = exchange.fetch_balance()

            # CCXT balance structure can vary slightly. Access safely.
            # The top level keys are usually currency codes.
            if balances and isinstance(balances, dict) and target_currency in balances:
                currency_balance = balances[target_currency]
                if isinstance(currency_balance, dict):
                    # Extract total and free (available) balance for the target currency
                    # 'free' balance is usually what's available for new orders
                    total_bal = _parse_decimal(currency_balance.get('total'), f'{target_currency} total', lg, allow_none=True)
                    free_bal = _parse_decimal(currency_balance.get('free'), f'{target_currency} free', lg, allow_none=True)

                    # Use the free balance for trading decisions
                    if free_bal is not None and free_bal >= Decimal('0'):
                        lg.debug(f"Successfully fetched balance for {target_currency}: Total={total_bal.normalize() if total_bal is not None else 'N/A'}, Free={free_bal.normalize()}.")
                        balance_value = free_bal
                        break # Exit retry loop on success
                    else:
                        # Free balance is None or negative, treat as unavailable
                         lg.warning(f"Invalid or zero/negative free balance found for {target_currency}: {currency_balance.get('free')}. Attempt {attempt + 1}.")
                         balance_value = Decimal('0.0') # Treat as zero available balance
                         break # Exit loop, balance is effectively zero
                else:
                    lg.warning(f"Balance data format unexpected for {target_currency}: {currency_balance}. Attempt {attempt + 1}.")
                    balance_value = None # Treat as fetch failure

            elif balances and isinstance(balances, dict):
                 # Balances fetched, but specific currency not found.
                 lg.warning(f"Balance fetched, but currency '{target_currency}' not found in response. Available: {list(balances.keys())[:10]}... Attempt {attempt + 1}.")
                 balance_value = Decimal('0.0') # Treat as zero balance for the requested currency
                 break # Exit loop, currency not held

            else:
                # fetch_balance returned None or empty/invalid dict
                lg.warning(f"Fetched empty or invalid balance data: {balances}. Attempt {attempt + 1}.")
                balance_value = None # Ensure None for retry logic to continue

        except Exception as e:
            retry = _handle_ccxt_exception(e, lg, action_desc, target_currency, attempt)
            if not retry:
                 # If not retrying, return None after logging
                 return None
            # If retry is True, the helper has already waited, continue the loop

    if balance_value is None:
        lg.error(f"{NEON_RED}Failed to fetch balance for {target_currency} after {MAX_API_RETRIES + 1} attempts.{RESET}")
        return None

    # Return the fetched free balance (could be Decimal('0.0'))
    return balance_value


def fetch_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[MarketInfo]:
    """
    Fetches and parses market information for a specific symbol from CCXT markets dictionary.
    Performs critical validation checks.

    Args:
        exchange: The CCXT exchange instance (with markets loaded).
        symbol: The trading symbol (e.g., 'BTC/USDT').
        logger: The logger instance for the symbol.

    Returns:
        A parsed MarketInfo TypedDict, or None if the symbol is not found
        or essential market data is incomplete/invalid.
    """
    lg = logger
    lg.debug(f"Fetching market info for {symbol}...")
    if not exchange.markets:
        lg.critical(f"{NEON_RED}FATAL ERROR: Exchange markets not loaded. Cannot fetch info for {symbol}.{RESET}")
        return None

    market = exchange.markets.get(symbol)

    if not market:
        lg.critical(f"{NEON_RED}FATAL ERROR: Market information for symbol '{symbol}' not found in exchange.markets.{RESET}")
        lg.critical(f"{NEON_RED}Ensure the symbol is correct and exists on the exchange in {'Testnet' if CONFIG.get('use_sandbox',True) else 'LIVE'} mode. Available markets: {list(exchange.markets.keys())[:10]}...{RESET}")
        return None

    # Perform basic structural checks on essential fields
    if not isinstance(market, dict):
        lg.critical(f"{NEON_RED}FATAL ERROR: Market data for '{symbol}' is not a dictionary.{RESET}")
        return None
    if market.get('symbol') != symbol:
        lg.critical(f"{NEON_RED}FATAL ERROR: Market data for '{symbol}' has incorrect symbol key: {market.get('symbol')}.{RESET}")
        return None
    if not isinstance(market.get('precision'), dict):
         lg.critical(f"{NEON_RED}FATAL ERROR: Market data for '{symbol}' is missing or has invalid 'precision' dictionary.{RESET}")
         return None
    if market['precision'].get('price') is None or market['precision'].get('amount') is None:
         lg.critical(f"{NEON_RED}FATAL ERROR: Market precision for '{symbol}' is missing 'price' or 'amount'. Precision: {market['precision']}{RESET}")
         return None
    if not isinstance(market.get('limits'), dict):
         lg.critical(f"{NEON_RED}FATAL ERROR: Market data for '{symbol}' is missing or has invalid 'limits' dictionary.{RESET}")
         return None
    if not isinstance(market['limits'].get('amount'), dict) or not isinstance(market['limits'].get('cost'), dict):
         lg.critical(f"{NEON_RED}FATAL ERROR: Market limits for '{symbol}' is missing 'amount' or 'cost' sub-dictionaries. Limits: {market['limits']}{RESET}")
         return None


    # Safely parse critical Decimal fields using the helper
    # Precision steps are crucial, use allow_none=False and check validity after
    price_precision_step_dec = _parse_decimal(market['precision'].get('price'), 'precision.price', lg, allow_none=False, default=Decimal('NaN'))
    amount_precision_step_dec = _parse_decimal(market['precision'].get('amount'), 'precision.amount', lg, allow_none=False, default=Decimal('NaN'))
    # Contract size defaults to 1 if not specified (common for spot/linear)
    contract_size_dec = _parse_decimal(market.get('contractSize'), 'contractSize', lg, allow_none=False, default=Decimal('1'))

    # Limits can be None if not specified by the exchange, use allow_none=True
    min_amount_dec = _parse_decimal(market['limits']['amount'].get('min'), 'limits.amount.min', lg, allow_none=True)
    max_amount_dec = _parse_decimal(market['limits']['amount'].get('max'), 'limits.amount.max', lg, allow_none=True)
    min_cost_dec = _parse_decimal(market['limits']['cost'].get('min'), 'limits.cost.min', lg, allow_none=True)
    max_cost_dec = _parse_decimal(market['limits']['cost'].get('max'), 'limits.cost.max', lg, allow_none=True)

    # Critical check: Ensure essential precision and contract size values were parsed successfully and are valid (> 0)
    if price_precision_step_dec is None or not price_precision_step_dec.is_finite() or price_precision_step_dec <= Decimal('0'):
         lg.critical(f"{NEON_RED}FATAL ERROR: Invalid or zero price precision step for '{symbol}': {market['precision'].get('price')} -> {price_precision_step_dec}. Cannot proceed.{RESET}")
         return None
    if amount_precision_step_dec is None or not amount_precision_step_dec.is_finite() or amount_precision_step_dec <= Decimal('0'):
         lg.critical(f"{NEON_RED}FATAL ERROR: Invalid or zero amount precision step for '{symbol}': {market['precision'].get('amount')} -> {amount_precision_step_dec}. Cannot proceed.{RESET}")
         return None
    if contract_size_dec is None or not contract_size_dec.is_finite() or contract_size_dec <= Decimal('0'):
         lg.critical(f"{NEON_RED}FATAL ERROR: Invalid or zero contract size for '{symbol}': {market.get('contractSize')} -> {contract_size_dec}. Cannot proceed.{RESET}")
         return None


    # Determine contract type string for logging
    contract_type_str = "Unknown"
    # Use CCXT unified flags where possible
    is_spot = market.get('spot', False)
    is_swap = market.get('swap', False)
    is_future = market.get('future', False)
    is_option = market.get('option', False)
    is_contract = is_swap or is_future or is_option or market.get('contract', False) # Combine flags
    is_linear = market.get('linear', False) if is_contract else False
    is_inverse = market.get('inverse', False) if is_contract else False

    if is_spot:
        contract_type_str = "Spot"
    elif is_contract:
        if is_linear:
            contract_type_str = "Linear"
        elif is_inverse:
            contract_type_str = "Inverse"
        elif is_option:
            contract_type_str = "Option"
        elif is_swap:
             contract_type_str = "Swap" # Perpetual Swap
        elif is_future:
             contract_type_str = "Future" # Dated Future
        else:
             # Fallback based on 'type' field if specific flags are missing
             market_type_str = str(market.get('type', '')).lower()
             if 'swap' in market_type_str: contract_type_str = "Swap"
             elif 'future' in market_type_str: contract_type_str = "Future"
             else: contract_type_str = "Contract" # Generic contract
    else:
         # Fallback if no flags are set
         market_type_str = str(market.get('type', '')).lower()
         if 'spot' in market_type_str: contract_type_str = "Spot"
         elif 'swap' in market_type_str: contract_type_str = "Swap"
         elif 'future' in market_type_str: contract_type_str = "Future"
         elif 'option' in market_type_str: contract_type_str = "Option"


    # Populate the MarketInfo TypedDict
    market_info: MarketInfo = {
        # Standard CCXT fields (copied directly, type hints need careful casting/checking)
        'id': str(market.get('id', '')),
        'symbol': str(market.get('symbol', '')),
        'base': str(market.get('base', '')),
        'quote': str(market.get('quote', '')),
        'settle': market.get('settle'), # Can be None
        'baseId': str(market.get('baseId', '')),
        'quoteId': str(market.get('quoteId', '')),
        'settleId': market.get('settleId'), # Can be None
        'type': str(market.get('type', '')),
        'spot': is_spot,
        'margin': market.get('margin', False),
        'swap': is_swap,
        'future': is_future,
        'option': is_option,
        'active': market.get('active'), # Can be None
        'contract': is_contract, # Use derived flag
        'linear': is_linear, # Use derived flag
        'inverse': is_inverse, # Use derived flag
        'quanto': market.get('quanto'), # Can be None
        'taker': float(market.get('taker', 0.0)), # Fees as float are usually fine
        'maker': float(market.get('maker', 0.0)),
        'contractSize': market.get('contractSize'), # Original raw field
        'expiry': market.get('expiry'), # Can be None
        'expiryDatetime': market.get('expiryDatetime'), # Can be None
        'strike': market.get('strike'), # Can be None
        'optionType': market.get('optionType'), # Can be None
        'precision': market.get('precision', {}), # Keep original precision dict
        'limits': market.get('limits', {}), # Keep original limits dict
        'info': market.get('info', {}), # Keep original raw info dict
        # Added/Derived fields (Decimal conversions and convenience flags)
        'is_contract': is_contract,
        'is_linear': is_linear,
        'is_inverse': is_inverse,
        'contract_type_str': contract_type_str,
        'min_amount_decimal': min_amount_dec,
        'max_amount_decimal': max_amount_dec,
        'min_cost_decimal': min_cost_dec,
        'max_cost_decimal': max_cost_dec,
        'amount_precision_step_decimal': amount_precision_step_dec, # Guaranteed non-None by check above
        'price_precision_step_decimal': price_precision_step_dec,   # Guaranteed non-None by check above
        'contract_size_decimal': contract_size_dec # Guaranteed non-None and > 0 by check above
    }

    lg.debug(f"Successfully parsed market info for {symbol}. Type: {market_info['contract_type_str']}, Amount Step: {market_info['amount_precision_step_decimal'].normalize()}, Price Step: {market_info['price_precision_step_decimal'].normalize()}.")
    if market_info['is_contract']:
         lg.debug(f"  Contract Info: Linear={market_info['is_linear']}, Inverse={market_info['is_inverse']}, Size={market_info['contract_size_decimal'].normalize()}")
    if market_info['min_amount_decimal'] is not None:
         lg.debug(f"  Min Amount: {market_info['min_amount_decimal'].normalize()}")
    if market_info['min_cost_decimal'] is not None:
         lg.debug(f"  Min Cost ({market_info['quote']}): {market_info['min_cost_decimal'].normalize()}")

    return market_info

def fetch_open_positions(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> List[PositionInfo]:
    """
    Fetches open positions for a symbol using CCXT with retry logic and parses them.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        logger: The logger instance for the symbol.

    Returns:
        A list of parsed PositionInfo TypedDicts for the given symbol.
        Returns an empty list if no open positions or fetching fails after retries.
        Each PositionInfo object includes bot state flags initialized to False.
    """
    lg = logger
    positions_list: List[PositionInfo] = []
    action_desc = f"fetching open positions for {symbol}"

    for attempt in range(MAX_API_RETRIES + 1):
        if _shutdown_requested:
             lg.info(f"Shutdown requested during position fetch for {symbol}. Aborting.")
             return []
        try:
            lg.debug(f"{action_desc}, attempt {attempt + 1}/{MAX_API_RETRIES + 1}")
            # fetch_positions can take a list of symbols or None for all
            # Passing the specific symbol is more efficient
            raw_positions = exchange.fetch_positions([symbol]) # Fetch only for the specific symbol

            # Filter for valid, open positions with non-zero size
            # CCXT often returns only open positions, but filter defensively.
            # Bybit V5 position size is in 'info.size' or unified 'contracts'.
            open_raw_positions = []
            if raw_positions and isinstance(raw_positions, list):
                for p in raw_positions:
                    if not isinstance(p, dict): continue
                    # Check unified 'contracts' field first, fallback to 'info.size'
                    raw_size = p.get('contracts') or p.get('info', {}).get('size')
                    pos_size_dec = _parse_decimal(raw_size, 'size', lg, allow_none=True)

                    if (p.get('symbol') == symbol and
                        p.get('side') in ['long', 'short'] and
                        pos_size_dec is not None and
                        pos_size_dec > POSITION_QTY_EPSILON): # Use epsilon for comparison
                        open_raw_positions.append(p)

            if not open_raw_positions:
                lg.debug(f"No open positions found for {symbol}.")
                return [] # Return empty list if no open positions

            lg.debug(f"Found {len(open_raw_positions)} raw open position(s) for {symbol}.")

            for raw_pos in open_raw_positions:
                # Safely parse relevant fields to Decimal
                # Use the size already parsed above
                pos_size_dec = _parse_decimal(raw_pos.get('contracts') or raw_pos.get('info', {}).get('size'), 'size', lg, allow_none=False) # Must be valid here
                entry_price_dec = _parse_decimal(raw_pos.get('entryPrice'), 'entryPrice', lg, allow_none=True)
                mark_price_dec = _parse_decimal(raw_pos.get('markPrice'), 'markPrice', lg, allow_none=True)
                liquidation_price_dec = _parse_decimal(raw_pos.get('liquidationPrice'), 'liquidationPrice', lg, allow_none=True)
                leverage_dec = _parse_decimal(raw_pos.get('leverage'), 'leverage', lg, allow_none=True)
                unrealized_pnl_dec = _parse_decimal(raw_pos.get('unrealizedPnl'), 'unrealizedPnl', lg, allow_none=True)
                notional_dec = _parse_decimal(raw_pos.get('notional'), 'notional', lg, allow_none=True)
                collateral_dec = _parse_decimal(raw_pos.get('collateral'), 'collateral', lg, allow_none=True)
                initial_margin_dec = _parse_decimal(raw_pos.get('initialMargin'), 'initialMargin', lg, allow_none=True)
                maintenance_margin_dec = _parse_decimal(raw_pos.get('maintenanceMargin'), 'maintenanceMargin', lg, allow_none=True)

                # Extract native SL/TP/TSL from raw 'info' or CCXT unified fields
                pos_info = raw_pos.get('info', {}) # Get the exchange-specific info dict

                # Check common unified and Bybit V5 specific keys
                sl_raw = pos_info.get('stopLoss') or raw_pos.get('stopLossPrice')
                tp_raw = pos_info.get('takeProfit') or raw_pos.get('takeProfitPrice')
                tsl_trigger_raw = pos_info.get('trailingStop') or raw_pos.get('trailingStopLoss') # Trigger price when active
                tsl_activation_raw = pos_info.get('activePrice') # Bybit V5 activation price

                # Parse raw protection strings to Decimal, allowing None
                sl_dec = _parse_decimal(sl_raw, 'stopLossPrice_raw', lg, allow_none=True)
                tp_dec = _parse_decimal(tp_raw, 'takeProfitPrice_raw', lg, allow_none=True)
                tsl_trigger_dec = _parse_decimal(tsl_trigger_raw, 'trailingStopPrice_raw', lg, allow_none=True)
                tsl_activation_dec = _parse_decimal(tsl_activation_raw, 'tslActivationPrice_raw', lg, allow_none=True)

                # Bybit V5 uses '0' or '0.0' string if SL/TP/TSL is NOT set. Treat these as None.
                if sl_dec is not None and sl_dec.is_zero(): sl_dec = None
                if tp_dec is not None and tp_dec.is_zero(): tp_dec = None
                if tsl_trigger_dec is not None and tsl_trigger_dec.is_zero(): tsl_trigger_dec = None
                if tsl_activation_dec is not None and tsl_activation_dec.is_zero(): tsl_activation_dec = None

                # Check for critical missing data for an open position
                if entry_price_dec is None or entry_price_dec <= Decimal('0'):
                     lg.warning(f"Found position for {symbol} with invalid/zero entry price: {raw_pos.get('entryPrice')}. Skipping.")
                     continue # Skip if entry price is invalid (makes BE/TSL calculation impossible)

                # Get position ID (can be in root or info)
                position_id = raw_pos.get('id') or pos_info.get('positionIdx') # Bybit V5 uses positionIdx in info (0, 1, 2)
                position_id_str = str(position_id) if position_id is not None else None


                parsed_position: PositionInfo = {
                    # Standard CCXT fields (copied directly or cast)
                    'id': position_id_str, # Use derived ID
                    'symbol': str(raw_pos.get('symbol', '')), # Should match input symbol
                    'timestamp': raw_pos.get('timestamp'), # Optional int
                    'datetime': raw_pos.get('datetime'), # Optional str
                    'contracts': raw_pos.get('contracts'), # Original float field
                    'contractSize': raw_pos.get('contractSize'), # Original raw field
                    'side': str(raw_pos.get('side', '')), # 'long' or 'short'
                    'notional': raw_pos.get('notional'), # Original raw field
                    'leverage': raw_pos.get('leverage'), # Original raw field
                    'unrealizedPnl': raw_pos.get('unrealizedPnl'), # Original raw field
                    'realizedPnl': raw_pos.get('realizedPnl'), # Original raw field
                    'collateral': raw_pos.get('collateral'), # Original raw field
                    'entryPrice': raw_pos.get('entryPrice'), # Original raw field
                    'markPrice': raw_pos.get('markPrice'), # Original raw field
                    'liquidationPrice': raw_pos.get('liquidationPrice'), # Original raw field
                    'marginMode': raw_pos.get('marginMode'), # e.g., 'isolated'
                    'hedged': raw_pos.get('hedged'), # e.g., False
                    'maintenanceMargin': raw_pos.get('maintenanceMargin'), # Original raw field
                    'maintenanceMarginPercentage': raw_pos.get('maintenanceMarginPercentage'), # Optional float
                    'initialMargin': raw_pos.get('initialMargin'), # Original raw field
                    'initialMarginPercentage': raw_pos.get('initialMarginPercentage'), # Optional float
                    'marginRatio': raw_pos.get('marginRatio'), # Optional float
                    'lastUpdateTimestamp': raw_pos.get('lastUpdateTimestamp'), # Optional int
                    'info': raw_pos.get('info', {}), # Full raw info dict
                    # Added/Derived Decimal fields
                    'size_decimal': pos_size_dec, # Guaranteed > 0 by filter above
                    'entryPrice_decimal': entry_price_dec, # Guaranteed > 0 by check above
                    'markPrice_decimal': mark_price_dec,
                    'liquidationPrice_decimal': liquidation_price_dec,
                    'leverage_decimal': leverage_dec,
                    'unrealizedPnl_decimal': unrealized_pnl_dec,
                    'notional_decimal': notional_dec,
                    'collateral_decimal': collateral_dec,
                    'initialMargin_decimal': initial_margin_dec,
                    'maintenanceMargin_decimal': maintenance_margin_dec,
                    # Protection Order Status (raw strings and parsed Decimals)
                    'stopLossPrice_raw': str(sl_raw) if sl_raw is not None else None,
                    'takeProfitPrice_raw': str(tp_raw) if tp_raw is not None else None,
                    'trailingStopPrice_raw': str(tsl_trigger_raw) if tsl_trigger_raw is not None else None, # TSL Trigger Price
                    'tslActivationPrice_raw': str(tsl_activation_raw) if tsl_activation_raw is not None else None, # TSL Activation Price
                    'stopLossPrice_dec': sl_dec,
                    'takeProfitPrice_dec': tp_dec,
                    'trailingStopPrice_dec': tsl_trigger_dec,
                    'tslActivationPrice_dec': tsl_activation_dec,
                    # Bot State Tracking (Initialized here, managed elsewhere by linking to SYMBOL_STATE)
                    'be_activated': False, # Default to False, will be updated from state if position is tracked
                    'tsl_activated': False # Default to False, will be updated from state if position is tracked
                }
                positions_list.append(parsed_position)

            # Successfully processed positions
            lg.debug(f"Parsed {len(positions_list)} open position(s) for {symbol}.")
            return positions_list

        except Exception as e:
            retry = _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
            if not retry:
                 # If not retrying, return empty list after logging
                 return []
            # If retry is True, the helper has already waited, continue the loop

    lg.error(f"{NEON_RED}Failed to fetch and parse open positions for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return [] # Return empty list on final failure


# --- Strategy Logic (Placeholders) ---
# TODO: Implement the actual Volumatic Trend and Order Block strategy logic here.
# The functions below provide a basic structure and placeholder calculations.

def calculate_indicators(dataframe: pd.DataFrame, config_params: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """
    Calculates strategy indicators (Volumatic Trend placeholders, ATR, Pivots)
    and adds them as new columns to the DataFrame.

    Args:
        dataframe: The input pandas DataFrame with kline data (OHLCV, Decimal types).
        config_params: The 'strategy_params' dictionary from the configuration.
        logger: The logger instance for the symbol.

    Returns:
        The DataFrame with added indicator columns. Original Decimal types preserved where possible.
    """
    lg = logger
    df = dataframe.copy() # Work on a copy

    if df.empty:
        lg.warning("Cannot calculate indicators on an empty DataFrame.")
        return df

    # Ensure necessary columns are Decimal and finite before TA calculation
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
         lg.error("DataFrame missing required OHLCV columns for indicator calculation.")
         return pd.DataFrame() # Return empty DF if critical columns missing

    # --- Convert Decimal columns to float temporarily for pandas_ta ---
    # Pandas_ta generally works best with floats. Convert back critical results if needed.
    # Be mindful of potential precision loss, but acceptable for most standard indicators.
    df_float = pd.DataFrame(index=df.index)
    try:
        for col in required_cols:
             # Safely convert Decimal to float, handling potential NaNs or Infs if they slipped through
             df_float[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop rows where conversion to float failed
        df_float.dropna(subset=required_cols, inplace=True)
        if df_float.empty:
             lg.warning("DataFrame became empty after converting OHLCV to float for indicators.")
             return pd.DataFrame()
    except Exception as e:
         lg.error(f"{NEON_RED}Error converting Decimal columns to float for TA calculation: {e}{RESET}")
         return pd.DataFrame()


    # --- Volumatic Trend Placeholder Calculation ---
    # TODO: Replace this with actual Volumatic Trend logic.
    vt_length = config_params.get("vt_length", DEFAULT_VT_LENGTH)
    vt_atr_period = config_params.get("vt_atr_period", DEFAULT_VT_ATR_PERIOD)
    vt_vol_ema_length = config_params.get("vt_vol_ema_length", DEFAULT_VT_VOL_EMA_LENGTH)
    vt_atr_multiplier = config_params.get("vt_atr_multiplier", DEFAULT_VT_ATR_MULTIPLIER)

    try:
        lg.debug("Calculating placeholder indicators (EMA/SMA cross, ATR, Volume EMA)...")
        # Example: Simple EMA/SMA cross as placeholder for VT core trend
        df_float['SMA_VT'] = ta.sma(df_float['close'], length=vt_length)
        df_float['EMA_VT'] = ta.ema(df_float['close'], length=vt_length // 2 if vt_length > 1 else 1) # Shorter EMA for cross

        # Calculate ATR
        df_float['ATR'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=vt_atr_period)

        # Example: Simple upper/lower bands (like Keltner Channels based on EMA and ATR) - Placeholder for VT bands
        df_float['VT_Upper_Band'] = df_float['EMA_VT'] + df_float['ATR'] * vt_atr_multiplier
        df_float['VT_Lower_Band'] = df_float['EMA_VT'] - df_float['ATR'] * vt_atr_multiplier

        # Placeholder for Volume Analysis (e.g., simple EMA of Volume)
        df_float['Volume_EMA'] = ta.ema(df_float['volume'], length=vt_vol_ema_length)

        lg.debug("Placeholder indicators calculated.")

    except Exception as e:
        lg.error(f"{NEON_RED}Error calculating placeholder Volumatic Trend/ATR indicators: {e}{RESET}", exc_info=True)
        # Ensure indicator columns exist even if calculation failed, filled with NaN
        for col in ['SMA_VT', 'EMA_VT', 'ATR', 'VT_Upper_Band', 'VT_Lower_Band', 'Volume_EMA']:
             if col not in df_float.columns:
                  df_float[col] = np.nan # Add column with NaN if missing


    # --- Pivot Point Calculation for Order Blocks ---
    # TODO: Refine pivot logic and OB definition based on actual strategy rules.
    ph_left = config_params.get("ph_left", DEFAULT_PH_LEFT)
    ph_right = config_params.get("ph_right", DEFAULT_PH_RIGHT)
    pl_left = config_params.get("pl_left", DEFAULT_PL_LEFT)
    pl_right = config_params.get("pl_right", DEFAULT_PL_RIGHT)
    ob_source = config_params.get("ob_source", DEFAULT_OB_SOURCE) # "Wicks" or "Body"

    try:
        lg.debug("Calculating placeholder Pivot Points...")
        # Define source columns based on ob_source
        # For Body source, a more complex definition considering candle direction is needed.
        # This simple version uses High/Low for Wicks, Close/Open for Body (approximate).
        high_src_col = 'high' if ob_source == 'Wicks' else 'close' # Simplistic body high
        low_src_col = 'low' if ob_source == 'Wicks' else 'open'   # Simplistic body low

        # Use pandas_ta for pivot points if available and suitable, or implement manually.
        # Manual implementation (can be slow):
        df_float['is_pivot_high'] = False
        df_float['is_pivot_low'] = False

        # Iterate using rolling windows (potentially faster than list comprehension for large data)
        # Requires careful handling of window edges.
        # Using a simpler iterative approach for clarity here, acknowledge performance impact.
        for i in range(ph_left, len(df_float) - ph_right):
             window = df_float[high_src_col].iloc[i - ph_left : i + ph_right + 1]
             if not window.empty and df_float[high_src_col].iloc[i] == window.max():
                  df_float.loc[df_float.index[i], 'is_pivot_high'] = True

        for i in range(pl_left, len(df_float) - pl_right):
             window = df_float[low_src_col].iloc[i - pl_left : i + pl_right + 1]
             if not window.empty and df_float[low_src_col].iloc[i] == window.min():
                  df_float.loc[df_float.index[i], 'is_pivot_low'] = True

        lg.debug("Placeholder Pivot Points calculated.")

    except Exception as e:
         lg.error(f"{NEON_RED}Error calculating Pivot Points for Order Blocks: {e}{RESET}", exc_info=True)
         # Ensure pivot columns exist even if calculation failed
         if 'is_pivot_high' not in df_float.columns: df_float['is_pivot_high'] = False
         if 'is_pivot_low' not in df_float.columns: df_float['is_pivot_low'] = False


    # --- Merge calculated float indicators back into the original Decimal DataFrame ---
    # Reindex df_float to match df's index in case rows were dropped
    df_float = df_float.reindex(df.index)

    indicator_cols = ['SMA_VT', 'EMA_VT', 'ATR', 'VT_Upper_Band', 'VT_Lower_Band', 'Volume_EMA', 'is_pivot_high', 'is_pivot_low']
    for col in indicator_cols:
        if col in df_float.columns:
            if col in ['is_pivot_high', 'is_pivot_low']:
                # Boolean columns - fillna with False
                df[col] = df_float[col].fillna(False).astype(bool)
            else:
                # Numeric columns - convert back to Decimal, fill NaN with Decimal('NaN')
                df[col] = df_float[col].apply(lambda x: _parse_decimal(x, col, lg, allow_none=True, default=Decimal('NaN')))
        else:
            # If indicator calculation failed entirely, add the column filled appropriately
             df[col] = False if col in ['is_pivot_high', 'is_pivot_low'] else Decimal('NaN')
             lg.warning(f"Indicator column '{col}' was not found after calculation, filling with default.")


    lg.debug(f"Finished calculating indicators. DataFrame shape: {df.shape}")
    return df

def identify_order_blocks(dataframe: pd.DataFrame, config_params: Dict[str, Any], logger: logging.Logger) -> Tuple[List[OrderBlock], List[OrderBlock]]:
    """
    Identifies potential Order Blocks (OBs) based on pivot points and determines
    their active status based on subsequent price action within the fetched data.

    Args:
        dataframe: The input pandas DataFrame with kline data and pivot indicators.
                   Assumes Decimal types for OHLC.
        config_params: The 'strategy_params' dictionary from the configuration.
        logger: The logger instance for the symbol.

    Returns:
        A tuple containing two lists: (active_bull_boxes, active_bear_boxes).
        OrderBlocks contain Decimal prices.
    """
    lg = logger
    active_bull_boxes: List[OrderBlock] = []
    active_bear_boxes: List[OrderBlock] = []

    # TODO: Implement robust OB identification and validation logic.
    # This includes considering candle patterns (engulfing, momentum), volume at OB, etc.
    # The current logic is a simplified placeholder based only on pivots and basic violation checks.

    ob_source = config_params.get("ob_source", DEFAULT_OB_SOURCE)
    ob_max_boxes = config_params.get("ob_max_boxes", DEFAULT_OB_MAX_BOXES)
    # ob_extend = config_params.get("ob_extend", DEFAULT_OB_EXTEND) # Not used in current logic

    if dataframe.empty:
        lg.warning("Cannot identify order blocks on an empty DataFrame.")
        return [], []

    # Ensure necessary columns exist from calculate_indicators
    if 'is_pivot_high' not in dataframe.columns or 'is_pivot_low' not in dataframe.columns:
         lg.error("DataFrame missing 'is_pivot_high' or 'is_pivot_low' columns. Cannot identify OBs.")
         return [], []
    if not all(col in dataframe.columns for col in ['open', 'high', 'low', 'close']):
         lg.error("DataFrame missing OHLC columns. Cannot identify OBs.")
         return [], []

    # Identify potential OB candidates based on pivots
    potential_bull_obs_idx = dataframe.index[dataframe['is_pivot_low']]
    potential_bear_obs_idx = dataframe.index[dataframe['is_pivot_high']]

    lg.debug(f"Found {len(potential_bull_obs_idx)} potential Bullish OB pivots and {len(potential_bear_obs_idx)} potential Bearish OB pivots.")

    # Helper to check if a subsequent close price violates an OB
    def check_violation(ob_type: str, ob_bottom: Decimal, ob_top: Decimal, check_price: Decimal) -> bool:
         if ob_type == "BULL":
             # Bullish OB is violated if price closes *below* the bottom
             return check_price < ob_bottom
         elif ob_type == "BEAR":
             # Bearish OB is violated if price closes *above* the top
             return check_price > ob_top
         return False # Should not happen

    # Process potential Bullish OBs (associated with Pivot Lows)
    for index in potential_bull_obs_idx:
        try:
            row = dataframe.loc[index]
            # Define the box range based on source
            # TODO: Refine Body OB definition (e.g., use the body of the candle *before* the pivot low?)
            if ob_source == "Wicks":
                bottom = row['low']
                top = row['high']
            else: # "Body" - simplistic: use Open/Close of the pivot candle
                bottom = min(row['open'], row['close'])
                top = max(row['open'], row['close'])

            # Ensure prices are valid Decimals and form a valid range
            if not all(isinstance(p, Decimal) and p.is_finite() for p in [bottom, top]) or bottom >= top:
                 lg.debug(f"Skipping invalid bullish OB prices at {index}: bottom={bottom}, top={top}")
                 continue

            ob_id = f"BULL_{index.value}_{ob_source}" # Unique ID based on timestamp and source

            # Check for violation by subsequent candles within the fetched data
            violated = False
            violation_ts = None
            subsequent_candles = dataframe.loc[index:].iloc[1:] # Get candles after the pivot candle
            for sub_index, sub_row in subsequent_candles.iterrows():
                 sub_close = sub_row['close']
                 if isinstance(sub_close, Decimal) and sub_close.is_finite():
                      if check_violation("BULL", bottom, top, sub_close):
                           violated = True
                           violation_ts = sub_index # Timestamp of the violating candle
                           break # Violated, stop checking this OB

            # If not violated within the fetched data, consider it "active" for this analysis cycle
            # NOTE: This does not track violations across different data fetches (stateful tracking needed for full robustness)
            if not violated:
                # Limit the number of active boxes kept
                if len(active_bull_boxes) < ob_max_boxes:
                    active_bull_boxes.append({
                        'id': ob_id,
                        'type': 'BULL',
                        'timestamp': index, # pd.Timestamp
                        'top': top,
                        'bottom': bottom,
                        'active': True, # Considered active based on this fetch
                        'violated': False,
                        'violation_ts': None
                    })
                # else: # Pruning logic if needed (e.g., remove oldest if max exceeded)
                #      pass
            # else:
                 # lg.debug(f"Bullish OB at {index} identified but violated at {violation_ts}.")

        except Exception as e:
            lg.error(f"{NEON_RED}Error processing potential bullish OB at index {index}: {e}{RESET}", exc_info=True)

    # Process potential Bearish OBs (associated with Pivot Highs)
    for index in potential_bear_obs_idx:
        try:
            row = dataframe.loc[index]
            # Define the box range based on source
            # TODO: Refine Body OB definition (e.g., use the body of the candle *before* the pivot high?)
            if ob_source == "Wicks":
                bottom = row['low']
                top = row['high']
            else: # "Body" - simplistic: use Open/Close of the pivot candle
                bottom = min(row['open'], row['close'])
                top = max(row['open'], row['close'])

            # Ensure prices are valid Decimals and form a valid range
            if not all(isinstance(p, Decimal) and p.is_finite() for p in [bottom, top]) or bottom >= top:
                 lg.debug(f"Skipping invalid bearish OB prices at {index}: bottom={bottom}, top={top}")
                 continue

            ob_id = f"BEAR_{index.value}_{ob_source}"

            # Check for violation by subsequent candles within the fetched data
            violated = False
            violation_ts = None
            subsequent_candles = dataframe.loc[index:].iloc[1:] # Get candles after the pivot candle
            for sub_index, sub_row in subsequent_candles.iterrows():
                 sub_close = sub_row['close']
                 if isinstance(sub_close, Decimal) and sub_close.is_finite():
                      if check_violation("BEAR", bottom, top, sub_close):
                           violated = True
                           violation_ts = sub_index
                           break

            # If not violated within the fetched data, consider it "active"
            if not violated:
                if len(active_bear_boxes) < ob_max_boxes:
                    active_bear_boxes.append({
                        'id': ob_id,
                        'type': 'BEAR',
                        'timestamp': index, # pd.Timestamp
                        'top': top,
                        'bottom': bottom,
                        'active': True,
                        'violated': False,
                        'violation_ts': None
                    })
                # else: # Pruning logic
                #      pass
            # else:
                 # lg.debug(f"Bearish OB at {index} identified but violated at {violation_ts}.")

        except Exception as e:
            lg.error(f"{NEON_RED}Error processing potential bearish OB at index {index}: {e}{RESET}", exc_info=True)


    # Sort active boxes by timestamp (most recent first might be more relevant for signals)
    active_bull_boxes.sort(key=lambda x: x['timestamp'], reverse=True)
    active_bear_boxes.sort(key=lambda x: x['timestamp'], reverse=True)

    lg.debug(f"Finished identifying order blocks. Found {len(active_bull_boxes)} active bullish and {len(active_bear_boxes)} active bearish boxes (within fetched data).")

    return active_bull_boxes, active_bear_boxes


def analyze_strategy(dataframe: pd.DataFrame, config_params: Dict[str, Any], logger: logging.Logger) -> StrategyAnalysisResults:
    """
    Analyzes the DataFrame with indicators to determine the current trend,
    bands, ATR, and active order blocks based on the strategy rules.

    Args:
        dataframe: The input pandas DataFrame with calculated indicators and pivots.
                   Assumes Decimal types for numeric columns where appropriate.
        config_params: The 'strategy_params' dictionary from the configuration.
        logger: The logger instance for the symbol.

    Returns:
        A StrategyAnalysisResults TypedDict containing the analysis outcome.
    """
    lg = logger

    # Default result structure in case of early exit
    default_result: StrategyAnalysisResults = {
        'dataframe': dataframe,
        'last_close': Decimal('NaN'),
        'current_trend_up': None,
        'trend_just_changed': False,
        'active_bull_boxes': [],
        'active_bear_boxes': [],
        'vol_norm_int': None,
        'atr': Decimal('NaN'),
        'upper_band': Decimal('NaN'),
        'lower_band': Decimal('NaN')
    }

    if dataframe.empty:
        lg.warning("Cannot analyze strategy on an empty DataFrame.")
        return default_result

    # Ensure necessary columns exist from calculate_indicators
    # Check for placeholder indicators used in current logic
    required_cols = ['close', 'ATR', 'VT_Upper_Band', 'VT_Lower_Band', 'SMA_VT', 'EMA_VT']
    if not all(col in dataframe.columns for col in required_cols):
         lg.error("DataFrame missing required indicator columns for strategy analysis.")
         # Try to return partial results if possible
         default_result['last_close'] = dataframe['close'].iloc[-1] if 'close' in dataframe.columns and not dataframe.empty else Decimal('NaN')
         default_result['atr'] = dataframe['ATR'].iloc[-1] if 'ATR' in dataframe.columns and not dataframe.empty else Decimal('NaN')
         # Bands might be missing if ATR failed etc.
         default_result['upper_band'] = dataframe['VT_Upper_Band'].iloc[-1] if 'VT_Upper_Band' in dataframe.columns and not dataframe.empty else Decimal('NaN')
         default_result['lower_band'] = dataframe['VT_Lower_Band'].iloc[-1] if 'VT_Lower_Band' in dataframe.columns and not dataframe.empty else Decimal('NaN')
         return default_result

    # Ensure we have at least 2 rows for trend change detection
    if len(dataframe) < 2:
        lg.warning("Not enough data ( < 2 rows) to determine trend change.")
        # Still try to determine current trend if possible
        last_row = dataframe.iloc[-1]
        current_sma = last_row['SMA_VT']
        current_ema = last_row['EMA_VT']
        current_trend_up: Optional[bool] = None
        if isinstance(current_ema, Decimal) and isinstance(current_sma, Decimal) and current_ema.is_finite() and current_sma.is_finite():
             if current_ema > current_sma: current_trend_up = True
             elif current_ema < current_sma: current_trend_up = False

        active_bull_boxes, active_bear_boxes = identify_order_blocks(dataframe, config_params, logger)
        last_close = last_row['close'] if isinstance(last_row.get('close'), Decimal) else Decimal('NaN')
        current_atr = last_row['ATR'] if isinstance(last_row.get('ATR'), Decimal) else Decimal('NaN')
        upper_band = last_row['VT_Upper_Band'] if isinstance(last_row.get('VT_Upper_Band'), Decimal) else Decimal('NaN')
        lower_band = last_row['VT_Lower_Band'] if isinstance(last_row.get('VT_Lower_Band'), Decimal) else Decimal('NaN')

        return {
            'dataframe': dataframe,
            'last_close': last_close,
            'current_trend_up': current_trend_up,
            'trend_just_changed': False, # Cannot determine change
            'active_bull_boxes': active_bull_boxes,
            'active_bear_boxes': active_bear_boxes,
            'vol_norm_int': None, # Placeholder
            'atr': current_atr,
            'upper_band': upper_band,
            'lower_band': lower_band
        }

    # --- Volumatic Trend Analysis (Placeholder: EMA/SMA Cross) ---
    # TODO: Replace with actual Volumatic Trend logic.
    last_row = dataframe.iloc[-1]
    prev_row = dataframe.iloc[-2]

    current_sma = last_row['SMA_VT']
    current_ema = last_row['EMA_VT']
    prev_sma = prev_row['SMA_VT']
    prev_ema = prev_row['EMA_VT']

    # Check if indicator values are valid Decimals
    valid_current = isinstance(current_ema, Decimal) and isinstance(current_sma, Decimal) and current_ema.is_finite() and current_sma.is_finite()
    valid_prev = isinstance(prev_ema, Decimal) and isinstance(prev_sma, Decimal) and prev_ema.is_finite() and prev_sma.is_finite()

    current_trend_up: Optional[bool] = None
    if valid_current:
        if current_ema > current_sma: current_trend_up = True
        elif current_ema < current_sma: current_trend_up = False

    prev_trend_up: Optional[bool] = None
    if valid_prev:
        if prev_ema > prev_sma: prev_trend_up = True
        elif prev_ema < prev_sma: prev_trend_up = False

    # Check if trend just changed on the last candle
    trend_just_changed = False
    # Trend changed if previous trend was determined and different from current determined trend
    if current_trend_up is not None and prev_trend_up is not None and current_trend_up != prev_trend_up:
          trend_just_changed = True
          lg.debug(f"Trend just changed: {'Up' if current_trend_up else 'Down'}")
    # Handle case where trend becomes undetermined (e.g., cross) - is this a change? Depends on strategy.
    # elif current_trend_up is None and prev_trend_up is not None:
    #      trend_just_changed = True # Changed from determined to undetermined
    #      lg.debug("Trend just changed: Undetermined")


    # --- Order Block Identification ---
    # This calls the separate function to get active OBs from the DataFrame
    active_bull_boxes, active_bear_boxes = identify_order_blocks(dataframe, config_params, logger)

    # --- Get latest indicator values ---
    last_close = last_row['close'] if isinstance(last_row.get('close'), Decimal) and last_row['close'].is_finite() else Decimal('NaN')
    current_atr = last_row['ATR'] if isinstance(last_row.get('ATR'), Decimal) and last_row['ATR'].is_finite() else Decimal('NaN')
    upper_band = last_row['VT_Upper_Band'] if isinstance(last_row.get('VT_Upper_Band'), Decimal) and last_row['VT_Upper_Band'].is_finite() else Decimal('NaN')
    lower_band = last_row['VT_Lower_Band'] if isinstance(last_row.get('VT_Lower_Band'), Decimal) and last_row['VT_Lower_Band'].is_finite() else Decimal('NaN')
    # Placeholder for normalized volume
    vol_norm_int = None # TODO: Implement actual normalized volume logic if needed

    # Log analysis summary
    trend_str = 'Up' if current_trend_up is True else 'Down' if current_trend_up is False else 'Sideways/Undetermined'
    atr_str = current_atr.normalize() if current_atr is not None and current_atr.is_finite() else 'N/A'
    lg.debug(f"Strategy analysis complete. Trend: {trend_str}, Last Close: {last_close.normalize() if last_close.is_finite() else 'N/A'}, ATR: {atr_str}")
    lg.debug(f"Active OBs: Bullish={len(active_bull_boxes)}, Bearish={len(active_bear_boxes)}")


    # Return the analysis results
    return {
        'dataframe': dataframe,
        'last_close': last_close,
        'current_trend_up': current_trend_up,
        'trend_just_changed': trend_just_changed,
        'active_bull_boxes': active_bull_boxes,
        'active_bear_boxes': active_bear_boxes,
        'vol_norm_int': vol_norm_int,
        'atr': current_atr,
        'upper_band': upper_band,
        'lower_band': lower_band
    }


def generate_signal(analysis_results: StrategyAnalysisResults, current_positions: List[PositionInfo], config_params: Dict[str, Any], market_info: MarketInfo, exchange: ccxt.Exchange, logger: logging.Logger) -> SignalResult:
    """
    Generates a trading signal ("BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT", "NONE")
    based on the strategy analysis results and current open positions.

    Args:
        analysis_results: The results from analyze_strategy.
        current_positions: A list of current open PositionInfo objects for the symbol.
        config_params: The 'strategy_params' dictionary from the configuration.
        market_info: MarketInfo TypedDict for the symbol.
        exchange: The CCXT exchange instance (needed for formatting SL/TP).
        logger: The logger instance for the symbol.

    Returns:
        A SignalResult TypedDict indicating the action to take and the reason.
        Includes calculated initial SL/TP prices (Decimal) for entry signals.
    """
    lg = logger

    # TODO: Replace placeholder signal logic with actual strategy rules.
    # This requires combining Volumatic Trend signals, OB interactions, volume analysis, etc.

    # Extract analysis results
    df = analysis_results['dataframe']
    last_close = analysis_results['last_close']
    current_trend_up = analysis_results['current_trend_up']
    trend_just_changed = analysis_results['trend_just_changed']
    active_bull_boxes = analysis_results['active_bull_boxes']
    active_bear_boxes = analysis_results['active_bear_boxes']
    current_atr = analysis_results['atr']

    # Extract relevant config parameters
    ob_entry_proximity_factor = _parse_decimal(config_params.get("ob_entry_proximity_factor"), 'ob_entry_proximity_factor', lg, default=Decimal(DEFAULT_OB_ENTRY_PROXIMITY_FACTOR))
    ob_exit_proximity_factor = _parse_decimal(config_params.get("ob_exit_proximity_factor"), 'ob_exit_proximity_factor', lg, default=Decimal(DEFAULT_OB_EXIT_PROXIMITY_FACTOR))
    initial_sl_atr_mult = _parse_decimal(CONFIG.get("protection", {}).get("initial_stop_loss_atr_multiple"), 'initial_sl_atr_mult', lg, default=Decimal(DEFAULT_INITIAL_STOP_LOSS_ATR_MULTIPLE))
    initial_tp_atr_mult = _parse_decimal(CONFIG.get("protection", {}).get("initial_take_profit_atr_multiple"), 'initial_tp_atr_mult', lg, default=Decimal(DEFAULT_INITIAL_TAKE_PROFIT_ATR_MULTIPLE))

    # Default signal result
    signal_result: SignalResult = {'signal': 'NONE', 'reason': 'No signal conditions met', 'initial_sl_price': None, 'initial_tp_price': None}

    # Check if minimum data is available for signal generation
    if current_trend_up is None or not isinstance(current_atr, Decimal) or not current_atr.is_finite() or current_atr <= Decimal('0') or not isinstance(last_close, Decimal) or not last_close.is_finite() or last_close <= Decimal('0'):
        lg.debug("Signal generation: Trend, ATR, or Last Close not valid/determined. Returning NONE.")
        signal_result['reason'] = 'Insufficient data or indicators not ready'
        return signal_result

    # --- Determine if an exit signal exists for any current position ---
    # Assuming a maximum of one position per symbol based on current logic/config default
    current_position: Optional[PositionInfo] = current_positions[0] if current_positions else None

    if current_position:
        pos_side = current_position['side']
        entry_price = current_position['entryPrice_decimal']
        if entry_price is None: # Should not happen if position exists, but check
            lg.warning(f"Position {current_position.get('id')} exists but has no entry price. Cannot generate exit signal.")
            return signal_result # Return NONE

        # Exit condition 1: Trend Reversal (Placeholder: VT Cross)
        if pos_side == 'long' and current_trend_up is False and trend_just_changed:
            lg.info(f"{NEON_YELLOW}Exit Signal: Trend reversed to Down (Placeholder VT) for active LONG position.{RESET}")
            signal_result['signal'] = 'EXIT_LONG'
            signal_result['reason'] = 'Trend reversal (Placeholder VT)'
            return signal_result
        elif pos_side == 'short' and current_trend_up is True and trend_just_changed:
            lg.info(f"{NEON_YELLOW}Exit Signal: Trend reversed to Up (Placeholder VT) for active SHORT position.{RESET}")
            signal_result['signal'] = 'EXIT_SHORT'
            signal_result['reason'] = 'Trend reversal (Placeholder VT)'
            return signal_result

        # Exit condition 2: Price nearing/hitting opposite OB (Placeholder)
        # Find the closest opposite OB that has not been violated
        closest_opposite_ob: Optional[OrderBlock] = None
        ob_check_price = last_close # Use last close price for checking proximity

        if pos_side == 'long' and active_bear_boxes:
             # Find the closest active Bearish OB above current price
             relevant_obs = [ob for ob in active_bear_boxes if ob['bottom'] >= ob_check_price]
             if relevant_obs:
                 closest_opposite_ob = min(relevant_obs, key=lambda ob: ob['bottom']) # Closest based on bottom edge
                 # Check proximity to the bottom of this bearish OB
                 if ob_check_price >= closest_opposite_ob['bottom'] / ob_exit_proximity_factor: # Within X% proximity below bottom
                      lg.info(f"{NEON_YELLOW}Exit Signal: Price {last_close.normalize()} near Bearish OB bottom {closest_opposite_ob['bottom'].normalize()} for active LONG position.{RESET}")
                      signal_result['signal'] = 'EXIT_LONG'
                      signal_result['reason'] = f"Price near Bearish OB {closest_opposite_ob['id']}"
                      return signal_result

        elif pos_side == 'short' and active_bull_boxes:
             # Find the closest active Bullish OB below current price
             relevant_obs = [ob for ob in active_bull_boxes if ob['top'] <= ob_check_price]
             if relevant_obs:
                  closest_opposite_ob = max(relevant_obs, key=lambda ob: ob['top']) # Closest based on top edge
                  # Check proximity to the top of this bullish OB
                  if ob_check_price <= closest_opposite_ob['top'] * ob_exit_proximity_factor: # Within X% proximity above top
                      lg.info(f"{NEON_YELLOW}Exit Signal: Price {last_close.normalize()} near Bullish OB top {closest_opposite_ob['top'].normalize()} for active SHORT position.{RESET}")
                      signal_result['signal'] = 'EXIT_SHORT'
                      signal_result['reason'] = f"Price near Bullish OB {closest_opposite_ob['id']}"
                      return signal_result

        # Exit condition 3: Stop Loss or Take Profit hit (handled natively by exchange, bot detects position closure)

        # Exit condition 4: Other Strategy Rules (Placeholder - add your logic here)
        # TODO: Implement strategy-specific exit rules (e.g., band breaks, volume spikes)

        # If none of the explicit exit conditions met, but still in position
        lg.debug(f"Holding active {pos_side.upper()} position. No explicit exit signal generated.")
        signal_result['signal'] = 'HOLD'
        signal_result['reason'] = 'Holding existing position, no exit conditions met'
        return signal_result

    # --- Determine if an entry signal exists (only if no current position) ---
    else: # No current position open
        # Entry conditions:
        # 1. Trend is determined (Up for BUY, Down for SELL)
        # 2. Price is near a relevant Order Block (Bullish OB for BUY, Bearish OB for SELL)
        # 3. Other strategy-specific entry rules (e.g., volume confirmation, break of structure)
        # TODO: Implement robust entry logic combining VT, OB, Volume etc.

        # Entry condition 1: Trend Check (Placeholder: VT Cross)
        potential_entry_side: Optional[str] = None
        if current_trend_up is True:
             potential_entry_side = 'long'
        elif current_trend_up is False:
             potential_entry_side = 'short'
        else:
             lg.debug("Signal generation: Trend is undetermined. No entry signal.")
             return signal_result # Return NONE

        # Entry condition 2: Price Proximity to Relevant Order Block (Placeholder)
        entry_ob: Optional[OrderBlock] = None
        entry_ob_edge: Optional[Decimal] = None # The specific edge (top/bottom) we are reacting to

        if potential_entry_side == 'long' and active_bull_boxes:
             # For a Long entry, look for price near the top of a Bullish OB *below* current price
             relevant_obs = [ob for ob in active_bull_boxes if ob['top'] <= last_close] # Only consider OBs below/at current price
             if relevant_obs:
                 # Closest Bullish OB below current price (highest top)
                 closest_ob = max(relevant_obs, key=lambda ob: ob['top'])
                 # Check if current price is within entry proximity *above* the top of this OB
                 if last_close <= closest_ob['top'] * ob_entry_proximity_factor: # Within X% proximity above top
                      entry_ob = closest_ob
                      entry_ob_edge = entry_ob['top']
                      lg.debug(f"Potential LONG entry signal: Price {last_close.normalize()} near Bullish OB top {entry_ob_edge.normalize()}.")

        elif potential_entry_side == 'short' and active_bear_boxes:
             # For a Short entry, look for price near the bottom of a Bearish OB *above* current price
             relevant_obs = [ob for ob in active_bear_boxes if ob['bottom'] >= last_close] # Only consider OBs above/at current price
             if relevant_obs:
                 # Closest Bearish OB above current price (lowest bottom)
                 closest_ob = min(relevant_obs, key=lambda ob: ob['bottom'])
                 # Check if current price is within entry proximity *below* the bottom of this OB
                 if last_close >= closest_ob['bottom'] / ob_entry_proximity_factor: # Within X% proximity below bottom
                      entry_ob = closest_ob
                      entry_ob_edge = entry_ob['bottom']
                      lg.debug(f"Potential SHORT entry signal: Price {last_close.normalize()} near Bearish OB bottom {entry_ob_edge.normalize()}.")

        # Entry condition 3: Other Strategy Rules (Placeholder - add your logic here)
        # TODO: Add checks for volume confirmation, structure breaks, band interactions etc.


        # --- Final Signal Determination (Placeholder: Trend + OB Proximity) ---
        initial_sl_price: Optional[Decimal] = None
        initial_tp_price: Optional[Decimal] = None

        if potential_entry_side == 'long' and entry_ob and entry_ob_edge:
             signal_result['signal'] = 'BUY'
             signal_result['reason'] = f"Up trend (VT Placeholder) & price near Bullish OB {entry_ob['id']} @ {entry_ob_edge.normalize()}"
             # Calculate initial SL/TP based on ATR and entry price (last_close)
             if current_atr > Decimal('0'):
                  sl_raw = last_close - current_atr * initial_sl_atr_mult
                  # Ensure SL is below the OB bottom for robustness
                  sl_raw = min(sl_raw, entry_ob['bottom'] - market_info['price_precision_step_decimal']) # Place slightly below OB bottom
                  # Ensure SL is below entry price
                  if sl_raw >= last_close:
                      lg.warning(f"Calculated LONG SL {sl_raw} >= entry {last_close}. Adjusting SL below entry.")
                      sl_raw = last_close - market_info['price_precision_step_decimal']
                  initial_sl_price = _safe_decimal_conversion(_format_price(exchange, market_info['symbol'], sl_raw))

                  if initial_tp_atr_mult > Decimal('0'):
                       tp_raw = last_close + current_atr * initial_tp_atr_mult
                       # Ensure TP is above entry price
                       if tp_raw <= last_close:
                            lg.warning(f"Calculated LONG TP {tp_raw} <= entry {last_close}. Disabling TP.")
                            initial_tp_price = None
                       else:
                            initial_tp_price = _safe_decimal_conversion(_format_price(exchange, market_info['symbol'], tp_raw))

        elif potential_entry_side == 'short' and entry_ob and entry_ob_edge:
             signal_result['signal'] = 'SELL'
             signal_result['reason'] = f"Down trend (VT Placeholder) & price near Bearish OB {entry_ob['id']} @ {entry_ob_edge.normalize()}"
             # Calculate initial SL/TP based on ATR and entry price (last_close)
             if current_atr > Decimal('0'):
                  sl_raw = last_close + current_atr * initial_sl_atr_mult
                  # Ensure SL is above the OB top for robustness
                  sl_raw = max(sl_raw, entry_ob['top'] + market_info['price_precision_step_decimal']) # Place slightly above OB top
                   # Ensure SL is above entry price
                  if sl_raw <= last_close:
                      lg.warning(f"Calculated SHORT SL {sl_raw} <= entry {last_close}. Adjusting SL above entry.")
                      sl_raw = last_close + market_info['price_precision_step_decimal']
                  initial_sl_price = _safe_decimal_conversion(_format_price(exchange, market_info['symbol'], sl_raw))

                  if initial_tp_atr_mult > Decimal('0'):
                       tp_raw = last_close - current_atr * initial_tp_atr_mult
                       # Ensure TP is below entry price
                       if tp_raw >= last_close:
                            lg.warning(f"Calculated SHORT TP {tp_raw} >= entry {last_close}. Disabling TP.")
                            initial_tp_price = None
                       else:
                            initial_tp_price = _safe_decimal_conversion(_format_price(exchange, market_info['symbol'], tp_raw))

        # Final check and assignment of SL/TP prices
        signal_result['initial_sl_price'] = initial_sl_price if initial_sl_price and initial_sl_price > Decimal('0') else None
        signal_result['initial_tp_price'] = initial_tp_price if initial_tp_price and initial_tp_price > Decimal('0') else None

        if signal_result['signal'] != 'NONE':
             lg.info(f"{NEON_GREEN if signal_result['signal'] == 'BUY' else NEON_RED}Entry Signal Generated: {signal_result['signal']} ({signal_result['reason']}) @ Price: {last_close.normalize()} | SL: {initial_sl_price.normalize() if initial_sl_price else 'None'}, TP: {initial_tp_price.normalize() if initial_tp_price else 'None'}{RESET}")

        return signal_result


# --- Position Sizing Logic ---
def calculate_position_size(account_balance: Optional[Decimal], current_price: Decimal, atr_value: Decimal, market_info: MarketInfo, logger: logging.Logger, exchange: ccxt.Exchange) -> Optional[Decimal]:
    """
    Calculates the position size (amount in base currency for spot or contracts for futures)
    based on available balance, risk per trade, ATR-based SL distance, and market constraints.

    Uses the Risk % of Capital approach: Position Size = (Capital * Risk %) / (Loss per Unit at SL)
    Assumes SL is placed based on ATR * initial_stop_loss_atr_multiple.
    Handles Linear/Spot contracts accurately. Provides warning for Inverse contracts due to complexity.

    Args:
        account_balance: The available balance in the quote currency (Decimal). Can be None.
        current_price: The current market price (Decimal).
        atr_value: The current ATR value for the symbol (Decimal).
        market_info: The MarketInfo TypedDict for the symbol.
        logger: The logger instance for the symbol.
        exchange: CCXT exchange instance (needed for formatting).

    Returns:
        The calculated position size as a Decimal, formatted to exchange precision,
        or None if calculation is not possible or results in an invalid size.
    """
    lg = logger

    # Retrieve configuration parameters safely
    risk_per_trade = _parse_decimal(CONFIG.get("risk_per_trade"), 'risk_per_trade', lg, default=Decimal(DEFAULT_CONFIG['risk_per_trade']))
    initial_sl_atr_mult = _parse_decimal(CONFIG.get("protection", {}).get("initial_stop_loss_atr_multiple"), 'initial_sl_atr_mult', lg, default=Decimal(DEFAULT_INITIAL_STOP_LOSS_ATR_MULTIPLE))
    leverage_config = _parse_decimal(CONFIG.get("leverage"), 'leverage', lg, default=Decimal(DEFAULT_CONFIG['leverage'])) # Desired leverage (used for info, not directly in risk calc here)

    # --- Input Validation ---
    if account_balance is None or not account_balance.is_finite() or account_balance <= Decimal('0'):
        lg.warning("Cannot calculate position size: Account balance is zero, invalid, or None.")
        return None
    if not current_price.is_finite() or current_price <= Decimal('0'):
        lg.warning(f"Cannot calculate position size: Current price {current_price} is zero or invalid.")
        return None
    if not atr_value.is_finite() or atr_value <= Decimal('0'):
        lg.warning(f"Cannot calculate position size: ATR value {atr_value} is zero or invalid.")
        return None
    if not risk_per_trade.is_finite() or risk_per_trade <= Decimal('0') or risk_per_trade > Decimal('1'):
         lg.error(f"Cannot calculate position size: Invalid risk_per_trade config value {risk_per_trade}. Must be > 0 and <= 1.")
         return None
    if not initial_sl_atr_mult.is_finite() or initial_sl_atr_mult <= Decimal('0'):
         lg.error(f"Cannot calculate position size: Invalid initial_stop_loss_atr_multiple config value {initial_sl_atr_mult}. Must be > 0.")
         return None
    if market_info['amount_precision_step_decimal'] is None or market_info['amount_precision_step_decimal'] <= Decimal('0'):
        lg.error(f"Cannot calculate position size: Market amount precision step is missing or invalid for {market_info['symbol']}.")
        return None
    if market_info['price_precision_step_decimal'] is None or market_info['price_precision_step_decimal'] <= Decimal('0'):
        lg.error(f"Cannot calculate position size: Market price precision step is missing or invalid for {market_info['symbol']}.")
        return None

    # --- Calculation ---
    try:
        # Calculate Stop Loss distance in quote currency price units
        sl_distance_price_units = atr_value * initial_sl_atr_mult
        if sl_distance_price_units <= Decimal('0'):
             lg.warning(f"Calculated stop loss price distance ({sl_distance_price_units}) is zero or negative. Cannot size position.")
             return None

        # Calculate the total capital to risk in quote currency
        risked_capital_quote = account_balance * risk_per_trade

        # Calculate position size based on risk and SL distance
        # This formula is generally correct for Linear contracts (Quote=USDT, ContractSize=1) and Spot markets.
        # It calculates how many base units (or contracts if ContractSize=1) can be bought
        # such that if the price moves by `sl_distance_price_units`, the loss equals `risked_capital_quote`.
        # Loss per unit = sl_distance_price_units * ContractSize (if linear/spot, ContractSize=1)
        # Quantity = risked_capital_quote / Loss per unit
        calculated_amount = risked_capital_quote / sl_distance_price_units

        # Handle Inverse Contracts (Warning)
        if market_info['is_inverse']:
            # TODO: Implement accurate sizing for inverse contracts. This is complex as PnL is in the base currency.
            # The simple formula above is likely incorrect for inverse.
            # Common workarounds: Size based on fixed USD amount, or use a more complex formula involving ContractSize and Price.
            lg.warning(f"{NEON_YELLOW}Position sizing formula used is primarily for Spot/Linear contracts. Sizing for Inverse contract '{market_info['symbol']}' (Contract Size: {market_info['contract_size_decimal'].normalize()} {market_info['settle']}) may be inaccurate using ATR distance in quote. Verify calculation or implement specific inverse sizing logic.{RESET}")
            # For now, proceed with the calculated amount but acknowledge the potential inaccuracy.

        # --- Apply Precision and Limits ---
        # Format the calculated amount to the exchange's required precision step
        formatted_amount_str = _format_amount(exchange, market_info['symbol'], calculated_amount)
        if formatted_amount_str is None:
            lg.warning(f"Calculated amount {calculated_amount.normalize()} cannot be formatted to precision for {market_info['symbol']}. Sizing failed.")
            return None

        final_amount = _parse_decimal(formatted_amount_str, 'formatted_amount', lg, allow_none=False, default=Decimal('0.0'))
        if final_amount <= POSITION_QTY_EPSILON:
             lg.warning(f"Calculated amount {calculated_amount.normalize()} resulted in zero or negligible amount {final_amount.normalize()} after formatting. Cannot place order.")
             return None

        # Check against market limits (Amount)
        min_amount = market_info['min_amount_decimal']
        max_amount = market_info['max_amount_decimal']

        if min_amount is not None and final_amount < min_amount:
            lg.warning(f"Calculated amount {final_amount.normalize()} is below minimum amount {min_amount.normalize()}. Adjusting to minimum.")
            final_amount = min_amount
            # Re-format after adjustment
            formatted_min_amount_str = _format_amount(exchange, market_info['symbol'], final_amount)
            if formatted_min_amount_str is None:
                 lg.error(f"Cannot format minimum amount {min_amount.normalize()} for {market_info['symbol']}. Sizing failed.")
                 return None
            final_amount = _parse_decimal(formatted_min_amount_str, 'formatted_min_amount', lg, default=Decimal('0.0'))
            if final_amount <= POSITION_QTY_EPSILON: return None # Still invalid after adjusting

        if max_amount is not None and final_amount > max_amount:
            lg.warning(f"Calculated amount {final_amount.normalize()} is above maximum amount {max_amount.normalize()}. Adjusting to maximum.")
            final_amount = max_amount
            # Re-format after adjustment
            formatted_max_amount_str = _format_amount(exchange, market_info['symbol'], final_amount)
            if formatted_max_amount_str is None:
                 lg.error(f"Cannot format maximum amount {max_amount.normalize()} for {market_info['symbol']}. Sizing failed.")
                 return None
            final_amount = _parse_decimal(formatted_max_amount_str, 'formatted_max_amount', lg, default=Decimal('0.0'))
            if final_amount <= POSITION_QTY_EPSILON: return None # Still invalid after adjusting

        # Check against market limits (Cost/Value)
        # Cost = Price * Amount * ContractSize (for linear/spot, ContractSize is 1 usually)
        # For Inverse, Cost calculation is different (Amount is contracts, value depends on ContractSize in Quote)
        min_cost = market_info['min_cost_decimal']
        max_cost = market_info['max_cost_decimal']
        contract_size = market_info['contract_size_decimal']

        current_cost: Optional[Decimal] = None
        if market_info['is_linear'] or market_info['spot']:
            current_cost = current_price * final_amount * contract_size # contract_size is likely 1
        elif market_info['is_inverse']:
            # Value of inverse position = Amount (contracts) * ContractSize (in Quote)
            current_cost = final_amount * contract_size
        else:
            # Fallback or unknown type - estimate cost like linear/spot
            lg.warning(f"Unknown market type '{market_info['contract_type_str']}'. Estimating cost like Linear/Spot.")
            current_cost = current_price * final_amount * contract_size

        if current_cost is not None:
             lg.debug(f"Estimated order cost: {current_cost.normalize()} {market_info['quote']}")
             # Check Min Cost
             if min_cost is not None and current_cost < min_cost:
                  lg.warning(f"Estimated order cost {current_cost.normalize()} is below minimum cost {min_cost.normalize()}. Attempting to adjust amount.")
                  # Calculate minimum amount needed to meet min_cost
                  min_amount_from_cost: Optional[Decimal] = None
                  if market_info['is_linear'] or market_info['spot']:
                      if current_price > Decimal('0') and contract_size > Decimal('0'):
                           min_amount_from_cost = min_cost / (current_price * contract_size)
                  elif market_info['is_inverse']:
                       if contract_size > Decimal('0'):
                            min_amount_from_cost = min_cost / contract_size

                  if min_amount_from_cost is not None:
                      # Take the larger of the originally calculated amount or the amount needed for min cost
                      final_amount = max(final_amount, min_amount_from_cost)
                      # Re-format and re-check amount limits
                      formatted_adj_amount_str = _format_amount(exchange, market_info['symbol'], final_amount)
                      if formatted_adj_amount_str is None:
                           lg.error(f"Cannot format amount adjusted for min cost ({final_amount.normalize()}). Sizing failed.")
                           return None
                      final_amount = _parse_decimal(formatted_adj_amount_str, 'formatted_adj_amount', lg, default=Decimal('0.0'))
                      if final_amount <= POSITION_QTY_EPSILON: return None
                      # Re-check max amount limit
                      if max_amount is not None and final_amount > max_amount:
                           lg.warning(f"Amount adjusted for min cost ({final_amount.normalize()}) exceeds max amount ({max_amount.normalize()}). Clamping to max amount.")
                           final_amount = max_amount
                           # Format final clamped amount
                           formatted_adj_amount_str = _format_amount(exchange, market_info['symbol'], final_amount)
                           if formatted_adj_amount_str is None: return None
                           final_amount = _parse_decimal(formatted_adj_amount_str, 'formatted_adj_amount', lg, default=Decimal('0.0'))
                           if final_amount <= POSITION_QTY_EPSILON: return None
                  else:
                       lg.error("Cannot calculate minimum amount from min cost due to zero price or contract size. Sizing failed.")
                       return None

             # Check Max Cost (apply limit if necessary)
             if max_cost is not None and current_cost > max_cost:
                  lg.warning(f"Estimated order cost {current_cost.normalize()} is above maximum cost {max_cost.normalize()}. Attempting to reduce amount.")
                  # Calculate maximum allowed amount based on max_cost
                  max_amount_from_cost: Optional[Decimal] = None
                  if market_info['is_linear'] or market_info['spot']:
                       if current_price > Decimal('0') and contract_size > Decimal('0'):
                            max_amount_from_cost = max_cost / (current_price * contract_size)
                  elif market_info['is_inverse']:
                       if contract_size > Decimal('0'):
                            max_amount_from_cost = max_cost / contract_size

                  if max_amount_from_cost is not None:
                       # Take the smaller of the current amount or the max allowed by cost
                       final_amount = min(final_amount, max_amount_from_cost)
                       # Re-format and re-check amount limits
                       formatted_adj_amount_str = _format_amount(exchange, market_info['symbol'], final_amount)
                       if formatted_adj_amount_str is None:
                            lg.error(f"Cannot format amount adjusted for max cost ({final_amount.normalize()}). Sizing failed.")
                            return None
                       final_amount = _parse_decimal(formatted_adj_amount_str, 'formatted_adj_amount', lg, default=Decimal('0.0'))
                       if final_amount <= POSITION_QTY_EPSILON: return None
                       # Re-check min amount limit
                       if min_amount is not None and final_amount < min_amount:
                            lg.warning(f"Amount adjusted for max cost ({final_amount.normalize()}) fell below min amount ({min_amount.normalize()}). Cannot meet both constraints. Sizing failed.")
                            return None
                  else:
                       lg.error("Cannot calculate maximum amount from max cost due to zero price or contract size. Sizing failed.")
                       return None

        # Final check on the final calculated amount
        if final_amount is None or final_amount <= POSITION_QTY_EPSILON:
             lg.warning(f"Final calculated amount {final_amount} after limit/precision adjustments is zero or invalid. Cannot place order.")
             return None

        lg.info(f"Calculated position size: {final_amount.normalize()} {market_info['base']} ({market_info['contract_type_str']})")
        return final_amount

    except InvalidOperation as e:
        lg.error(f"{NEON_RED}Decimal calculation error during position sizing: {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during position sizing: {e}{RESET}", exc_info=True)
        return None


# --- Trading Execution Functions ---

def execute_trade(exchange: ccxt.Exchange, symbol: str, side: str, amount: Decimal, price: Decimal, initial_sl_price: Optional[Decimal], initial_tp_price: Optional[Decimal], market_info: MarketInfo, logger: logging.Logger) -> Optional[Dict]:
    """
    Executes a market order with optional initial Stop Loss and Take Profit.
    Uses exchange-specific parameters for Bybit V5 SL/TP if needed.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        side: 'buy' or 'sell'.
        amount: The order amount as a Decimal, already formatted to precision.
        price: The current price (used for log/context, market orders don't specify price).
        initial_sl_price: The initial Stop Loss price as a Decimal, formatted to precision, or None.
        initial_tp_price: The initial Take Profit price as a Decimal, formatted to precision, or None.
        market_info: The MarketInfo TypedDict for the symbol.
        logger: The logger instance for the symbol.

    Returns:
        The CCXT order dictionary if successful, or None if the order fails after retries.
    """
    lg = logger

    # --- Input Validation ---
    if amount is None or amount <= POSITION_QTY_EPSILON:
         lg.error(f"Attempted to execute trade with invalid amount: {amount}. Aborting.")
         return None
    if side not in ['buy', 'sell']:
         lg.error(f"Invalid side '{side}' for trade execution. Aborting.")
         return None
    if market_info['symbol'] != symbol: # Basic check
         lg.error(f"Mismatched market info for {symbol} passed to execute_trade. Aborting.")
         return None

    # --- Prepare Order Parameters ---
    # Format amount and prices again just before sending (should already be formatted)
    formatted_amount_str = _format_amount(exchange, symbol, amount)
    if formatted_amount_str is None:
         lg.error(f"Failed to format order amount {amount.normalize()} for {symbol}. Aborting trade.")
         return None
    final_amount_dec = _parse_decimal(formatted_amount_str, 'final_amount', lg, default=Decimal('0.0'))
    if final_amount_dec <= POSITION_QTY_EPSILON:
        lg.error(f"Final amount {final_amount_dec} is negligible after formatting. Aborting trade.")
        return None

    # CCXT expects amount as float in create_order
    amount_float = float(final_amount_dec)

    # Prepare params dictionary for SL/TP
    order_params: Dict[str, Any] = {}

    # Add SL/TP parameters using Bybit V5 specific keys within 'params' if unified doesn't work reliably
    # CCXT's Bybit implementation *should* map top-level 'stopLoss'/'takeProfit' to the correct V5 params.
    # We will use the top-level CCXT parameters first.
    if initial_sl_price is not None:
        formatted_sl_price_str = _format_price(exchange, symbol, initial_sl_price)
        if formatted_sl_price_str:
            # CCXT expects price as float for stopLoss/takeProfit params
            order_params['stopLoss'] = float(initial_sl_price)
            lg.info(f"Adding initial SL {formatted_sl_price_str} to order parameters.")
        else:
            lg.warning(f"Failed to format initial SL price {initial_sl_price}. SL will not be set.")

    if initial_tp_price is not None:
        formatted_tp_price_str = _format_price(exchange, symbol, initial_tp_price)
        if formatted_tp_price_str:
            # CCXT expects price as float
            order_params['takeProfit'] = float(initial_tp_price)
            lg.info(f"Adding initial TP {formatted_tp_price_str} to order parameters.")
        else:
            lg.warning(f"Failed to format initial TP price {initial_tp_price}. TP will not be set.")

    # Bybit V5 specific: Trigger price type (optional, defaults usually okay)
    # order_params['params'] = {'slTriggerBy': 'MarkPrice', 'tpTriggerBy': 'MarkPrice'} # Example

    # Determine order type (Market)
    order_type = 'market'
    action_desc = f"creating {side.upper()} {order_type} order"

    # --- Execute Order with Retries ---
    placed_order = None
    for attempt in range(MAX_API_RETRIES + 1):
        if _shutdown_requested:
             lg.info(f"Shutdown requested during order placement for {symbol}. Aborting.")
             return None
        try:
            lg.info(f"Attempting to place {side.upper()} {order_type} order for {formatted_amount_str} {market_info['base']} on {symbol} @ ~{price.normalize()}, attempt {attempt + 1}/{MAX_API_RETRIES + 1}...")
            if 'stopLoss' in order_params: lg.info(f"  Initial SL: {order_params['stopLoss']}")
            if 'takeProfit' in order_params: lg.info(f"  Initial TP: {order_params['takeProfit']}")

            # Send the order to the exchange
            order = exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount_float,  # Pass formatted amount as float
                price=None,           # Market order, price is not specified
                params=order_params   # Pass SL/TP parameters here
            )

            # Validate response (basic check for ID)
            if order and isinstance(order, dict) and order.get('id'):
                order_id = order.get('id')
                order_status = order.get('status', 'unknown')
                lg.info(f"{NEON_GREEN}Successfully initiated {side.upper()} order {order_id} for {formatted_amount_str} on {symbol}. Status: {order_status}{RESET}")
                # Wait a moment for the exchange to process the order and potentially open the position
                lg.debug(f"Waiting {POSITION_CONFIRM_DELAY_SECONDS}s for exchange processing...")
                time.sleep(POSITION_CONFIRM_DELAY_SECONDS)
                placed_order = order
                break # Exit retry loop on success
            else:
                lg.warning(f"Order placement for {symbol} returned invalid response or no ID: {order}. Attempt {attempt + 1}.")
                # Treat as failure and potentially retry
                placed_order = None

        except ccxt.InsufficientFunds as e:
             # Special handling for insufficient funds - not retryable for this trade
             _handle_ccxt_exception(e, lg, action_desc, symbol, attempt) # Log and potentially notify
             return None # Fatal for this trade attempt
        except ccxt.InvalidOrder as e:
             # Special handling for invalid order parameters - not retryable
             _handle_ccxt_exception(e, lg, action_desc, symbol, attempt) # Log and potentially notify
             return None # Fatal for this trade attempt
        except Exception as e:
            # Use the helper for other CCXT/network errors
            retry = _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
            if not retry:
                 # If helper returns False, it's a fatal error or max retries reached
                 return None
            # If retry is True, the helper has already waited, continue the loop

    if placed_order is None:
        lg.error(f"{NEON_RED}Failed to place order for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
        # Send notification for failed order placement
        send_notification(f"Order Failed: {symbol}", f"Failed to place {side.upper()} order for {symbol} after retries.", lg)
        return None

    lg.debug(f"Trade execution attempt complete for {symbol}.")
    return placed_order


def close_position(exchange: ccxt.Exchange, symbol: str, position: PositionInfo, market_info: MarketInfo, logger: logging.Logger) -> bool:
    """
    Closes an existing open position using a market order.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        position: The PositionInfo TypedDict of the position to close.
        market_info: The MarketInfo TypedDict for the symbol.
        logger: The logger instance for the symbol.

    Returns:
        True if the close order is successfully placed (does not guarantee execution),
        False if placing the order fails after retries.
    """
    lg = logger

    # --- Input Validation ---
    if position is None or position['size_decimal'] <= POSITION_QTY_EPSILON or position['side'] not in ['long', 'short']:
         lg.error(f"Attempted to close position for {symbol} with invalid position data: {position}")
         return False

    pos_side = position['side']
    pos_amount = position['size_decimal']
    # Closing a position requires trading the opposite side
    close_side = 'sell' if pos_side == 'long' else 'buy'

    # --- Prepare Close Order ---
    # Format the position size for the close order amount
    formatted_amount_str = _format_amount(exchange, symbol, pos_amount)
    if formatted_amount_str is None:
         lg.error(f"Failed to format position amount {pos_amount.normalize()} for closing order on {symbol}. Aborting close.")
         return False
    final_amount_dec = _parse_decimal(formatted_amount_str, 'close_amount', lg, default=Decimal('0.0'))
    if final_amount_dec <= POSITION_QTY_EPSILON:
        lg.error(f"Formatted close amount {final_amount_dec} is negligible for {symbol}. Aborting close.")
        return False

    # CCXT expects float for amount
    amount_float = float(final_amount_dec)

    # For Bybit V5 Unified Margin, simply placing an opposing market order for the full size closes the position.
    # Some exchanges might require specific 'reduceOnly' parameters. CCXT usually handles this.
    close_params = {'reduceOnly': True} # Use reduceOnly if supported and appropriate for the exchange/account type
    # Check if Bybit V5 needs/supports reduceOnly explicitly in create_order params
    # Bybit V5 API docs suggest 'reduceOnly' flag is available for limit/market orders.
    # Let's include it for safety, CCXT might ignore if not applicable.

    action_desc = f"closing {pos_side.upper()} position"

    # --- Execute Close Order with Retries ---
    close_order_placed = False
    for attempt in range(MAX_API_RETRIES + 1):
        if _shutdown_requested:
             lg.info(f"Shutdown requested during position close for {symbol}. Aborting.")
             return False
        try:
            lg.info(f"Attempting to place {close_side.upper()} MARKET order to close {formatted_amount_str} {pos_side.upper()} position on {symbol}, attempt {attempt + 1}/{MAX_API_RETRIES + 1}...")

            # Place the market order to close the position
            order = exchange.create_order(
                symbol=symbol,
                type='market',
                side=close_side,
                amount=amount_float, # Pass formatted amount as float
                params=close_params  # Include reduceOnly flag
            )

            # Validate response
            if order and isinstance(order, dict) and order.get('id'):
                order_id = order.get('id')
                order_status = order.get('status', 'unknown')
                lg.info(f"{NEON_GREEN}Successfully placed closing order {order_id} for {symbol}. Status: {order_status}{RESET}")
                # Wait a moment for the exchange to process the close
                lg.debug(f"Waiting {POSITION_CONFIRM_DELAY_SECONDS}s for position closure confirmation...")
                time.sleep(POSITION_CONFIRM_DELAY_SECONDS)
                close_order_placed = True
                break # Exit retry loop on success
            else:
                 lg.warning(f"Closing order placement for {symbol} returned invalid response or no ID: {order}. Attempt {attempt + 1}.")
                 close_order_placed = False # Ensure False for retry logic

        except ccxt.InsufficientFunds as e:
             # This shouldn't happen when closing a position with sufficient margin, but handle defensively
             _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
             send_notification(f"Funds Error (Close): {symbol}", f"Failed to close position due to unexpected funds issue on {symbol}. Details: {e}", lg, notification_type='email')
             return False # Fatal for this close attempt
        except ccxt.InvalidOrder as e:
             # Could happen if reduceOnly causes issues or amount is slightly off due to fees/slippage
             _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
             send_notification(f"Invalid Order (Close): {symbol}", f"Closing order parameters invalid for {symbol}. Details: {e}.", lg, notification_type='email')
             return False # Fatal for this close attempt
        except Exception as e:
            retry = _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
            if not retry:
                 # If helper returns False, it's a fatal error or max retries reached
                 return False
            # If retry is True, the helper has already waited, continue the loop

    if not close_order_placed:
        lg.error(f"{NEON_RED}Failed to place closing order for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
        send_notification(f"Close Order Failed: {symbol}", f"Failed to place closing order for {symbol} after retries.", lg)
        return False

    lg.debug(f"Position close attempt complete for {symbol}.")
    return True


def edit_position_protection(exchange: ccxt.Exchange, symbol: str, position: PositionInfo, market_info: MarketInfo, new_sl_price: Optional[Decimal] = None, new_tp_price: Optional[Decimal] = None, new_tsl_activation_price: Optional[Decimal] = None, logger: logging.Logger) -> bool:
    """
    Modifies the Stop Loss, Take Profit, and/or Trailing Stop Loss for an existing position.
    Uses Bybit V5 specific parameters via CCXT `edit_position`'s `params` dictionary.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        position: The PositionInfo TypedDict of the position to modify.
        market_info: The MarketInfo TypedDict for the symbol.
        new_sl_price: The new SL price (Decimal) to set, or None to not change. Use 0 to remove.
        new_tp_price: The new TP price (Decimal) to set, or None to not change. Use 0 to remove.
        new_tsl_activation_price: The price (Decimal) at which TSL becomes active, or None to not change TSL activation.
                                   Setting this enables TSL with the callback rate from config. Use 0 to remove TSL.

    Returns:
        True if the modification request was successful, False otherwise.
    """
    lg = logger

    # --- Input Validation ---
    if position is None or position['size_decimal'] <= POSITION_QTY_EPSILON or position['side'] not in ['long', 'short']:
         lg.error(f"Attempted to modify protection for invalid position on {symbol}.")
         return False
    if market_info['symbol'] != symbol:
         lg.error(f"Mismatched market info for {symbol} passed to edit_position_protection.")
         return False

    pos_side = position['side']
    current_sl_dec = position['stopLossPrice_dec']
    current_tp_dec = position['takeProfitPrice_dec']
    current_tsl_activation_dec = position['tslActivationPrice_dec']
    price_tick_size = market_info['price_precision_step_decimal']

    update_params: Dict[str, Any] = {}
    needs_update = False
    action_desc = f"modifying protection for {pos_side.upper()} position on {symbol}"

    # --- Prepare Stop Loss ---
    if new_sl_price is not None:
        # Format SL price (allow 0 for removal)
        if new_sl_price == Decimal('0'):
            formatted_sl_price_str = "0" # Use string "0" to remove SL/TP on Bybit V5
            if current_sl_dec is not None: # Only update if SL was previously set
                 update_params['stopLoss'] = formatted_sl_price_str
                 needs_update = True
                 lg.info(f"Removing SL for {symbol}.")
        else:
            formatted_sl_price_str = _format_price(exchange, symbol, new_sl_price)
            if formatted_sl_price_str is None:
                 lg.warning(f"Failed to format new SL price {new_sl_price.normalize()} for {symbol}. Cannot update SL.")
            else:
                 parsed_new_sl = _parse_decimal(formatted_sl_price_str, 'new_sl', lg)
                 # Check if new SL price is actually different from current SL price
                 if current_sl_dec is None or abs(parsed_new_sl - current_sl_dec) > price_tick_size / Decimal('2'):
                     update_params['stopLoss'] = formatted_sl_price_str # Pass formatted string
                     needs_update = True
                     lg.info(f"Setting new SL for {symbol}: {formatted_sl_price_str}")
                 # else: lg.debug(f"New SL price same as current. Skipping.")

    # --- Prepare Take Profit ---
    if new_tp_price is not None:
        # Format TP price (allow 0 for removal)
        if new_tp_price == Decimal('0'):
            formatted_tp_price_str = "0"
            if current_tp_dec is not None: # Only update if TP was previously set
                update_params['takeProfit'] = formatted_tp_price_str
                needs_update = True
                lg.info(f"Removing TP for {symbol}.")
        else:
            formatted_tp_price_str = _format_price(exchange, symbol, new_tp_price)
            if formatted_tp_price_str is None:
                 lg.warning(f"Failed to format new TP price {new_tp_price.normalize()} for {symbol}. Cannot update TP.")
            else:
                 parsed_new_tp = _parse_decimal(formatted_tp_price_str, 'new_tp', lg)
                 # Check if new TP price is actually different from current TP price
                 if current_tp_dec is None or abs(parsed_new_tp - current_tp_dec) > price_tick_size / Decimal('2'):
                     update_params['takeProfit'] = formatted_tp_price_str # Pass formatted string
                     needs_update = True
                     lg.info(f"Setting new TP for {symbol}: {formatted_tp_price_str}")
                 # else: lg.debug(f"New TP price same as current. Skipping.")

    # --- Prepare Trailing Stop Loss ---
    # Bybit V5 requires setting `activePrice` and `trailingStop` (callback distance/price/percentage string)
    # CCXT `edit_position` might map differently. We use `params` for Bybit V5 specifics.
    if new_tsl_activation_price is not None:
        if new_tsl_activation_price == Decimal('0'):
            # Request to remove TSL
            if current_tsl_activation_dec is not None: # Only update if TSL was active
                 update_params['trailingStop'] = "0" # Set trigger to 0
                 update_params['activePrice'] = "0"  # Set activation to 0
                 needs_update = True
                 lg.info(f"Removing TSL for {symbol}.")
        else:
            # Request to set/update TSL activation
            formatted_tsl_activation_str = _format_price(exchange, symbol, new_tsl_activation_price)
            if formatted_tsl_activation_str is None:
                lg.warning(f"Failed to format new TSL activation price {new_tsl_activation_price.normalize()} for {symbol}. Cannot update TSL.")
            else:
                parsed_new_tsl_act = _parse_decimal(formatted_tsl_activation_str, 'new_tsl_act', lg)
                # Check if new activation price is different from current
                if current_tsl_activation_dec is None or abs(parsed_new_tsl_act - current_tsl_activation_dec) > price_tick_size / Decimal('2'):
                     # Retrieve callback rate from config
                     config_tsl_callback_rate = _parse_decimal(CONFIG.get("protection", {}).get("trailing_stop_callback_rate"), 'tsl_callback_rate', lg, default=Decimal(DEFAULT_TRAILING_STOP_CALLBACK_RATE))
                     if config_tsl_callback_rate <= Decimal('0'):
                          lg.error(f"Invalid TSL callback rate in config: {config_tsl_callback_rate}. Cannot activate TSL.")
                     else:
                          # Bybit V5 API `setTradingStop` uses `trailingStop` for the callback rate/distance string.
                          # Format as percentage string (e.g., "0.5%")
                          formatted_tsl_callback_rate_str = f"{config_tsl_callback_rate * Decimal('100'):.2f}%" # e.g., "0.50%"

                          update_params['activePrice'] = formatted_tsl_activation_str # Activation price
                          update_params['trailingStop'] = formatted_tsl_callback_rate_str # Callback % string

                          # Optional: Set trigger type
                          # update_params['trailingTriggerBy'] = 'MarkPrice'

                          needs_update = True
                          lg.info(f"Setting/Updating TSL for {symbol}: Activation Price={formatted_tsl_activation_str}, Callback Rate={formatted_tsl_callback_rate_str}")
                     # else: lg.debug(f"New TSL activation price same as current. Skipping TSL update.")


    # --- Execute API Call ---
    if not needs_update:
         lg.debug(f"No protection parameters changed for {symbol}. No update needed.")
         return True # Nothing to do, considered successful

    # Call the exchange API to modify the position protection
    modified_successfully = False
    # Add required parameters for Bybit V5 setTradingStop (mapped by CCXT edit_position)
    # These might be automatically included by CCXT based on the position object, but explicit is safer if needed.
    bybit_params = {
        'symbol': symbol,
        'side': pos_side, # 'Buy' for long, 'Sell' for short
        **update_params # Add SL/TP/TSL params
    }
    # Bybit V5 API uses 'Buy'/'Sell' for side in setTradingStop, map from 'long'/'short'
    bybit_params['side'] = 'Buy' if pos_side == 'long' else 'Sell'

    for attempt in range(MAX_API_RETRIES + 1):
        if _shutdown_requested:
             lg.info(f"Shutdown requested during protection modification for {symbol}. Aborting.")
             return False
        try:
            lg.debug(f"Attempting to {action_desc}, attempt {attempt + 1}/{MAX_API_RETRIES + 1} with params: {bybit_params}")

            # Use `set_trading_stop` via implicit API call if CCXT `edit_position` doesn't map well
            # response = exchange.private_post_v5_position_set_trading_stop(bybit_params) # Example implicit call

            # Try using unified `edit_position` first, passing Bybit params inside 'params'
            # Note: CCXT's `edit_position` might not exist or work for all exchanges/parameters.
            # Let's assume we need the implicit call for Bybit V5 `setTradingStop`.
            # Check CCXT version and Bybit implementation details.
            # As of recent CCXT versions, modifying SL/TP/TSL might require specific methods or implicit calls.
            # Let's stick to the implicit call pattern for Bybit V5 setTradingStop for reliability.

            response = exchange.private_post_v5_position_set_trading_stop(bybit_params)

            # Check Bybit V5 response code
            if response and isinstance(response, dict) and response.get('retCode') == 0:
                 lg.info(f"{NEON_GREEN}Successfully modified protection for {symbol} position.{RESET}")
                 modified_successfully = True
                 break # Exit retry loop on success
            else:
                 ret_code = response.get('retCode') if isinstance(response, dict) else 'N/A'
                 ret_msg = response.get('retMsg') if isinstance(response, dict) else 'N/A'
                 lg.warning(f"Protection modification failed for {symbol}. Code: {ret_code}, Msg: {ret_msg}. Response: {response}. Attempt {attempt + 1}.")
                 modified_successfully = False # Ensure False for retry logic
                 # Check for specific error codes that shouldn't be retried (e.g., invalid parameters)
                 # Bybit V5 Error Codes: https://bybit-exchange.github.io/docs/v5/error
                 if ret_code in [110007, 110015, 110043, 110044, 110045, 110046, 110047, 110048, 110055, 110057]: # Parameter errors, SL/TP invalid errors
                      lg.error(f"{NEON_RED}Protection modification failed due to non-retryable parameter error (Code: {ret_code}). Aborting update.{RESET}")
                      return False # Don't retry parameter errors

        except Exception as e:
             retry = _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
             if not retry:
                  return False # Exit if helper says not to retry
             # If retry is True, helper waited, loop continues

    if not modified_successfully:
         lg.error(f"{NEON_RED}Failed to modify protection for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
         # Send notification for persistent failure
         send_notification(f"Protect Update Failed: {symbol}", f"Failed to update SL/TP/TSL for {symbol} position after retries.", lg)
         return False

    lg.debug(f"Protection modification attempt complete for {symbol}.")
    return True


# --- Position Management (BE, TSL) Logic ---

# Dictionary to store bot-managed state per symbol (e.g., position BE/TSL status)
# In a real-world bot, this state should be persistent (database, file).
# For this example, it's in-memory state tied to the symbol processing loop.
SYMBOL_STATE: Dict[str, Dict[str, Any]] = {} # { 'symbol': { 'be_activated': False, 'tsl_activated': False, 'position_id': None, ... } }

def get_symbol_state(symbol: str) -> Dict[str, Any]:
    """Retrieves or initializes the bot's in-memory state for a specific symbol."""
    if symbol not in SYMBOL_STATE:
        SYMBOL_STATE[symbol] = {
            'be_activated': False,
            'tsl_activated': False,
            'position_id': None, # Track the current position ID to ensure state matches
            # Add other state variables here if needed
        }
        # Use init_logger here as symbol logger might not exist yet
        init_logger.debug(f"Initialized state for {symbol}: {SYMBOL_STATE[symbol]}")
    return SYMBOL_STATE[symbol]

def reset_symbol_state(symbol: str, logger: logging.Logger):
    """Resets the bot's in-memory state for a specific symbol."""
    if symbol in SYMBOL_STATE:
        logger.debug(f"Resetting state for {symbol}.")
        del SYMBOL_STATE[symbol] # Remove entry to force re-initialization next time
    # Ensure it's initialized if get_symbol_state is called immediately after
    get_symbol_state(symbol)


def check_and_manage_position(exchange: ccxt.Exchange, symbol: str, position: PositionInfo, market_info: MarketInfo, analysis_results: StrategyAnalysisResults, logger: logging.Logger) -> bool:
    """
    Manages an existing open position, including Break-Even and Trailing Stop Loss logic.
    This function is called when `generate_signal` returns 'HOLD'.
    Updates the in-memory SYMBOL_STATE flags based on successful API calls.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol.
        position: The PositionInfo TypedDict of the current open position (with bot state flags potentially updated).
        market_info: MarketInfo TypedDict for the symbol.
        analysis_results: The results from analyze_strategy (contains current price, ATR).
        logger: The logger instance for the symbol.

    Returns:
        True if the position is still considered open and managed, False if an exit
        was attempted or the position was somehow invalidated.
    """
    lg = logger

    # --- Input Validation ---
    if position is None or position['size_decimal'] <= POSITION_QTY_EPSILON or position['entryPrice_decimal'] is None or position['side'] not in ['long', 'short']:
         lg.warning(f"Attempted to manage invalid position for {symbol}.")
         # Reset state if position seems invalid? Risky, could be temporary fetch issue.
         # reset_symbol_state(symbol, lg)
         return False # Position cannot be managed

    pos_id = position['id']
    pos_size = position['size_decimal']
    pos_side = position['side']
    entry_price = position['entryPrice_decimal']
    current_price = analysis_results['last_close'] # Use the most recent close from klines
    current_atr = analysis_results['atr'] # Use the most recent ATR from klines

    # Get symbol state (ensures it's initialized if needed)
    symbol_state = get_symbol_state(symbol)
    # Ensure state matches the position being managed
    if symbol_state['position_id'] != pos_id:
         lg.warning(f"State mismatch! Managing position {pos_id} but state tracks {symbol_state['position_id']}. Resetting state for {symbol}.")
         reset_symbol_state(symbol, lg)
         symbol_state['position_id'] = pos_id # Start tracking this position
         # Note: This means BE/TSL flags start fresh for this newly tracked position

    be_activated = symbol_state['be_activated']
    tsl_activated = symbol_state['tsl_activated']

    lg.debug(f"Managing {pos_side.upper()} position {pos_id} ({pos_size.normalize()}) on {symbol}. Entry: {entry_price.normalize()}, Current: {current_price.normalize()}. State: BE={be_activated}, TSL={tsl_activated}")

    # Check if current price and ATR are valid for calculations
    if not isinstance(current_price, Decimal) or not current_price.is_finite() or current_price <= Decimal('0') or \
       not isinstance(current_atr, Decimal) or not current_atr.is_finite() or current_atr <= Decimal('0'):
        lg.warning("Current price or ATR is invalid during position management. Skipping BE/TSL checks.")
        return True # Still considered open, but cannot manage stops this cycle

    # --- Break-Even (BE) Logic ---
    enable_be = CONFIG.get("protection", {}).get("enable_break_even", False)
    be_trigger_atr_mult = _parse_decimal(CONFIG.get("protection", {}).get("break_even_trigger_atr_multiple"), 'be_trigger_atr_mult', lg, default=Decimal(DEFAULT_BREAK_EVEN_TRIGGER_ATR_MULTIPLE))
    be_offset_ticks = CONFIG.get("protection", {}).get("break_even_offset_ticks", DEFAULT_BREAK_EVEN_OFFSET_TICKS)
    price_tick_size = market_info['price_precision_step_decimal'] # Guaranteed non-None

    if enable_be and not be_activated:
        # Check if price has moved enough in profit to trigger BE
        profit_target_distance = current_atr * be_trigger_atr_mult
        profit_trigger_price = entry_price + profit_target_distance if pos_side == 'long' else entry_price - profit_target_distance

        lg.debug(f"BE Check: Side={pos_side}, Current={current_price}, Trigger={profit_trigger_price}")
        if (pos_side == 'long' and current_price >= profit_trigger_price) or \
           (pos_side == 'short' and current_price <= profit_trigger_price):

            lg.info(f"{NEON_GREEN}Break-Even trigger price {profit_trigger_price.normalize()} reached/crossed for {pos_side.upper()} on {symbol}. Activating BE.{RESET}")

            # Calculate the BE price (entry price +/- offset ticks)
            offset_value = price_tick_size * be_offset_ticks
            be_price_raw = entry_price + offset_value if pos_side == 'long' else entry_price - offset_value

            # Format BE price to exchange precision
            formatted_be_price_str = _format_price(exchange, symbol, be_price_raw)
            if formatted_be_price_str is None:
                 lg.error(f"{NEON_RED}Failed to calculate or format BE price {be_price_raw} for {symbol}. Cannot set BE.{RESET}")
            else:
                 be_price = _parse_decimal(formatted_be_price_str, 'be_price', lg)
                 if be_price is None or be_price <= Decimal('0'):
                      lg.error(f"{NEON_RED}Formatted BE price {formatted_be_price_str} is invalid. Cannot set BE.{RESET}")
                 else:
                      # Ensure BE price direction is correct relative to entry (offset should handle this)
                      if (pos_side == 'long' and be_price < entry_price) or \
                         (pos_side == 'short' and be_price > entry_price):
                          lg.warning(f"{NEON_YELLOW}Calculated BE price {be_price.normalize()} is worse than entry {entry_price.normalize()} for {pos_side} position. Adjusting BE to entry price.{RESET}")
                          # Adjust to entry price exactly if offset pushed it the wrong way
                          formatted_be_price_str = _format_price(exchange, symbol, entry_price)
                          if formatted_be_price_str is None:
                              lg.error("Failed to format entry price for adjusted BE. Cannot set BE.")
                              be_price = None # Flag as failed
                          else:
                              be_price = _parse_decimal(formatted_be_price_str, 'be_price_adjusted', lg)

                      if be_price is not None:
                           # Request to update the Stop Loss to the calculated BE price
                           lg.info(f"Attempting to move SL to Break-Even @ {be_price.normalize()} for {symbol}.")
                           # Pass the Decimal price to the helper
                           if edit_position_protection(exchange, symbol, position, market_info, new_sl_price=be_price, logger=lg):
                               lg.info(f"{NEON_GREEN}Successfully requested SL update to BE for {symbol}.{RESET}")
                               symbol_state['be_activated'] = True # Mark BE as activated in state
                           else:
                               lg.error(f"{NEON_RED}Failed to move SL to Break-Even for {symbol}. Will retry next cycle.{RESET}")
                               # Leave be_activated as False

    # --- Trailing Stop Loss (TSL) Logic ---
    enable_tsl = CONFIG.get("protection", {}).get("enable_trailing_stop", False)
    tsl_activation_percentage = _parse_decimal(CONFIG.get("protection", {}).get("trailing_stop_activation_percentage"), 'tsl_activation_percentage', lg, default=Decimal(DEFAULT_TRAILING_STOP_ACTIVATION_PERCENTAGE))
    # tsl_callback_rate is retrieved in edit_position_protection from config

    if enable_tsl and not tsl_activated:
        # Check if price has moved enough in profit (% from entry) to activate TSL
        activation_price_calculated = entry_price * (Decimal('1') + tsl_activation_percentage) if pos_side == 'long' else \
                                      entry_price * (Decimal('1') - tsl_activation_percentage)

        lg.debug(f"TSL Check: Side={pos_side}, Current={current_price}, Activation Trigger={activation_price_calculated}")
        activation_condition_met = False
        if (pos_side == 'long' and current_price >= activation_price_calculated) or \
           (pos_side == 'short' and current_price <= activation_price_calculated):
             activation_condition_met = True

        if activation_condition_met:
            lg.info(f"{NEON_GREEN}TSL activation price {activation_price_calculated.normalize()} reached/crossed for {pos_side.upper()} on {symbol}. Activating TSL.{RESET}")

            # Request to update TSL parameters via API
            # We pass the calculated activation price. The edit_position_protection function handles formatting and API call.
            lg.info(f"Attempting to activate TSL for {symbol} with activation price {activation_price_calculated.normalize()}.")
            if edit_position_protection(exchange, symbol, position, market_info, new_tsl_activation_price=activation_price_calculated, logger=lg):
                lg.info(f"{NEON_GREEN}Successfully requested TSL activation for {symbol}.{RESET}")
                symbol_state['tsl_activated'] = True # Mark TSL as activated in state
            else:
                lg.error(f"{NEON_RED}Failed to activate TSL for {symbol}. Will retry next cycle.{RESET}")
                # Leave tsl_activated as False


    return True # Position is still considered open and under management

# --- Main Bot Logic per Symbol ---

def handle_trading_pair(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger, market_info: MarketInfo):
    """
    Main logic loop for a single trading pair. Fetches data, runs strategy,
    checks position, generates signals, and executes trades or manages position.

    Args:
        exchange: The configured CCXT exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        logger: The logger instance specific to this symbol.
        market_info: The MarketInfo TypedDict for this symbol.
    """
    lg = logger
    lg.info(f"{NEON_BLUE}--- Processing {symbol} ---{RESET}")

    # --- Retrieve Configuration ---
    # Use .get with defaults for safety, although load_config should ensure keys exist
    interval = CONFIG.get("interval", default_config['interval'])
    fetch_limit = CONFIG.get("fetch_limit", default_config['fetch_limit'])
    min_klines_for_strategy = CONFIG.get("min_klines_for_strategy", default_config['min_klines_for_strategy'])
    enable_trading = CONFIG.get("enable_trading", False)
    max_concurrent_positions = CONFIG.get("max_concurrent_positions", default_config['max_concurrent_positions'])
    strategy_params = CONFIG.get("strategy_params", default_config['strategy_params'])

    # --- Fetch Data ---
    df = fetch_klines_ccxt(exchange, symbol, interval, fetch_limit, lg)
    if df is None: # fetch_klines returns None on critical failure
        lg.error(f"Failed to fetch kline data for {symbol}. Skipping cycle.")
        return
    if df.empty: # fetch_klines returns empty DF if no data or processing failed but fetch succeeded
         lg.warning(f"Fetched empty or invalid kline data for {symbol}. Skipping strategy analysis.")
         return

    # Check if enough klines for strategy calculation
    if len(df) < min_klines_for_strategy:
        lg.warning(f"Not enough klines ({len(df)}) for strategy ({min_klines_for_strategy} required). Skipping cycle for {symbol}.")
        return

    # --- Calculate Indicators ---
    # This modifies the DataFrame in place (or returns a modified copy)
    df_with_indicators = calculate_indicators(df, strategy_params, lg)
    if df_with_indicators.empty:
         lg.error(f"Indicator calculation failed for {symbol}. Skipping cycle.")
         return

    # --- Analyze Strategy ---
    analysis_results = analyze_strategy(df_with_indicators, strategy_params, lg)
    # Check if analysis was successful enough to proceed (e.g., ATR and price are valid)
    if not isinstance(analysis_results['atr'], Decimal) or not analysis_results['atr'].is_finite() or \
       not isinstance(analysis_results['last_close'], Decimal) or not analysis_results['last_close'].is_finite():
        lg.warning(f"Strategy analysis incomplete (ATR or Last Close missing/invalid) for {symbol}. Skipping signal generation.")
        return

    # --- Fetch Balance & Position ---
    account_balance: Optional[Decimal] = None
    if enable_trading: # Fetch balance only if needed for trading actions
        account_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if account_balance is None:
            lg.warning(f"Failed to fetch account balance in {QUOTE_CURRENCY}. Cannot size new trades or potentially manage existing ones if balance check fails.")
            # Continue processing to potentially manage existing positions based on signals, but block new entries.

    # Fetch current open positions for this symbol
    current_open_positions = fetch_open_positions(exchange, symbol, lg)

    # --- Manage Bot State (Link Fetched Position to State) ---
    symbol_state = get_symbol_state(symbol)
    bot_managed_position: Optional[PositionInfo] = None

    # Check if the position tracked in state still exists
    tracked_pos_id = symbol_state.get('position_id')
    position_closed_externally = False
    if tracked_pos_id is not None and not any(p['id'] == tracked_pos_id for p in current_open_positions):
         lg.info(f"Position {tracked_pos_id} previously tracked for {symbol} is now closed. Resetting state.")
         reset_symbol_state(symbol, lg) # Reset state for this symbol
         position_closed_externally = True

    # Find the position currently managed by the bot state (if any)
    if not position_closed_externally and tracked_pos_id is not None:
        for pos in current_open_positions:
             if pos['id'] == tracked_pos_id:
                  # This is the position the bot is tracking. Update its state flags.
                  pos['be_activated'] = symbol_state['be_activated']
                  pos['tsl_activated'] = symbol_state['tsl_activated']
                  bot_managed_position = pos
                  lg.debug(f"Found tracked position {tracked_pos_id} for {symbol}. State: BE={pos['be_activated']}, TSL={pos['tsl_activated']}")
                  break

    # Handle cases where a position exists but isn't tracked (e.g., after restart, manual trade)
    # If config allows only 1 position, and one exists but isn't tracked, start tracking it.
    if bot_managed_position is None and current_open_positions and max_concurrent_positions == 1:
         lg.warning(f"{NEON_YELLOW}Found an open position for {symbol} (ID: {current_open_positions[0]['id']}) not tracked by current bot state. Assuming it's the target position due to max_concurrent_positions=1.{RESET}")
         bot_managed_position = current_open_positions[0]
         # Initialize state for this newly found position
         symbol_state['position_id'] = bot_managed_position['id']
         symbol_state['be_activated'] = False # Assume state unknown
         symbol_state['tsl_
