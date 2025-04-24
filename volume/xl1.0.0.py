```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pyrmethus_volumatic_bot.py
# Unified and Enhanced Trading Bot incorporating Volumatic Trend, Pivot Order Blocks,
# and advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Merged and improved from multiple source files.
# Version 1.5.1: Enhanced comments, docstrings, error handling, Decimal usage, config validation,
#                native SL/TP/TSL handling for Bybit V5, margin checks, and notifications.

"""
Pyrmethus Volumatic Bot: A Python Trading Bot for Bybit V5

This bot implements a trading strategy based on the combination of:
1.  **Volumatic Trend:** An EMA/SWMA crossover system with ATR-based bands,
    incorporating normalized volume analysis. (Specific EMA/SWMA logic is placeholder).
2.  **Pivot Order Blocks (OBs):** Identifying potential support/resistance zones
    based on pivot highs and lows derived from candle wicks or bodies.

This version synthesizes features and robustness from previous iterations, including:
-   Robust configuration loading from both .env (secrets) and config.json (parameters).
-   Detailed configuration validation with automatic correction to defaults and saving.
-   Flexible notification options (Termux SMS and Email) with error handling.
-   Enhanced logging with colorama, rotation, sensitive data redaction, and timezone support.
-   Comprehensive API interaction handling with retries, exponential backoff, and error logging for CCXT.
-   Accurate Decimal usage for all financial calculations via dedicated helper functions.
-   Structured data types using TypedDicts for clarity and type safety.
-   Implementation of native Bybit V5 Stop Loss, Take Profit, and Trailing Stop Loss.
-   Logic for managing Break-Even stop adjustments via native SL updates.
-   Pre-order checks for minimum order size/value and available margin.
-   Support for multiple trading pairs (processed sequentially per cycle).
-   Graceful shutdown on interruption (Ctrl+C) or critical errors.

Disclaimer:
- **EXTREME RISK**: Trading cryptocurrencies, especially futures contracts with leverage and automated systems, involves substantial risk of financial loss. This script is provided for EDUCATIONAL PURPOSES ONLY. You could lose your entire investment and potentially more. Use this software entirely at your own risk. The authors and contributors assume NO responsibility for any trading losses.
- **NATIVE SL/TP/TSL RELIANCE**: The bot's protective stop mechanisms rely entirely on Bybit's exchange-native order execution. Their performance is subject to exchange conditions, potential slippage during volatile periods, API reliability, order book liquidity, and specific exchange rules. These orders are NOT GUARANTEED to execute at the precise trigger price specified. Market volatility can cause significant deviations.
- **PARAMETER SENSITIVITY & OPTIMIZATION**: The performance of this bot is HIGHLY dependent on the chosen strategy parameters (indicator settings, risk levels, SL/TP/TSL percentages, filter thresholds). These parameters require EXTENSIVE backtesting, optimization, and forward testing on a TESTNET environment before considering ANY live deployment. Default parameters are unlikely to be profitable and serve only as examples.
- **API RATE LIMITS & BANS**: Excessive API requests can lead to temporary or permanent bans from the exchange. Monitor API usage and adjust script timing (loop_delay_seconds) accordingly. CCXT's built-in rate limiter is enabled but may not prevent all issues under heavy load or rapid market conditions.
- **SLIPPAGE**: Market orders, used for entry and potentially for SL/TP/TSL execution by the exchange, are susceptible to slippage. This means the actual execution price may differ significantly from the price observed when the order was placed, especially during high volatility or low liquidity.
- **TEST THOROUGHLY**: **DO NOT RUN THIS SCRIPT WITH REAL FUNDS WITHOUT EXTENSIVE AND SUCCESSFUL TESTING ON A TESTNET OR DEMO ACCOUNT.** Ensure you fully understand every part of the code, its logic, its limitations, and its potential risks before any live deployment. Verify calculations and exchange interactions carefully.
- **ONE-WAY MODE ASSUMPTION**: This bot is designed primarily for Bybit's **One-Way Mode** for perpetual contracts. Hedge Mode is not explicitly supported and may lead to unexpected behavior.
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
import traceback # For detailed error logging
import subprocess # Used for Termux SMS
import shutil     # Used to check for termux-sms-send command
import smtplib    # Used for Email notifications
from email.mime.text import MIMEText # Used for Email notifications
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation, DivisionByZero
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

# --- Timezone Handling ---
# Attempt to import the standard library's zoneinfo (Python 3.9+)
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    # Ensure tzdata is installed for non-UTC timezones with zoneinfo
    try:
        # Attempt to load a common non-UTC zone to check if tzdata is available
        # We don't need this specific zone, just need the check to pass
        _ = ZoneInfo("America/New_York") # Example timezone
    except ZoneInfoNotFoundError:
        # Handle the case where zoneinfo is available but tzdata is not found
        print(f"\033[93mWarning: 'zoneinfo' is available, but 'tzdata' package seems missing or corrupt.\033[0m") # Yellow text
        print(f"\033[93m         `pip install tzdata` is recommended for reliable timezone support.\033[0m")
        # Continue with zoneinfo, but it might fail for non-UTC zones requested by the user
    except Exception as tz_init_err:
         # Catch any other unexpected errors during ZoneInfo initialization
         print(f"\033[93mWarning: Error initializing test timezone with 'zoneinfo': {tz_init_err}\033[0m")
         # Continue cautiously
except ImportError:
    # Fallback for older Python versions or if zoneinfo itself is not installed
    print(f"\033[93mWarning: 'zoneinfo' module not found (requires Python 3.9+). Falling back to basic UTC implementation.\033[0m")
    print(f"\033[93m         For accurate local time logging, upgrade Python or use a backport library.\033[0m")

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
            # If already timezone-aware, convert to UTC.
            return dt.astimezone(timezone.utc)

        def fromutc(self, dt: datetime) -> datetime:
            """Converts a UTC datetime to this timezone (which is always UTC in the fallback)."""
            if not isinstance(dt, datetime):
                raise TypeError("fromutc() requires a datetime argument")
            if dt.tzinfo is None:
                 raise ValueError("fromutc: naive datetime has no timezone to observe")
            # Ensure it's converted to UTC
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
import pandas_ta as ta  # Technical Analysis library (requires pandas)

# API & Networking
import requests         # For HTTP requests (often used implicitly by ccxt)
import ccxt             # Crypto Exchange Trading Library

# Utilities
from colorama import Fore, Style, init as colorama_init # Colored console output
from dotenv import load_dotenv                        # Load environment variables from .env file

# --- Initial Setup ---
# Set Decimal precision globally for accurate financial calculations
# 28 digits is usually sufficient, but verify based on asset precision needs.
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
    print(f"{Fore.YELLOW}Warning: Invalid SMTP_PORT '{SMTP_PORT_STR}' in .env file. Using default: {SMTP_PORT}.{Style.RESET_ALL}")
SMTP_USER: Optional[str] = os.getenv("SMTP_USER")
SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")
NOTIFICATION_EMAIL_RECIPIENT: Optional[str] = os.getenv("NOTIFICATION_EMAIL_RECIPIENT") # Renamed for clarity
TERMUX_SMS_RECIPIENT: Optional[str] = os.getenv("TERMUX_SMS_RECIPIENT") # Renamed for clarity

# --- Constants ---
BOT_VERSION = "1.5.1"

# --- Configuration File & Logging ---
CONFIG_FILE: str = "config.json"    # Name of the configuration file
LOG_DIRECTORY: str = "bot_logs"     # Directory to store log files

# --- Timezone Configuration ---
DEFAULT_TIMEZONE_STR: str = "UTC" # Default timezone if not specified elsewhere (Safer to default to UTC)
# Prioritize TIMEZONE from .env, fallback to the default defined above.
TIMEZONE_STR: str = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR).strip()
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
MAX_API_RETRIES: int = 3           # Max number of retries for most failed API calls (excluding rate limits)
RETRY_DELAY_SECONDS: int = 5       # Initial delay in seconds between retries (often increased exponentially)
POSITION_CONFIRM_DELAY_SECONDS: int = 8 # Delay after placing order to confirm position status (allows exchange processing time)
LOOP_DELAY_SECONDS: int = 15       # Base delay in seconds between main loop cycles (per symbol)
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
# This is a common constraint on Bybit V5 for USDT perpetuals. Check exchange docs for specifics.
MIN_ORDER_VALUE_USDT: Decimal = Decimal("1.0")

# Threshold for considering a position size or price difference as zero/negligible
POSITION_QTY_EPSILON: Decimal = Decimal("1e-9")

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
_termux_sms_command_exists: Optional[bool] = None # Cache the result of checking termux-sms-send existence
CONFIG: Dict[str, Any] = {}       # Global config dictionary, loaded by load_config()
DEFAULT_CONFIG: Dict[str, Any] = {} # Default config structure, set by load_config()

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
    vol_norm_int: Optional[int] # Normalized volume indicator value (e.g., 0-100), if used by strategy
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
    maker: float            # Maker fee rate (as a fraction, e.0.0002 for 0.02%)
    contractSize: Optional[Any] # Size of one contract (often 1 for linear, value in USD for inverse). Use decimal version for calcs.
    expiry: Optional[int]   # Timestamp (milliseconds UTC) of future/option expiry
    expiryDatetime: Optional[str] # ISO 8601 datetime string of expiry
    strike: Optional[float] # Strike price for options
    optionType: Optional[str] # 'call' or 'put' for options
    precision: Dict[str, Any] # Price and amount precision rules (e.g., {'price': 0.01, 'amount': 0.001}). Use decimal versions for calcs.
    limits: Dict[str, Any]    # Order size and cost limits (e.g., {'amount': {'min': 0.001, 'max': 100}}). Use decimal versions for calcs.
    info: Dict[str, Any]      # Raw market data dictionary directly from the exchange API response (useful for debugging or accessing non-standard fields like 'category')
    # --- Added/Derived Fields for Convenience and Precision ---
    is_contract: bool         # Enhanced convenience flag: True if swap, future, or option based on ccxt flags
    is_linear: bool           # Enhanced convenience flag: True if linear contract (and is_contract is True)
    is_inverse: bool          # Enhanced convenience flag: True if inverse contract (and is_contract is True)
    contract_type_str: str    # User-friendly string describing the market type: "Spot", "Linear", "Inverse", "Option", or "Unknown"
    min_amount_decimal: Optional[Decimal] # Minimum order size (in base currency for spot, contracts for futures) as Decimal, or None if not specified
    max_amount_decimal: Optional[Decimal] # Maximum order size (in base/contracts) as Decimal, or None
    min_cost_decimal: Optional[Decimal]   # Minimum order cost (value in quote currency, usually price * amount * contractSize) as Decimal, or None
    max_cost_decimal: Optional[Decimal]   # Maximum order cost (value in quote currency) as Decimal, or None
    min_price_decimal: Optional[Decimal]  # Minimum order price as Decimal, or None
    max_price_decimal: Optional[Decimal]  # Maximum order price as Decimal, or None
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
    side: Optional[str]      # Position side: 'long' or 'short' (or 'none' if derived as flat)
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
    info: Dict[str, Any]         # Raw position data dictionary directly from the exchange API response (essential for accessing non-standard fields like protection orders and positionIdx)
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
    # --- Protection Order Status (Extracted from 'info' or root, strings often returned by Exchange API) ---
    # Note: These are the RAW string values returned by the API, need parsing to Decimal.
    stopLossPrice_raw: Optional[str] # Current stop loss price set on the exchange (as string, often '0' or '0.0' if not set)
    takeProfitPrice_raw: Optional[str] # Current take profit price set on the exchange (as string, often '0' or '0.0' if not set)
    # Bybit V5 'trailingStop' is the trigger price once active, not the distance/callback rate.
    trailingStopPrice_raw: Optional[str] # Current trailing stop trigger price (as string, e.g., Bybit V5 'trailingStop')
    tslActivationPrice_raw: Optional[str] # Trailing stop activation price (as string, if available/set, e.g., Bybit V5 'activePrice')
    # --- Parsed Protection Order Status (Decimal) ---
    stopLossPrice_dec: Optional[Decimal] # Parsed SL price as Decimal (None if not set or invalid)
    takeProfitPrice_dec: Optional[Decimal] # Parsed TP price as Decimal (None if not set or invalid)
    trailingStopPrice_dec: Optional[Decimal] # Parsed TSL trigger price as Decimal (None if not set or invalid)
    tslActivationPrice_dec: Optional[Decimal] # Parsed TSL activation price as Decimal (None if not set or invalid)
    # --- Bot State Tracking (Managed internally by the bot logic, reflects *bot's* knowledge/actions on this position instance) ---
    be_activated: bool           # Has the break-even logic been triggered and successfully executed *by the bot* for this position instance?
    tsl_activated: bool          # Has the trailing stop loss been activated *by the bot* (based on price check) or detected as active on the exchange?

class SignalResult(TypedDict):
    """Represents the outcome of the signal generation process based on strategy analysis."""
    signal: str              # The generated signal: "BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT"
    reason: str              # A human-readable explanation for why the signal was generated
    initial_sl_price: Optional[Decimal] # Calculated initial stop loss price if the signal is for a new entry ("BUY" or "SELL")
    initial_tp_price: Optional[Decimal] # Calculated initial take profit price if the signal is for a new entry ("BUY" or "SELL")

# --- Logging Setup ---
class SensitiveFormatter(logging.Formatter):
    """
    Custom log formatter that redacts sensitive API keys and secrets from log messages.
    Inherits from `logging.Formatter` and overrides the `format` method.
    """
    # Use highly distinct placeholders unlikely to appear naturally in logs.
    _api_key_placeholder = "***API_KEY_REDACTED***"
    _api_secret_placeholder = "***API_SECRET_REDACTED***"

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
        name: The name for the logger (e.g., 'init', 'main', 'BTC_USDT').
              Used in log messages and to generate the log filename.

    Returns:
        A configured logging.Logger instance ready for use.
    """
    # Sanitize the logger name for safe use in filenames and hierarchical logging
    # Replace /, :, and spaces with underscores.
    safe_filename_part = re.sub(r'[^\w\-.]', '_', name)
    # Use dot notation for logger names to support potential hierarchical logging features.
    logger_name = f"pyrmethus.{safe_filename_part}"
    # Construct the full path for the log file.
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")

    logger = logging.getLogger(logger_name)

    # Prevent adding handlers multiple times if the logger was already configured
    if logger.hasHandlers():
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
            print(f"{NEON_YELLOW}Warning: Invalid CONSOLE_LOG_LEVEL '{console_log_level_str}'. Defaulting to INFO.{RESET}") # Use print as logger not fully set up
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
         # Check if both handlers were likely added successfully
         handler_names = [h.__class__.__name__ for h in logger.handlers]
         file_handler_added = any('RotatingFileHandler' in name for name in handler_names)
         console_handler_added = any('StreamHandler' in name for name in handler_names)
         logger.debug(f"Logger '{logger_name}' initialized. File Handler Level: {fh.level if file_handler_added else 'N/A'}, Console Handler Level: {sh.level if console_handler_added else 'N/A'}")
    else:
         print(f"{NEON_RED}{BRIGHT}Warning: Logger '{logger_name}' initialized but no handlers were successfully added.{RESET}")

    return logger

# --- Initial Logger Setup ---
# Create a logger instance specifically for the initialization phase.
init_logger = setup_logger("init")
init_logger.info(f"{Fore.MAGENTA}{BRIGHT}===== Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing ====={Style.RESET_ALL}")
init_logger.info(f"Using Timezone for Console Logs: {TIMEZONE_STR} ({TIMEZONE})")
init_logger.debug(f"Decimal Precision Set To: {getcontext().prec}")
# Remind user about dependencies, helpful for troubleshooting setup issues.
init_logger.debug("Ensure required packages are installed: pandas, pandas_ta, numpy, ccxt, requests, python-dotenv, colorama, tzdata (recommended)")

# --- Notification Setup ---
def send_notification(subject: str, body: str, logger: logging.Logger, notification_type: Optional[str] = None) -> bool:
    """
    Sends a notification via email or Termux SMS if configured and enabled.

    Args:
        subject (str): The subject line for the notification (used for email and prefixed to SMS).
        body (str): The main content of the notification.
        logger (logging.Logger): Logger instance to use for logging notification status/errors.
        notification_type (Optional[str]): 'email' or 'sms'. If None, uses the type from config.

    Returns:
        bool: True if the notification was attempted successfully (not necessarily received), False otherwise.
    """
    lg = logger # Alias for convenience

    # Check master notification enable/disable from config
    # Ensure CONFIG has been loaded
    if not CONFIG or not CONFIG.get("notifications", {}).get("enable_notifications", False):
        lg.debug(f"Notifications are disabled by config. Skipping '{subject}'.")
        return False

    # Determine notification type to use
    type_to_use = notification_type if notification_type else CONFIG.get('notifications', {}).get('notification_type', 'email')
    type_to_use = type_to_use.lower() # Normalize to lowercase

    if type_to_use == "email":
        # Check if email settings are complete from .env
        if not all([SMTP_SERVER, SMTP_PORT > 0, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_EMAIL_RECIPIENT]):
            lg.warning("Email notification is enabled but settings are incomplete (SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_EMAIL_RECIPIENT env vars). Cannot send email.")
            return False

        try:
            msg = MIMEText(body, 'plain', 'utf-8') # Specify encoding
            msg['Subject'] = f"[Pyrmethus Bot] {subject}" # Prefix subject for clarity
            msg['From'] = SMTP_USER
            msg['To'] = NOTIFICATION_EMAIL_RECIPIENT

            lg.debug(f"Attempting to send email notification to {NOTIFICATION_EMAIL_RECIPIENT} via {SMTP_SERVER}:{SMTP_PORT}...")
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server: # Added timeout
                server.ehlo() # Identify client to server
                server.starttls() # Upgrade connection to secure
                server.ehlo() # Re-identify after TLS
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, NOTIFICATION_EMAIL_RECIPIENT, msg.as_string())
            lg.info(f"{NEON_GREEN}Successfully sent email notification: '{subject}'{RESET}")
            return True
        except smtplib.SMTPAuthenticationError:
            lg.error(f"{NEON_RED}Failed to send email notification: SMTP Authentication Error. Check username/password.{RESET}")
            return False
        except smtplib.SMTPServerDisconnected:
             lg.error(f"{NEON_RED}Failed to send email notification: SMTP Server Disconnected unexpectedly.{RESET}")
             return False
        except smtplib.SMTPException as e:
            lg.error(f"{NEON_RED}Failed to send email notification (SMTP Error): {e}{RESET}")
            return False
        except Exception as e:
            lg.error(f"{NEON_RED}Failed to send email notification (Unexpected Error): {e}{RESET}", exc_info=True)
            return False

    elif type_to_use == "sms":
         # Check if Termux SMS is configured and command is available
         global _termux_sms_command_exists
         if TERMUX_SMS_RECIPIENT is None:
             lg.warning("SMS notification is enabled but TERMUX_SMS_RECIPIENT environment variable is not set.")
             return False

         # Check for command existence only once per script run
         if _termux_sms_command_exists is None:
             termux_command_path = shutil.which('termux-sms-send')
             _termux_sms_command_exists = termux_command_path is not None
             if not _termux_sms_command_exists:
                  lg.warning(f"{NEON_YELLOW}SMS notification enabled, but 'termux-sms-send' command not found in PATH. Ensure Termux:API is installed (`pkg install termux-api`).{RESET}")
             else:
                  lg.debug(f"Found 'termux-sms-send' command at: {termux_command_path}")

         if not _termux_sms_command_exists:
             return False # Don't proceed if command is missing

         # Prepare the command. Message should be the last argument(s).
         # Prefix message for clarity. Limit message length if needed (SMS limits vary).
         sms_message = f"[Pyrmethus] {subject}: {body}"[:150] # Example limit
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
                 timeout=sms_timeout,
                 encoding='utf-8'    # Specify encoding
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
        lg.error(f"{NEON_RED}Invalid notification_type specified: '{type_to_use}'. Must be 'email' or 'sms'.{RESET}")
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
    updated_config = config.copy()
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            # Key is missing, add it with the default value
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config: Added missing key '{full_key_path}' with default value: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # If both default and loaded values are dicts, recurse into nested dict
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                # If nested dict was changed, update the parent dict and mark as changed
                updated_config[key] = nested_config
                changed = True
        # Optional: Could add type mismatch check here, but validation below handles it more robustly.
    return updated_config, changed

# Global flag used within validation helpers to track if saving is needed
config_needs_saving: bool = False

def _validate_and_correct_numeric(cfg: Dict, key_path: str, min_val: Union[int, float, Decimal], max_val: Union[int, float, Decimal],
                             is_strict_min: bool = False, is_int: bool = False, allow_zero: bool = False) -> bool:
    """
    Validates a numeric config value at `key_path` (e.g., "protection.leverage").

    Checks type (int/float/str numeric), range [min_val, max_val] or (min_val, max_val] if strict.
    Uses the default value from `DEFAULT_CONFIG` and logs a warning/info if correction needed.
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
    global DEFAULT_CONFIG # Access the global default config structure
    global config_needs_saving # Allow modification of the global flag

    keys = key_path.split('.')
    current_level = cfg
    default_level = DEFAULT_CONFIG
    try:
        # Traverse nested dictionaries to reach the target key and its default value
        for key in keys[:-1]:
            current_level = current_level[key]
            default_level = default_level[key]
        leaf_key = keys[-1]
        original_val = current_level.get(leaf_key)
        default_val = default_level.get(leaf_key) # Get default for fallback on error
    except (KeyError, TypeError):
        init_logger.error(f"{NEON_RED}Config validation error: Invalid path '{key_path}'. Cannot validate.{RESET}")
        return False # Path itself is wrong, cannot proceed

    if original_val is None:
        init_logger.warning(f"{NEON_YELLOW}Config validation: Key missing at '{key_path}' during numeric check. Using default: {repr(default_val)}{RESET}")
        current_level[leaf_key] = default_val
        config_needs_saving = True
        return True

    corrected = False
    final_val = original_val # Start with the original value
    target_type_str = 'integer' if is_int else 'float' # For logging

    try:
        # 1. Reject Boolean Type Explicitly
        if isinstance(original_val, bool):
             raise TypeError("Boolean type is not valid for numeric configuration.")

        # 2. Attempt Conversion to Decimal for Robust Validation
        try:
            str_val = str(original_val).strip()
            if str_val == "": raise ValueError("Empty string cannot be converted to a number.")
            num_val = Decimal(str_val)
        except (InvalidOperation, TypeError, ValueError):
             raise TypeError(f"Value '{repr(original_val)}' cannot be converted to a number.")

        # 3. Check for Non-Finite Values
        if not num_val.is_finite():
            raise ValueError("Non-finite value (NaN or Infinity) is not allowed.")

        # Convert range limits to Decimal
        min_dec = Decimal(str(min_val))
        max_dec = Decimal(str(max_val))

        # 4. Range Check
        is_zero = num_val.is_zero()
        min_check_passed = (num_val > min_dec) if is_strict_min else (num_val >= min_dec)
        max_check_passed = (num_val <= max_dec)
        range_check_passed = min_check_passed and max_check_passed

        if not range_check_passed and not (allow_zero and is_zero):
            range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
            allowed_str = f"{range_str}{' or 0' if allow_zero else ''}"
            raise ValueError(f"Value {num_val.normalize()} is outside the allowed range {allowed_str}.")

        # 5. Type Check and Correction
        needs_type_correction = False
        if is_int:
            if num_val % 1 != 0:
                needs_type_correction = True
                final_val = int(num_val.to_integral_value(rounding=ROUND_DOWN)) # Truncate
                init_logger.info(f"{NEON_YELLOW}Config Update: Truncated fractional part for integer key '{key_path}' from {repr(original_val)} to {repr(final_val)}.{RESET}")
                # Re-check range after truncation
                final_dec_trunc = Decimal(final_val)
                min_check_passed_trunc = (final_dec_trunc > min_dec) if is_strict_min else (final_dec_trunc >= min_dec)
                range_check_passed_trunc = min_check_passed_trunc and (final_dec_trunc <= max_dec)
                if not range_check_passed_trunc and not (allow_zero and final_dec_trunc.is_zero()):
                    raise ValueError(f"Value truncated to {final_val}, which is outside the allowed range.")
            elif not isinstance(original_val, int):
                 needs_type_correction = True
                 final_val = int(num_val)
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type for integer key '{key_path}' from {type(original_val).__name__} to int (value: {repr(final_val)}).{RESET}")
            else:
                 final_val = int(num_val) # Ensure it's int type

        else: # Expecting float/Decimal
            # Convert validated Decimal to float for storage in JSON (common practice)
            converted_float = float(num_val)
            # Check if original wasn't float or if conversion changed value significantly
            if not isinstance(original_val, float) or abs(float(original_val) - converted_float) > 1e-9:
                 needs_type_correction = True
                 final_val = converted_float
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type/precision for float key '{key_path}' from {type(original_val).__name__} '{repr(original_val)}' to float '{repr(final_val)}'.{RESET}")
            else:
                 final_val = converted_float # Keep float representation

        if needs_type_correction:
            corrected = True

    except (ValueError, InvalidOperation, TypeError, AssertionError) as e:
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
        current_level = cfg # Re-access top level
        for key in keys[:-1]:
            current_level = current_level[key]
        current_level[leaf_key] = final_val
        config_needs_saving = True # Mark the configuration as needing to be saved

    return corrected

# Helper function to validate boolean keys
def _validate_boolean(cfg_level: Dict[str, Any], default_level: Dict[str, Any], leaf_key: str, key_path: str) -> bool:
     global config_needs_saving
     original_val = cfg_level.get(leaf_key)
     default_val = default_level.get(leaf_key)
     if not isinstance(original_val, bool):
         init_logger.warning(f"{NEON_YELLOW}Config Warning: '{key_path}' must be true or false. Provided: {repr(original_val)} (Type: {type(original_val).__name__}). Using default: {repr(default_val)}{RESET}")
         cfg_level[leaf_key] = default_val
         config_needs_saving = True
         return True
     return False # No correction needed

# Helper function to validate string choice keys
def _validate_string_choice(cfg_level: Dict[str, Any], default_level: Dict[str, Any], leaf_key: str, key_path: str, valid_choices: List[str]) -> bool:
     global config_needs_saving
     original_val = cfg_level.get(leaf_key)
     default_val = default_level.get(leaf_key)
     if not isinstance(original_val, str) or original_val not in valid_choices:
         init_logger.warning(f"{NEON_YELLOW}Config Warning: '{key_path}' ('{original_val}') is invalid. Must be one of {valid_choices}. Using default: '{default_val}'{RESET}")
         cfg_level[leaf_key] = default_val
         config_needs_saving = True
         return True
     return False # No correction needed

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
    global DEFAULT_CONFIG # Make the default config accessible globally for validation helper
    global config_needs_saving # Use the global flag

    config_needs_saving = False # Reset flag at the start of loading

    init_logger.info(f"{Fore.CYAN}# Loading configuration from '{filepath}'...{Style.RESET_ALL}")

    # --- Define Default Configuration Structure ---
    DEFAULT_CONFIG = {
        # == Trading Core ==
        "trading_pairs": ["BTC/USDT"],          # List of market symbols to trade (e.g., ["BTC/USDT", "ETH/USDT"])
        "interval": "5",                        # Kline timeframe (must be one of VALID_INTERVALS strings)
        "enable_trading": False,                # Master switch: MUST BE true for live order placement. Safety default: false.
        "use_sandbox": True,                    # Use exchange's sandbox/testnet environment? Safety default: true.
        "quote_currency": "USDT",               # Primary currency for balance, PnL, risk calculations (e.g., USDT, BUSD). Case-sensitive. Must match exchange account.
        "max_concurrent_positions": 1,          # Maximum number of positions allowed open simultaneously across all pairs (Currently per symbol due to simple loop).

        # == Risk & Sizing ==
        "risk_per_trade": 0.01,                 # Fraction of available balance to risk per trade (e.g., 0.01 = 1%). Must be > 0.0 and <= 1.0.
        "leverage": 20,                         # Desired leverage for contract trading (integer). 0 or 1 typically means spot/no leverage. Exchange limits apply.
        "required_margin_buffer": 1.05,         # Multiplier for estimated margin check (e.g., 1.05 = require 5% extra free margin)

        # == API & Timing ==
        "retry_count": MAX_API_RETRIES,                 # Max number of retries for API calls (integer >= 0)
        "retry_delay": RETRY_DELAY_SECONDS,             # Base delay in seconds between API retry attempts (integer > 0)
        "loop_delay_seconds": LOOP_DELAY_SECONDS,       # Delay in seconds between processing cycles for each symbol (integer > 0)
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Delay after placing order before checking position status (integer > 0)
        "order_fill_timeout_seconds": 15,               # Max seconds to wait for market order fill confirmation (integer > 0)
        "post_close_delay_seconds": 3,                  # Delay after closing order before re-checking position state (integer >= 0)
        # Optional finer-grained timeouts (in ms) - fallback to defaults in initialize_exchange if missing
        "api_timing": {
             "recv_window": 10000,                      # Bybit V5 recvWindow (ms)
             "fetch_ticker_timeout": 15000,             # Timeout for fetch_ticker (ms)
             "fetch_balance_timeout": 20000,            # Timeout for fetch_balance (ms)
             "create_order_timeout": 30000,             # Timeout for create_order (ms)
             "cancel_order_timeout": 20000,             # Timeout for cancel_order (ms)
             "fetch_positions_timeout": 20000,          # Timeout for fetch_positions (ms)
             "fetch_ohlcv_timeout": 60000,              # Timeout for fetch_ohlcv (ms)
             "edit_order_timeout": 25000                # Timeout for edit_order (e.g., updating SL/TP) (ms)
        },

        # == Data Fetching ==
        "fetch_limit": DEFAULT_FETCH_LIMIT,             # Default number of historical klines to fetch (integer > 0)
        "orderbook_limit": 25,                          # (Currently Unused) Limit for order book depth fetching (integer > 0)

        # == Strategy Parameters (Volumatic Trend + Order Blocks) ==
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
            "ob_entry_proximity_factor": DEFAULT_OB_ENTRY_PROXIMITY_FACTOR, # Proximity factor for entry signal near OB (float >= 1.0)
            "ob_exit_proximity_factor": DEFAULT_OB_EXIT_PROXIMITY_FACTOR,  # Proximity factor for exit signal near opposite OB (float >= 1.0)
        },

        # == Protection Parameters (Stop Loss, Take Profit, Trailing, Break Even) ==
        "protection": {
            # -- Initial SL/TP (often ATR-based) --
            "initial_stop_loss_atr_multiple": DEFAULT_INITIAL_STOP_LOSS_ATR_MULTIPLE, # Initial SL distance = ATR * this multiple (float > 0)
            "initial_take_profit_atr_multiple": DEFAULT_INITIAL_TAKE_PROFIT_ATR_MULTIPLE, # Initial TP distance = ATR * this multiple (float >= 0, 0 means no initial TP)
            # -- Break Even --
            "enable_break_even": DEFAULT_ENABLE_BREAK_EVEN,         # Enable moving SL to break-even? (boolean)
            "break_even_trigger_atr_multiple": DEFAULT_BREAK_EVEN_TRIGGER_ATR_MULTIPLE, # Move SL to BE when price moves ATR * multiple in profit (float > 0)
            "break_even_offset_ticks": DEFAULT_BREAK_EVEN_OFFSET_TICKS, # Offset SL from entry by this many price ticks for BE (integer >= 0)
            # -- Trailing Stop Loss --
            "enable_trailing_stop": DEFAULT_ENABLE_TRAILING_STOP, # Enable Trailing Stop Loss? (boolean)
            "trailing_stop_callback_rate": DEFAULT_TRAILING_STOP_CALLBACK_RATE, # TSL callback/distance (float > 0). Interpretation depends on exchange/implementation (e.g., 0.005 = 0.5%).
            "trailing_stop_activation_percentage": DEFAULT_TRAILING_STOP_ACTIVATION_PERCENTAGE, # Activate TSL when price moves this % from entry (float >= 0).
        },

        # == Notifications ==
        "notifications": {
            "enable_notifications": True, # Master switch for all notifications
            "notification_type": "email", # Default notification channel: 'email' or 'sms'
            "sms_timeout_seconds": 30     # Timeout for termux-sms-send command (integer > 0)
        },

        # == Backtesting (Placeholder) ==
        "backtesting": {
            "enabled": False,
            "start_date": "2023-01-01", # Format YYYY-MM-DD
            "end_date": "2023-12-31",   # Format YYYY-MM-DD
        }
    }

    loaded_config: Dict[str, Any] = {} # Initialize as empty dict

    # --- Step 1: File Existence Check & Creation ---
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Configuration file '{filepath}' not found. Creating a new one with default settings.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully created default config file: {filepath}{RESET}")
            loaded_config = DEFAULT_CONFIG.copy() # Use a copy of defaults
            config_needs_saving = False # No need to save again immediately
            QUOTE_CURRENCY = DEFAULT_CONFIG.get("quote_currency", "USDT")
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return DEFAULT_CONFIG.copy()

        except IOError as e:
            init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Could not create config file '{filepath}': {e}.{RESET}")
            init_logger.critical(f"{NEON_RED}Please check directory permissions. Using internal defaults as fallback.{RESET}")
            QUOTE_CURRENCY = DEFAULT_CONFIG.get("quote_currency", "USDT")
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return DEFAULT_CONFIG.copy()

    # --- Step 2: File Loading ---
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        if not isinstance(loaded_config, dict):
            raise TypeError("Configuration file content is not a valid JSON object.")
        init_logger.info(f"Successfully loaded configuration from '{filepath}'.")
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from config file '{filepath}': {e}{RESET}")
        init_logger.error(f"{NEON_RED}The file might be corrupted. Attempting to back up corrupted file and recreate with defaults.{RESET}")
        backup_path = f"{filepath}.corrupted_{int(time.time())}.bak"
        try:
            shutil.move(filepath, backup_path) # More robust than os.replace across filesystems
            init_logger.info(f"Backed up corrupted config to: {backup_path}")
        except Exception as backup_err:
             init_logger.warning(f"Could not back up corrupted config file '{filepath}': {backup_err}")
        # Attempt to recreate the default file
        try:
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(DEFAULT_CONFIG, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully recreated default config file: {filepath}{RESET}")
            QUOTE_CURRENCY = DEFAULT_CONFIG.get("quote_currency", "USDT")
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return DEFAULT_CONFIG.copy() # Return the defaults
        except IOError as e_create:
            init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Error recreating config file after corruption: {e_create}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = DEFAULT_CONFIG.get("quote_currency", "USDT")
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return DEFAULT_CONFIG.copy()
    except Exception as e:
        init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Unexpected error loading config file '{filepath}': {e}{RESET}", exc_info=True)
        init_logger.critical(f"{NEON_RED}Using internal defaults as fallback.{RESET}")
        QUOTE_CURRENCY = DEFAULT_CONFIG.get("quote_currency", "USDT")
        init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        return DEFAULT_CONFIG.copy()

    # --- Step 3: Ensure Keys and Validate Parameters ---
    try:
        # Ensure all default keys exist, add missing ones
        updated_config, keys_added = _ensure_config_keys(loaded_config, DEFAULT_CONFIG)
        if keys_added:
            config_needs_saving = True # Mark for saving later

        # --- Validation Logic ---
        init_logger.debug("Starting configuration parameter validation...")
        changes = [] # List to track if any validation corrections occurred

        # General
        changes.append(_validate_and_correct_numeric(updated_config, "max_concurrent_positions", 1, 100, is_int=True))
        changes.append(_validate_and_correct_numeric(updated_config, "risk_per_trade", Decimal('0.0001'), Decimal('1.0'), is_strict_min=True)) # Min risk > 0
        changes.append(_validate_and_correct_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True)) # Allow 0 for spot/no leverage setting
        changes.append(_validate_and_correct_numeric(updated_config, "required_margin_buffer", 1.0, 2.0)) # Buffer should be >= 1.0
        changes.append(_validate_and_correct_numeric(updated_config, "retry_count", 0, 10, is_int=True, allow_zero=True))
        changes.append(_validate_and_correct_numeric(updated_config, "retry_delay", 1, 60, is_int=True))
        changes.append(_validate_and_correct_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True))
        changes.append(_validate_and_correct_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True))
        changes.append(_validate_and_correct_numeric(updated_config, "order_fill_timeout_seconds", 5, 120, is_int=True))
        changes.append(_validate_and_correct_numeric(updated_config, "post_close_delay_seconds", 0, 30, is_int=True, allow_zero=True))
        changes.append(_validate_and_correct_numeric(updated_config, "fetch_limit", 50, BYBIT_API_KLINE_LIMIT, is_int=True))
        changes.append(_validate_and_correct_numeric(updated_config, "orderbook_limit", 1, 1000, is_int=True))

        # API Timing (nested)
        if "api_timing" in updated_config and isinstance(updated_config["api_timing"], dict):
             at = updated_config["api_timing"]
             changes.append(_validate_and_correct_numeric(at, "recv_window", 1000, 60000, is_int=True))
             changes.append(_validate_and_correct_numeric(at, "fetch_ticker_timeout", 1000, 60000, is_int=True))
             changes.append(_validate_and_correct_numeric(at, "fetch_balance_timeout", 1000, 60000, is_int=True))
             changes.append(_validate_and_correct_numeric(at, "create_order_timeout", 1000, 60000, is_int=True))
             changes.append(_validate_and_correct_numeric(at, "cancel_order_timeout", 1000, 60000, is_int=True))
             changes.append(_validate_and_correct_numeric(at, "fetch_positions_timeout", 1000, 60000, is_int=True))
             changes.append(_validate_and_correct_numeric(at, "fetch_ohlcv_timeout", 5000, 120000, is_int=True))
             changes.append(_validate_and_correct_numeric(at, "edit_order_timeout", 1000, 60000, is_int=True))


        # Trading Pairs Validation
        trading_pairs_val = updated_config.get("trading_pairs")
        if not isinstance(trading_pairs_val, list) or not trading_pairs_val:
            init_logger.warning(f"{NEON_YELLOW}Config Warning: 'trading_pairs' value '{trading_pairs_val}' is invalid. Using default: {DEFAULT_CONFIG['trading_pairs']}{RESET}")
            updated_config["trading_pairs"] = DEFAULT_CONFIG["trading_pairs"]
            config_needs_saving = True
        else:
             cleaned_pairs = [p.strip().upper() for p in trading_pairs_val if isinstance(p, str) and p.strip() and '/' in p.strip()]
             if not cleaned_pairs:
                  init_logger.warning(f"{NEON_YELLOW}Config Warning: 'trading_pairs' contains only invalid entries. Using default: {DEFAULT_CONFIG['trading_pairs']}{RESET}")
                  updated_config["trading_pairs"] = DEFAULT_CONFIG["trading_pairs"]
                  config_needs_saving = True
             elif len(cleaned_pairs) != len(trading_pairs_val):
                  init_logger.warning(f"{NEON_YELLOW}Config Warning: Some 'trading_pairs' entries were invalid/cleaned. Original: {trading_pairs_val}, Using: {cleaned_pairs}{RESET}")
                  updated_config["trading_pairs"] = cleaned_pairs
                  config_needs_saving = True
             else: # Ensure list contains the cleaned pairs
                 updated_config["trading_pairs"] = cleaned_pairs


        # Interval Validation
        changes.append(_validate_string_choice(updated_config, DEFAULT_CONFIG, "interval", "interval", VALID_INTERVALS))

        # Boolean Validations (Top Level)
        changes.append(_validate_boolean(updated_config, DEFAULT_CONFIG, "enable_trading", "enable_trading"))
        changes.append(_validate_boolean(updated_config, DEFAULT_CONFIG, "use_sandbox", "use_sandbox"))

        # Quote Currency Validation
        quote_currency_val = updated_config.get("quote_currency")
        if not isinstance(quote_currency_val, str) or not quote_currency_val.strip():
             init_logger.warning(f"{NEON_YELLOW}Config Warning: Invalid 'quote_currency' value '{quote_currency_val}'. Using default '{DEFAULT_CONFIG['quote_currency']}'.{RESET}")
             updated_config["quote_currency"] = DEFAULT_CONFIG["quote_currency"]
             config_needs_saving = True
        else:
             cleaned_quote = quote_currency_val.strip().upper()
             if cleaned_quote != updated_config["quote_currency"]:
                  init_logger.info(f"{NEON_YELLOW}Config Update: Normalized 'quote_currency' to '{cleaned_quote}'.{RESET}")
                  updated_config["quote_currency"] = cleaned_quote
                  config_needs_saving = True

        # Strategy Params (nested)
        if "strategy_params" in updated_config and isinstance(updated_config["strategy_params"], dict):
            sp = updated_config["strategy_params"]
            def_sp = DEFAULT_CONFIG["strategy_params"]
            changes.append(_validate_and_correct_numeric(sp, "vt_length", 1, 500, is_int=True))
            changes.append(_validate_and_correct_numeric(sp, "vt_atr_period", 1, MAX_DF_LEN, is_int=True))
            changes.append(_validate_and_correct_numeric(sp, "vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True))
            changes.append(_validate_and_correct_numeric(sp, "vt_atr_multiplier", 0.1, 20.0))
            changes.append(_validate_and_correct_numeric(sp, "vt_step_atr_multiplier", 0.1, 20.0)) # Unused placeholder
            changes.append(_validate_and_correct_numeric(sp, "ph_left", 1, 100, is_int=True))
            changes.append(_validate_and_correct_numeric(sp, "ph_right", 1, 100, is_int=True))
            changes.append(_validate_and_correct_numeric(sp, "pl_left", 1, 100, is_int=True))
            changes.append(_validate_and_correct_numeric(sp, "pl_right", 1, 100, is_int=True))
            changes.append(_validate_and_correct_numeric(sp, "ob_max_boxes", 1, 200, is_int=True))
            changes.append(_validate_and_correct_numeric(sp, "ob_entry_proximity_factor", 1.0, 1.1))
            changes.append(_validate_and_correct_numeric(sp, "ob_exit_proximity_factor", 1.0, 1.1))
            changes.append(_validate_string_choice(sp, def_sp, "ob_source", "strategy_params.ob_source", ["Wicks", "Body"]))
            changes.append(_validate_boolean(sp, def_sp, "ob_extend", "strategy_params.ob_extend"))

        # Protection Params (nested)
        if "protection" in updated_config and isinstance(updated_config["protection"], dict):
            pp = updated_config["protection"]
            def_pp = DEFAULT_CONFIG["protection"]
            changes.append(_validate_and_correct_numeric(pp, "initial_stop_loss_atr_multiple", Decimal('0.1'), Decimal('100.0'), is_strict_min=True))
            changes.append(_validate_and_correct_numeric(pp, "initial_take_profit_atr_multiple", Decimal('0'), Decimal('100.0'), allow_zero=True))
            changes.append(_validate_boolean(pp, def_pp, "enable_break_even", "protection.enable_break_even"))
            changes.append(_validate_and_correct_numeric(pp, "break_even_trigger_atr_multiple", Decimal('0.1'), Decimal('10.0')))
            changes.append(_validate_and_correct_numeric(pp, "break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True))
            changes.append(_validate_boolean(pp, def_pp, "enable_trailing_stop", "protection.enable_trailing_stop"))
            changes.append(_validate_and_correct_numeric(pp, "trailing_stop_callback_rate", Decimal('0.0001'), Decimal('0.5'), is_strict_min=True))
            changes.append(_validate_and_correct_numeric(pp, "trailing_stop_activation_percentage", Decimal('0'), Decimal('0.5'), allow_zero=True))

        # Notifications (nested)
        if "notifications" in updated_config and isinstance(updated_config["notifications"], dict):
            np_cfg = updated_config["notifications"]
            def_np = DEFAULT_CONFIG["notifications"]
            changes.append(_validate_boolean(np_cfg, def_np, "enable_notifications", "notifications.enable_notifications"))
            changes.append(_validate_string_choice(np_cfg, def_np, "notification_type", "notifications.notification_type", ["email", "sms"]))
            changes.append(_validate_and_correct_numeric(np_cfg, "sms_timeout_seconds", 5, 120, is_int=True))

        # Backtesting (nested)
        if "backtesting" in updated_config and isinstance(updated_config["backtesting"], dict):
            bp = updated_config["backtesting"]
            def_bp = DEFAULT_CONFIG["backtesting"]
            changes.append(_validate_boolean(bp, def_bp, "enabled", "backtesting.enabled"))
            date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            for date_key in ["start_date", "end_date"]:
                date_val = bp.get(date_key)
                default_date = def_bp[date_key]
                if not isinstance(date_val, str) or not date_pattern.match(date_val):
                     init_logger.warning(f"{NEON_YELLOW}Config Warning: 'backtesting.{date_key}' ('{date_val}') is invalid. Using default: '{default_date}'{RESET}")
                     bp[date_key] = default_date
                     config_needs_saving = True

        # --- Check if any validation corrections were made ---
        if any(changes): # checks if any True value exists in the list
            config_needs_saving = True

        # --- Save Updated Config if Necessary ---
        if config_needs_saving:
             init_logger.info("Configuration has been updated or corrected. Saving changes...")
             try:
                 # Ensure Decimal objects are converted to float/int for JSON serialization
                 def convert_decimals_for_json(obj):
                     if isinstance(obj, Decimal):
                         # Convert to int if it's a whole number, otherwise float
                         return int(obj) if obj % 1 == 0 else float(obj)
                     if isinstance(obj, dict):
                         return {k: convert_decimals_for_json(v) for k, v in obj.items()}
                     if isinstance(obj, list):
                         return [convert_decimals_for_json(elem) for elem in obj]
                     return obj

                 output_config = convert_decimals_for_json(updated_config)

                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(output_config, f_write, indent=4, ensure_ascii=False)
                 init_logger.info(f"{NEON_GREEN}Saved updated configuration to: {filepath}{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error saving updated configuration to '{filepath}': {save_err}{RESET}", exc_info=True)
                 init_logger.warning("Proceeding with corrected config in memory, but file update failed.")

        # Update the global QUOTE_CURRENCY from the validated config
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT") # Ensure it's uppercase from validation
        init_logger.info(f"Quote currency set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")

        return updated_config # Return the validated and potentially corrected config

    except Exception as e:
        # Catch-all for errors during the validation/merging process itself
        init_logger.critical(f"{NEON_RED}FATAL: Unexpected error processing configuration validation: {e}. Using internal defaults.{RESET}", exc_info=True)
        QUOTE_CURRENCY = DEFAULT_CONFIG.get("quote_currency", "USDT") # Set global from default on error
        return DEFAULT_CONFIG.copy() # Fallback to defaults on unexpected error

# --- Load Global Configuration ---
# This loads the configuration, performs validation, and updates the global QUOTE_CURRENCY and CONFIG
CONFIG = load_config(CONFIG_FILE)
# DEFAULT_CONFIG is also now globally available from within load_config scope
# QUOTE_CURRENCY is updated inside load_config()

# --- Utility Functions ---
def _safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0"), allow_none: bool = False) -> Optional[Decimal]:
    """
    Safely converts a value to a Decimal, handling None, pandas NA, common string issues, and potential errors.

    Args:
        value: The value to convert (can be string, float, int, Decimal, None, pandas NA, etc.).
        default: The Decimal value to return if conversion fails or input is None/NA (and allow_none=False).
        allow_none: If True and input value is None/NA, returns None instead of default.

    Returns:
        The converted Decimal value, the default, or None if allow_none is True and input is None/NA.
    """
    if pd.isna(value) or value is None:
        return None if allow_none else default

    try:
        # Convert to string first for consistent handling, strip whitespace
        str_value = str(value).strip()
        # Handle common non-numeric strings that might come from APIs (especially for '0' values)
        if str_value.lower() in ["", "none", "null", "0", "0.0", "0.00", "0.000", "0.0000", "0.00000"]: # Add more zero variations if needed
            # If allow_none is True and value was originally None/NA, return None
            # Otherwise, return Decimal('0') if the value represents zero
            if allow_none and (pd.isna(value) or value is None):
                return None
            else:
                return Decimal("0.0")

        # Attempt conversion to Decimal
        dec_val = Decimal(str_value)

        # Check for non-finite values (NaN, Infinity) which Decimal can represent but are often undesirable
        if not dec_val.is_finite():
             init_logger.warning(f"Non-finite Decimal value encountered during conversion: '{value}'. Returning {'None' if allow_none else default}.")
             return None if allow_none else default

        return dec_val

    except (InvalidOperation, TypeError, ValueError) as e:
        # Log a warning, but only if the value was not None/NA initially
        init_logger.warning(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {'None' if allow_none else default}. Error: {e}")
        return None if allow_none else default

def _safe_market_decimal(
    value: Any,
    field_name: str,
    allow_zero: bool = False,
    allow_negative: bool = False,
    default: Optional[Decimal] = None
) -> Optional[Decimal]:
    """
    Safely convert market data (limits, precision) to Decimal. More specific
    error logging for market data parsing. Handles non-finite values.

    Args:
        value: The value from market info (float, int, str, None).
        field_name: Name of the field being parsed (for logging).
        allow_zero: If True, 0 is considered a valid value.
        allow_negative: If True, negative values are allowed.
        default: Default value to return on failure or if value is None.

    Returns:
        The value as Decimal, or default on failure/invalid/None.
    """
    if value is None:
        return default
    try:
        # Handle potential string representations like 'inf' or scientific notation explicitly
        str_val = str(value).strip().lower()
        if str_val in ['inf', '+inf', '-inf', 'nan']:
             init_logger.warning(f"Market data parsing: Non-finite value '{value}' for {field_name}. Using default: {default}")
             return default

        # Use safe conversion helper first
        dec_val = _safe_decimal_conversion(value, default=None, allow_none=True) # Allow None from helper

        if dec_val is None: # Conversion failed or was None initially
            return default

        # Check validity based on parameters
        if not dec_val.is_finite(): # Double check, although helper should catch this
            init_logger.warning(f"Market data parsing: Converted value for {field_name} is non-finite: {value}. Using default: {default}")
            return default
        if not allow_zero and dec_val.is_zero():
            # Log as debug because '0' might be a valid default or min cost from API
            init_logger.debug(f"Market data parsing: Zero value encountered for {field_name} where not allowed: {value}. Returning default.")
            return default
        if not allow_negative and dec_val < 0:
            init_logger.warning(f"Market data parsing: Negative value not allowed for {field_name}: {value}. Using default: {default}")
            return default

        # Quantize to a reasonable precision to avoid potential floating point issues from source string/float
        # Using 15 decimal places should be safe for most crypto assets
        return dec_val.quantize(Decimal('1e-15'), rounding=ROUND_DOWN)

    except Exception as e: # Catch any unexpected error during checks
        init_logger.warning(f"Market data parsing: Unexpected error validating value for {field_name}: '{value}' ({type(value).__name__}) - Error: {e}. Using default: {default}")
        return default

def _format_price(exchange: ccxt.Exchange, market_info: MarketInfo, price: Union[Decimal, float, str]) -> Optional[str]:
    """
    Formats a price according to the market's precision rules using enhanced MarketInfo.
    Rounds DOWN to the nearest price tick size.

    Args:
        exchange: The CCXT exchange instance (used for fallback).
        market_info: Enhanced MarketInfo for the symbol.
        price: The price value (Decimal, float, or string).

    Returns:
        The price formatted as a string according to market precision (rounded down),
        or None on error or invalid input.
    """
    if price is None: return None
    # Safely convert input to Decimal first
    price_dec = _safe_decimal_conversion(price, allow_none=True)

    if price_dec is None or not price_dec.is_finite() or price_dec <= POSITION_QTY_EPSILON:
         init_logger.error(f"Format Price: Cannot format invalid, non-finite, or non-positive price: {price}")
         return None

    symbol = market_info['symbol']
    price_step = market_info.get('price_precision_step_decimal')

    if price_step is None or not price_step.is_finite() or price_step <= POSITION_QTY_EPSILON:
        # Fallback to CCXT's price_to_precision if step is invalid or missing
        init_logger.warning(f"Format Price: Invalid/missing price precision step for {symbol} ({price_step}). Falling back to CCXT's price_to_precision.")
        try:
             # CCXT formatter expects float
             return exchange.price_to_precision(symbol, float(price_dec))
        except Exception as e:
             init_logger.error(f"Format Price: CCXT fallback failed for {symbol}: {e}. Returning raw decimal string.", exc_info=True)
             return str(price_dec.normalize()) # Last resort: raw decimal string
    try:
        # Quantize the price DOWN to the market's price step (tick size)
        formatted_price_dec = (price_dec / price_step).quantize(Decimal("1"), rounding=ROUND_DOWN) * price_step
        # Check if rounding down resulted in zero or negative when input was positive
        if formatted_price_dec <= POSITION_QTY_EPSILON and price_dec > POSITION_QTY_EPSILON:
            init_logger.warning(f"Format Price: Rounding down price {price_dec.normalize()} with step {price_step.normalize()} resulted in non-positive value ({formatted_price_dec.normalize()}). Price might be below first tick.")
            # Decide behavior: return None, return original, or return the step size itself?
            # Returning None is safest to prevent potentially invalid orders.
            return None
        return str(formatted_price_dec.normalize()) # Normalize to remove trailing zeros
    except (ValueError, InvalidOperation) as e:
        init_logger.error(f"Format Price: Error quantizing price {price_dec.normalize()} for {symbol} with step {price_step.normalize()}: {e}. Returning raw decimal string.", exc_info=True)
        return str(price_dec.normalize()) # Fallback to raw decimal string

def _format_amount(exchange: ccxt.Exchange, market_info: MarketInfo, amount: Union[Decimal, float, str]) -> Optional[str]:
    """
    Formats an amount (quantity) according to the market's precision rules using enhanced MarketInfo.
    Rounds DOWN to the nearest amount step size.

    Args:
        exchange: The CCXT exchange instance (used for fallback).
        market_info: Enhanced MarketInfo for the symbol.
        amount: The amount value (Decimal, float, or string).

    Returns:
        The amount formatted as a string according to market precision (rounded down),
        or None on error or invalid input.
    """
    if amount is None: return None
    # Safely convert input to Decimal first
    amount_dec = _safe_decimal_conversion(amount, allow_none=True)

    if amount_dec is None or not amount_dec.is_finite() or amount_dec <= POSITION_QTY_EPSILON:
         init_logger.error(f"Format Amount: Cannot format invalid, non-finite, or non-positive amount: {amount}")
         return None

    symbol = market_info['symbol']
    amount_step = market_info.get('amount_precision_step_decimal')

    if amount_step is None or not amount_step.is_finite() or amount_step <= POSITION_QTY_EPSILON:
        # Fallback to CCXT's amount_to_precision if step is invalid or missing
        init_logger.warning(f"Format Amount: Invalid/missing amount precision step for {symbol} ({amount_step}). Falling back to CCXT's amount_to_precision.")
        try:
            # CCXT formatter expects float
            return exchange.amount_to_precision(symbol, float(amount_dec))
        except Exception as e:
             init_logger.error(f"Format Amount: CCXT fallback failed for {symbol}: {e}. Returning raw decimal string.", exc_info=True)
             return str(amount_dec.normalize()) # Last resort: raw decimal string
    try:
        # Quantize the amount DOWN to the market's amount step
        formatted_amount_dec = (amount_dec / amount_step).quantize(Decimal("1"), rounding=ROUND_DOWN) * amount_step
        # Check if rounding down resulted in zero when input was positive
        if formatted_amount_dec <= POSITION_QTY_EPSILON and amount_dec > POSITION_QTY_EPSILON:
             init_logger.warning(f"Format Amount: Rounding down amount {amount_dec.normalize()} with step {amount_step.normalize()} resulted in zero. Amount might be below minimum step.")
             # Return None as a zero quantity order is usually invalid.
             return None
        return str(formatted_amount_dec.normalize()) # Normalize to remove trailing zeros
    except (ValueError, InvalidOperation) as e:
        init_logger.error(f"Format Amount: Error quantizing amount {amount_dec.normalize()} for {symbol} with step {amount_step.normalize()}: {e}. Returning raw decimal string.", exc_info=True)
        return str(amount_dec.normalize()) # Fallback to raw decimal string


# --- Market and Position Data Enhancement ---
def enhance_market_info(market: Dict[str, Any]) -> MarketInfo:
    """
    Adds custom fields and Decimal types to ccxt market dict for easier access and calculation.

    Args:
        market (Dict[str, Any]): The raw market dictionary from ccxt.exchange.market().

    Returns:
        MarketInfo: An enhanced dictionary with parsed decimal values and convenience flags.
    """
    enhanced: MarketInfo = market.copy() # type: ignore # Allow copy despite TypedDict

    # Basic contract type detection
    is_contract = market.get('contract', False) or market.get('swap', False) or market.get('future', False) or market.get('option', False)
    is_linear = market.get('linear', False) and is_contract
    is_inverse = market.get('inverse', False) and is_contract
    contract_type = "Linear" if is_linear else "Inverse" if is_inverse else ("Spot" if market.get('spot', False) else ("Option" if market.get('option', False) else "Unknown"))

    # --- Convert precision/limits to Decimal using safe helper ---
    limits = market.get('limits', {})
    amount_limits = limits.get('amount', {})
    cost_limits = limits.get('cost', {})
    price_limits = limits.get('price', {})
    precision = market.get('precision', {})
    symbol = market.get('symbol', 'N/A') # For logging

    # Determine step size (tick size) from precision. Handles integer (places) or float (step).
    amount_step = None
    amount_prec_val = precision.get('amount')
    if isinstance(amount_prec_val, int): amount_step = Decimal('1e-' + str(amount_prec_val))
    elif amount_prec_val is not None: amount_step = _safe_market_decimal(amount_prec_val, f"{symbol}.precision.amount", allow_zero=False)

    price_step = None
    price_prec_val = precision.get('price')
    if isinstance(price_prec_val, int): price_step = Decimal('1e-' + str(price_prec_val))
    elif price_prec_val is not None: price_step = _safe_market_decimal(price_prec_val, f"{symbol}.precision.price", allow_zero=False)

    contract_size = market.get('contractSize', 1.0) # Default to 1.0 if not specified

    # Assign enhanced fields using setdefault or direct assignment
    enhanced['is_contract'] = is_contract
    enhanced['is_linear'] = is_linear
    enhanced['is_inverse'] = is_inverse
    enhanced['contract_type_str'] = contract_type

    # Safely parse limits
    enhanced['min_amount_decimal'] = _safe_market_decimal(amount_limits.get('min'), f"{symbol}.limits.amount.min", allow_zero=True)
    enhanced['max_amount_decimal'] = _safe_market_decimal(amount_limits.get('max'), f"{symbol}.limits.amount.max")
    enhanced['min_cost_decimal'] = _safe_market_decimal(cost_limits.get('min'), f"{symbol}.limits.cost.min", allow_zero=True)
    enhanced['max_cost_decimal'] = _safe_market_decimal(cost_limits.get('max'), f"{symbol}.limits.cost.max")
    enhanced['min_price_decimal'] = _safe_market_decimal(price_limits.get('min'), f"{symbol}.limits.price.min", allow_zero=True)
    enhanced['max_price_decimal'] = _safe_market_decimal(price_limits.get('max'), f"{symbol}.limits.price.max")

    enhanced['amount_precision_step_decimal'] = amount_step
    enhanced['price_precision_step_decimal'] = price_step

    # Contract size - ensure it's a positive Decimal, default to 1.0
    enhanced['contract_size_decimal'] = _safe_market_decimal(contract_size, f"{symbol}.contractSize", default=Decimal('1.0'), allow_zero=False)
    if enhanced['contract_size_decimal'] is None or enhanced['contract_size_decimal'] <= POSITION_QTY_EPSILON:
         # init_logger might not be available if called early, use print as fallback
         log_func = init_logger.warning if 'init_logger' in globals() else print
         log_func(f"{NEON_YELLOW}Market data parsing: Invalid or non-positive contract size for {symbol}. Defaulting to 1.0.{RESET}")
         enhanced['contract_size_decimal'] = Decimal('1.0')

    return enhanced # type: ignore # Return as MarketInfo TypedDict

def enhance_position_info(position: Dict[str, Any], market_info: MarketInfo) -> PositionInfo:
    """
    Adds custom fields and Decimal types to ccxt position dict for easier access and calculation.
    Also extracts raw and parsed native protection order details from the 'info' dictionary.

    Args:
        position (Dict[str, Any]): The raw position dictionary from ccxt.exchange.fetch_positions().
        market_info (MarketInfo): The enhanced market information for this symbol.

    Returns:
        PositionInfo: An enhanced dictionary with parsed decimal values and state flags.
    """
    enhanced: PositionInfo = position.copy() # type: ignore # Allow copy despite TypedDict
    symbol = market_info['symbol']
    info = position.get('info', {}) # Raw exchange-specific data

    # --- Convert key numeric fields to Decimal ---
    # Bybit V5 uses 'size' in info for quantity (contracts/base units)
    size_raw = info.get('size') if info.get('size') is not None else position.get('contracts')
    enhanced['size_decimal'] = _safe_market_decimal(size_raw, f"{symbol}.position.size/contracts", allow_zero=True, allow_negative=False, default=Decimal('0')) # Size should be positive from Bybit

    # Determine side based on 'info.side' (Buy/Sell) or fallback based on unified 'side' ('long'/'short')
    # Bybit V5 uses 'Buy' for long, 'Sell' for short in position info.side
    side_raw = info.get('side')
    if side_raw == 'Buy':
        enhanced['side'] = 'long'
    elif side_raw == 'Sell':
        enhanced['side'] = 'short'
        # Make size negative for short positions for internal consistency if needed by strategy logic
        # However, keeping size positive and using the 'side' field is often cleaner. Let's keep size positive.
        # enhanced['size_decimal'] *= Decimal('-1')
    elif abs(enhanced['size_decimal']) <= POSITION_QTY_EPSILON:
        enhanced['side'] = 'none' # Explicitly mark flat
        enhanced['size_decimal'] = Decimal('0') # Ensure zero size if flat
    else:
        # Fallback if info.side is missing/unexpected, use ccxt unified side if available
        unified_side = position.get('side')
        enhanced['side'] = unified_side if unified_side in ['long', 'short'] else 'none'
        if enhanced['side'] == 'none': enhanced['size_decimal'] = Decimal('0')

    enhanced['entryPrice_decimal'] = _safe_market_decimal(info.get('avgPrice'), f"{symbol}.position.info.avgPrice") # Bybit V5 uses avgPrice
    if enhanced['entryPrice_decimal'] is None: # Fallback to unified field
        enhanced['entryPrice_decimal'] = _safe_market_decimal(position.get('entryPrice'), f"{symbol}.position.entryPrice")

    enhanced['markPrice_decimal'] = _safe_market_decimal(position.get('markPrice'), f"{symbol}.position.markPrice")
    enhanced['liquidationPrice_decimal'] = _safe_market_decimal(info.get('liqPrice'), f"{symbol}.position.info.liqPrice") # Bybit V5 uses liqPrice
    if enhanced['liquidationPrice_decimal'] is None: # Fallback
        enhanced['liquidationPrice_decimal'] = _safe_market_decimal(position.get('liquidationPrice'), f"{symbol}.position.liquidationPrice")

    enhanced['leverage_decimal'] = _safe_market_decimal(info.get('leverage'), f"{symbol}.position.info.leverage", default=Decimal('1.0')) # Bybit V5 uses leverage
    if enhanced['leverage_decimal'] is None: # Fallback
        enhanced['leverage_decimal'] = _safe_market_decimal(position.get('leverage'), f"{symbol}.position.leverage", default=Decimal('1.0'))

    enhanced['unrealizedPnl_decimal'] = _safe_market_decimal(info.get('unrealisedPnl'), f"{symbol}.position.info.unrealisedPnl", allow_zero=True, allow_negative=True) # Bybit V5 spelling
    if enhanced['unrealizedPnl_decimal'] is None: # Fallback
        enhanced['unrealizedPnl_decimal'] = _safe_market_decimal(position.get('unrealizedPnl'), f"{symbol}.position.unrealizedPnl", allow_zero=True, allow_negative=True)

    enhanced['notional_decimal'] = _safe_market_decimal(position.get('notional'), f"{symbol}.position.notional", allow_zero=True, allow_negative=True)
    enhanced['collateral_decimal'] = _safe_market_decimal(position.get('collateral'), f"{symbol}.position.collateral", allow_zero=True, allow_negative=True)
    enhanced['initialMargin_decimal'] = _safe_market_decimal(position.get('initialMargin'), f"{symbol}.position.initialMargin", allow_zero=True)
    enhanced['maintenanceMargin_decimal'] = _safe_market_decimal(position.get('maintenanceMargin'), f"{symbol}.position.maintenanceMargin", allow_zero=True)

    # --- Extract Raw and Parsed Native Protection Info (Bybit V5 specifics from 'info') ---
    enhanced['stopLossPrice_raw'] = info.get('stopLoss')
    enhanced['takeProfitPrice_raw'] = info.get('takeProfit')
    enhanced['trailingStopPrice_raw'] = info.get('trailingStop') # TSL Trigger Price
    enhanced['tslActivationPrice_raw'] = info.get('activePrice') # TSL Activation Price

    # Safely parse raw stop prices to Decimal, treat '0'/'0.0' as None (not set)
    enhanced['stopLossPrice_dec'] = _safe_decimal_conversion(enhanced['stopLossPrice_raw'], allow_none=True)
    enhanced['takeProfitPrice_dec'] = _safe_decimal_conversion(enhanced['takeProfitPrice_raw'], allow_none=True)
    enhanced['trailingStopPrice_dec'] = _safe_decimal_conversion(enhanced['trailingStopPrice_raw'], allow_none=True)
    enhanced['tslActivationPrice_dec'] = _safe_decimal_conversion(enhanced['tslActivationPrice_raw'], allow_none=True)

    # Filter out zero prices which mean "not set"
    if enhanced['stopLossPrice_dec'] is not None and enhanced['stopLossPrice_dec'].is_zero(): enhanced['stopLossPrice_dec'] = None
    if enhanced['takeProfitPrice_dec'] is not None and enhanced['takeProfitPrice_dec'].is_zero(): enhanced['takeProfitPrice_dec'] = None
    if enhanced['trailingStopPrice_dec'] is not None and enhanced['trailingStopPrice_dec'].is_zero(): enhanced['trailingStopPrice_dec'] = None
    if enhanced['tslActivationPrice_dec'] is not None and enhanced['tslActivationPrice_dec'].is_zero(): enhanced['tslActivationPrice_dec'] = None

    # Initialize bot state flags (will be managed per symbol in main loop's state)
    enhanced['be_activated'] = False
    enhanced['tsl_activated'] = False # Flag whether native TSL is considered active based on price/activation price

    return enhanced # type: ignore # Return as PositionInfo TypedDict

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes and validates the CCXT Bybit exchange object.

    Steps:
    1. Sets API keys, rate limiting, default type (linear), timeouts from config.
    2. Configures sandbox mode based on `config.json`.
    3. Loads exchange markets with retries, ensuring markets are actually populated.
    4. Performs an initial balance check for the configured `QUOTE_CURRENCY`.
       - If trading is enabled, a failed balance check is treated as a fatal error.
       - If trading is disabled, logs a warning but allows proceeding.
    5. Sends startup notification.

    Args:
        logger (logging.Logger): The logger instance to use for status messages.

    Returns:
        Optional[ccxt.Exchange]: The initialized ccxt.Exchange object if successful, otherwise None.
    """
    lg = logger # Alias for convenience
    try:
        # Get API timing config or use defaults
        api_timing_cfg = CONFIG.get('api_timing', {})
        default_timeouts = DEFAULT_CONFIG.get('api_timing', {})

        # Common CCXT exchange options
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': 'linear', # Assume linear contracts by default for V5
                'adjustForTimeDifference': True, # Auto-adjust for clock skew
                'recvWindow': api_timing_cfg.get('recv_window', default_timeouts.get('recv_window', 10000)),
                # Timeouts for various operations (in milliseconds)
                'fetchTickerTimeout': api_timing_cfg.get('fetch_ticker_timeout', default_timeouts.get('fetch_ticker_timeout', 15000)),
                'fetchBalanceTimeout': api_timing_cfg.get('fetch_balance_timeout', default_timeouts.get('fetch_balance_timeout', 20000)),
                'createOrderTimeout': api_timing_cfg.get('create_order_timeout', default_timeouts.get('create_order_timeout', 30000)),
                'cancelOrderTimeout': api_timing_cfg.get('cancel_order_timeout', default_timeouts.get('cancel_order_timeout', 20000)),
                'fetchPositionsTimeout': api_timing_cfg.get('fetch_positions_timeout', default_timeouts.get('fetch_positions_timeout', 20000)),
                'fetchOHLCVTimeout': api_timing_cfg.get('fetch_ohlcv_timeout', default_timeouts.get('fetch_ohlcv_timeout', 60000)),
                'fetchOrderTimeout': 20000, # Add timeout for fetchOrder
                'editOrderTimeout': api_timing_cfg.get('edit_order_timeout', default_timeouts.get('edit_order_timeout', 25000)), # Add timeout for editOrder
            }
        }
        # Instantiate the Bybit exchange object
        exchange = ccxt.bybit(exchange_options)

        # Configure Sandbox Mode
        is_sandbox = CONFIG.get('use_sandbox', True)
        exchange.set_sandbox_mode(is_sandbox)
        env_type = "Sandbox (Testnet)" if is_sandbox else "LIVE (Real Funds)"
        if is_sandbox:
            lg.warning(f"{NEON_YELLOW}<<< USING SANDBOX MODE (Testnet Environment) >>>{RESET}")
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}!!! <<< USING LIVE TRADING ENVIRONMENT - REAL FUNDS AT RISK >>> !!!{RESET}")

        # Load Markets with Retries
        lg.info(f"Attempting to load markets for {exchange.id} ({env_type})...")
        markets_loaded = False
        last_market_error = None
        max_retries = CONFIG.get('retry_count', MAX_API_RETRIES)
        retry_delay = CONFIG.get('retry_delay', RETRY_DELAY_SECONDS)

        for attempt in range(max_retries + 1):
            try:
                exchange.load_markets(reload=(attempt > 0)) # Force reload on retries
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"{NEON_GREEN}Markets loaded successfully ({len(exchange.markets)} symbols found).{RESET}")
                    markets_loaded = True
                    break # Exit retry loop on success
                else:
                    last_market_error = ValueError("Market loading returned empty result")
                    lg.warning(f"Market loading returned empty result (Attempt {attempt + 1}/{max_retries + 1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_market_error = e
                lg.warning(f"{NEON_YELLOW}Network error loading markets (Attempt {attempt + 1}/{max_retries + 1}): {e}.{RESET}")
            except ccxt.AuthenticationError as e:
                 last_market_error = e
                 lg.critical(f"{NEON_RED}Authentication error loading markets: {e}. Check API Key/Secret/Permissions. Exiting.{RESET}")
                 return None
            except Exception as e:
                last_market_error = e
                lg.critical(f"{NEON_RED}Unexpected error loading markets: {e}. Exiting.{RESET}", exc_info=True)
                return None

            # Apply exponential backoff delay before retrying
            if not markets_loaded and attempt < max_retries:
                 delay = retry_delay * (2 ** attempt) # Exponential backoff
                 lg.warning(f"Retrying market load in {delay:.1f} seconds...")
                 time.sleep(delay)

        if not markets_loaded:
            lg.critical(f"{NEON_RED}Failed to load markets for {exchange.id} after all retries. Last error: {last_market_error}. Exiting.{RESET}")
            return None

        lg.info(f"CCXT exchange initialized: {exchange.id} | Version: {ccxt.__version__} | Sandbox: {is_sandbox}")

        # Initial Balance Check
        lg.info(f"Attempting initial balance fetch for quote currency ({QUOTE_CURRENCY})...")
        initial_balance: Optional[Decimal] = None
        try:
            balance_data = fetch_balance(exchange, logger) # Use helper with retries
            if balance_data and QUOTE_CURRENCY in balance_data.get('total', {}):
                 initial_balance = _safe_decimal_conversion(balance_data['total'][QUOTE_CURRENCY])
        except ccxt.AuthenticationError as auth_err:
            lg.critical(f"{NEON_RED}Authentication Error during initial balance fetch: {auth_err}. Check API Key/Secret/Permissions. Exiting.{RESET}")
            return None
        except Exception as balance_err:
             lg.warning(f"{NEON_YELLOW}Initial balance fetch encountered an error: {balance_err}.{RESET}", exc_info=True)

        # Evaluate balance check result based on trading mode
        if initial_balance is not None:
            lg.info(f"{NEON_GREEN}Initial total balance: {initial_balance.normalize()} {QUOTE_CURRENCY}{RESET}")
            # Send notification about startup success
            if CONFIG.get("notifications", {}).get("enable_notifications", False):
                 send_notification(f"Bot Started ({exchange.id} {env_type})",
                                   f"Bot v{BOT_VERSION} initialized for {CONFIG.get('trading_pairs', [])}. Bal({QUOTE_CURRENCY}): {initial_balance.normalize()}",
                                   lg)
            return exchange # Success!
        else:
            lg.error(f"{NEON_RED}Initial balance fetch FAILED for {QUOTE_CURRENCY}.{RESET}")
            if CONFIG.get('enable_trading', False):
                lg.critical(f"{NEON_RED}Trading is enabled, but initial balance check failed. Cannot proceed safely. Exiting.{RESET}")
                if CONFIG.get("notifications", {}).get("enable_notifications", False):
                    send_notification(f"Bot FATAL Error ({exchange.id} {env_type})",
                                       f"Bot v{BOT_VERSION} failed init. Critical: Initial balance fetch failed.", lg)
                return None
            else:
                lg.warning(f"{NEON_YELLOW}Trading is disabled. Proceeding without confirmed initial balance, but errors might occur later.{RESET}")
                if CONFIG.get("notifications", {}).get("enable_notifications", False):
                    send_notification(f"Bot Warning ({exchange.id} {env_type})",
                                       f"Bot v{BOT_VERSION} initialized (Trading Disabled), but initial balance fetch failed. Investigate.", lg)
                return exchange # Allow proceeding in non-trading mode

    except Exception as e:
        lg.critical(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        env_type_notify = "Sandbox" if CONFIG.get('use_sandbox', True) else "LIVE"
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             try:
                 send_notification(f"Bot FATAL Error ({env_type_notify})",
                                   f"Bot v{BOT_VERSION} failed init. Error: {type(e).__name__} - {e}", lg)
             except Exception as notify_err:
                  print(f"{NEON_RED}FATAL: Failed to send notification about initialization error: {notify_err}{RESET}", file=sys.stderr)
        return None

# --- CCXT Data Fetching Helpers ---
def fetch_balance(exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Fetches account balance using CCXT's fetch_balance with retries.
    Handles Bybit V5 'category' parameter automatically for linear accounts.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        logger (logging.Logger): The logger instance.

    Returns:
        Optional[Dict[str, Any]]: The balance dictionary, or None on failure.
    """
    lg = logger
    attempts = 0
    last_exception = None
    max_retries = CONFIG.get('retry_count', MAX_API_RETRIES)
    retry_delay = CONFIG.get('retry_delay', RETRY_DELAY_SECONDS)

    while attempts <= max_retries:
        try:
            lg.debug(f"Fetching balance data (Attempt {attempts + 1}/{max_retries + 1})...")
            # Bybit V5 fetchBalance requires category for unified account (linear/inverse/spot)
            params = {'category': 'linear'} # Assume linear for this bot
            balance = exchange.fetch_balance(params=params)
            if isinstance(balance, dict) and ('total' in balance or 'free' in balance):
                 lg.debug("Balance data fetched successfully.")
                 return balance
            else:
                 last_exception = ValueError(f"Fetch balance returned invalid structure: {balance}")
                 lg.warning(f"Fetched balance has unexpected structure (Attempt {attempts + 1}). Retrying...")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching balance: {e}. Retry {attempts + 1}/{max_retries + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * 3 # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Skip attempt increment and retry delay below
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.error(f"{NEON_RED}Authentication Error fetching balance: {e}. Check API Key/Secret/Permissions. Stopping fetch.{RESET}")
             return None # Fatal error
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}")
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
            return None # Exit on unexpected errors

        attempts += 1
        if attempts <= max_retries:
            delay = retry_delay * (2 ** (attempts - 1)) # Exponential backoff
            time.sleep(delay)

    lg.error(f"{NEON_RED}Failed to fetch balance after {max_retries + 1} attempts. Last error: {last_exception}{RESET}")
    return None

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using `fetch_ticker` with retries.
    Prioritizes 'last' price, falls back to mid-price, ask, then bid.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        symbol (str): The trading symbol (e.g., "BTC/USDT").
        logger (logging.Logger): The logger instance for status messages.

    Returns:
        Optional[Decimal]: The current price as a Decimal, or None if fetching fails.
    """
    lg = logger
    attempts = 0
    last_exception = None
    max_retries = CONFIG.get('retry_count', MAX_API_RETRIES)
    retry_delay = CONFIG.get('retry_delay', RETRY_DELAY_SECONDS)

    while attempts <= max_retries:
        try:
            lg.debug(f"Fetching ticker data for {symbol} (Attempt {attempts + 1}/{max_retries + 1})")
            ticker = exchange.fetch_ticker(symbol)
            price: Optional[Decimal] = None

            # Helper to safely convert ticker values to positive Decimal
            def safe_decimal_from_ticker(value: Optional[Any], field_name: str) -> Optional[Decimal]:
                dec_val = _safe_decimal_conversion(value, allow_none=True)
                if dec_val is not None and dec_val > POSITION_QTY_EPSILON:
                    return dec_val
                elif dec_val is not None: # Log if zero or negative
                     lg.debug(f"Ticker field '{field_name}' value '{value}' is zero or negative.")
                return None

            # 1. Try 'last' price
            price = safe_decimal_from_ticker(ticker.get('last'), 'last')

            # 2. Fallback to mid-price
            if price is None:
                bid = safe_decimal_from_ticker(ticker.get('bid'), 'bid')
                ask = safe_decimal_from_ticker(ticker.get('ask'), 'ask')
                if bid is not None and ask is not None:
                    price = (bid + ask) / Decimal('2')
                    lg.debug(f"Using mid-price (Bid: {bid.normalize()}, Ask: {ask.normalize()}) -> {price.normalize()}")
                # 3. Fallback to 'ask'
                elif ask is not None:
                    price = ask
                    lg.debug(f"Using 'ask' price as fallback: {price.normalize()}")
                # 4. Fallback to 'bid'
                elif bid is not None:
                    price = bid
                    lg.debug(f"Using 'bid' price as fallback: {price.normalize()}")

            if price is not None:
                lg.debug(f"Current price successfully fetched for {symbol}: {price.normalize()}")
                return price.normalize()
            else:
                last_exception = ValueError(f"No valid price found in ticker data. Ticker: {ticker}")
                lg.warning(f"No valid price found in ticker (Attempt {attempts + 1}). Retrying...")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retry {attempts + 1}/{max_retries + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error fetching price: {e}. Stopping fetch.{RESET}")
             return None
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None

        attempts += 1
        if attempts <= max_retries:
            delay = retry_delay * (2 ** (attempts - 1)) # Exponential backoff
            time.sleep(delay)

    lg.error(f"{NEON_RED}Failed to fetch current price for {symbol} after {max_retries + 1} attempts. Last error: {last_exception}{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, market_info: MarketInfo, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches OHLCV (kline) data using CCXT's `fetch_ohlcv` with enhancements.

    - Handles Bybit V5 'category' parameter automatically based on market info.
    - Implements robust retry logic with exponential backoff.
    - Validates fetched data timestamp lag.
    - Processes data into a Pandas DataFrame with Decimal types.
    - Cleans data (NaN handling, zero price/volume checks).
    - Trims DataFrame to `MAX_DF_LEN`.
    - Ensures DataFrame is sorted by timestamp.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        market_info (MarketInfo): The enhanced market info for the symbol.
        timeframe (str): The CCXT timeframe string (e.g., "5m", "1h").
        limit (int): The desired number of klines (capped by exchange limits).
        logger (logging.Logger): The logger instance.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the OHLCV data, or an empty DataFrame on failure.
    """
    lg = logger
    symbol = market_info['symbol']
    attempts = 0
    last_exception = None
    ohlcv_data: List[List[Union[int, float]]] = []
    max_retries = CONFIG.get('retry_count', MAX_API_RETRIES)
    retry_delay = CONFIG.get('retry_delay', RETRY_DELAY_SECONDS)
    # Fetch limit is capped by config, function arg, and exchange max per request
    fetch_limit_actual = min(limit, CONFIG.get('fetch_limit', DEFAULT_FETCH_LIMIT), BYBIT_API_KLINE_LIMIT)

    if fetch_limit_actual <= 0:
         lg.error(f"{NEON_RED}Kline fetch limit is zero or negative ({fetch_limit_actual}). Cannot fetch data.{RESET}")
         return pd.DataFrame()

    # Determine category based on market type
    category = 'spot' if market_info.get('spot') else ('linear' if market_info.get('is_linear') else ('inverse' if market_info.get('is_inverse') else 'linear')) # Default linear
    params = {'category': category}

    while attempts <= max_retries:
        try:
            lg.debug(f"Fetching {fetch_limit_actual} klines for {symbol} ({timeframe}, Cat: {category}) (Attempt {attempts + 1}/{max_retries + 1})...")
            # Note: CCXT's fetch_ohlcv limit is the number of candles requested.
            # Bybit V5 API limit is 1000 per request. CCXT doesn't automatically paginate fetch_ohlcv for Bybit.
            # We request `fetch_limit_actual` which is already capped at 1000.
            ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe, limit=fetch_limit_actual, params=params)

            if not ohlcv_data:
                last_exception = ValueError("Fetch OHLCV returned empty data.")
                lg.warning(f"Fetched OHLCV data is empty for {symbol} ({timeframe}) (Attempt {attempts + 1}). Retrying...")
            else:
                lg.debug(f"Successfully fetched {len(ohlcv_data)} klines for {symbol} ({timeframe}).")
                break # Exit retry loop

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching klines for {symbol} ({timeframe}): {e}. Retry {attempts + 1}/{max_retries + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching klines for {symbol} ({timeframe}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error fetching klines: {e}. Stopping fetch.{RESET}")
             return pd.DataFrame()
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol} ({timeframe}): {e}{RESET}")
            # Check if symbol/timeframe is valid
            if "Invalid symbol" in str(e) or "Invalid interval" in str(e):
                 lg.critical(f"{NEON_RED}Invalid symbol or timeframe specified: {symbol} / {timeframe}. Stopping fetch.{RESET}")
                 return pd.DataFrame()
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching klines for {symbol} ({timeframe}): {e}{RESET}", exc_info=True)
            return pd.DataFrame()

        attempts += 1
        if attempts <= max_retries:
            delay = retry_delay * (2 ** (attempts - 1)) # Exponential backoff
            time.sleep(delay)

    if not ohlcv_data:
        lg.error(f"{NEON_RED}Failed to fetch klines for {symbol} ({timeframe}) after {max_retries + 1} attempts. Last error: {last_exception}{RESET}")
        return pd.DataFrame()

    # --- Data Processing & Cleaning ---
    df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    numeric_cols = ["open", "high", "low", "close", "volume"]

    try:
        # Convert timestamp to datetime (UTC) and set index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True, drop=False) # Keep timestamp column

        # Convert OHLCV to numeric, coercing errors, then to Decimal
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Apply safe decimal conversion AFTER initial numeric conversion
            df[col] = df[col].apply(_safe_decimal_conversion, default=Decimal(0), allow_none=True) # Allow None temporarily

        # --- Robust NaN Handling ---
        initial_nan_count = df[numeric_cols].isnull().sum().sum()
        if initial_nan_count > 0:
            nan_counts_per_col = df[numeric_cols].isnull().sum()
            lg.warning(f"{NEON_YELLOW}Data Fetch: Found {initial_nan_count} NaN values in OHLCV for {symbol} after conversion:\n"
                           f"{nan_counts_per_col[nan_counts_per_col > 0]}\nAttempting forward fill (ffill)...{RESET}")
            df.ffill(inplace=True) # Forward fill first

            remaining_nan_count = df[numeric_cols].isnull().sum().sum()
            if remaining_nan_count > 0:
                lg.warning(f"{NEON_YELLOW}NaNs remain after ffill ({remaining_nan_count}). Attempting backward fill (bfill)...{RESET}")
                df.bfill(inplace=True) # Backward fill remaining NaNs

                final_nan_count = df[numeric_cols].isnull().sum().sum()
                if final_nan_count > 0:
                    lg.error(f"{NEON_RED}Data Fetch: Unfillable NaN values ({final_nan_count}) remain after ffill/bfill for {symbol}. Data quality insufficient. Returning empty DataFrame.{RESET}")
                    return pd.DataFrame()

        # Ensure no zero prices or negative volumes after cleaning
        if df.empty or \
           (df[['open', 'high', 'low', 'close']] <= POSITION_QTY_EPSILON).any().any() or \
           (df['volume'] < 0).any():
             lg.warning(f"{NEON_YELLOW}Data Fetch: Found zero/negative prices or negative volumes after cleaning for {symbol}. Returning empty DataFrame.{RESET}")
             return pd.DataFrame()

        # Ensure DataFrame is sorted by timestamp ascending
        df.sort_index(inplace=True)

        # Trim DataFrame to MAX_DF_LEN
        if len(df) > MAX_DF_LEN:
            original_len = len(df)
            df = df.tail(MAX_DF_LEN).copy()
            lg.debug(f"Data Fetch: Trimmed DataFrame for {symbol} from {original_len} to {len(df)} rows.")

        # Check for stale data (last candle timestamp)
        if not df.empty:
            last_candle_time_utc = df.index[-1]
            now_utc = pd.Timestamp.now(tz='UTC')
            interval_seconds = exchange.parse_timeframe(timeframe) or 60 # Default 60s if parse fails
            time_diff_seconds = (now_utc - last_candle_time_utc).total_seconds()

            # Allow some buffer (e.g., 10% of interval or 5 seconds minimum)
            allowed_lag = max(interval_seconds * 0.1, 5.0)
            # Critical lag (e.g., 2 intervals)
            critical_lag = interval_seconds * 2

            if time_diff_seconds < allowed_lag:
                lg.debug(f"Data Fetch: Last candle {symbol} ({timeframe}) timestamp {last_candle_time_utc.strftime('%H:%M:%S')} is very recent ({time_diff_seconds:.1f}s ago).")
            elif time_diff_seconds >= critical_lag:
                 lg.warning(f"{NEON_YELLOW}Data Fetch: Last candle {symbol} ({timeframe}) timestamp {last_candle_time_utc.strftime('%H:%M:%S')} is potentially stale ({time_diff_seconds:.1f}s ago, interval={interval_seconds}s).{RESET}")
            # else: Normal lag

    except Exception as processing_err:
        lg.error(f"{NEON_RED}Data Fetch: Error processing kline data for {symbol}: {processing_err}{RESET}", exc_info=True)
        return pd.DataFrame() # Return empty on processing error

    lg.debug(f"Data Fetch: Successfully processed {len(df)} OHLCV candles for {symbol}.")
    return df

def get_current_position(exchange: ccxt.Exchange, market_info: MarketInfo, logger: logging.Logger) -> PositionInfo:
    """
    Fetches current position details for a specific symbol using Bybit V5 API specifics.
    Assumes One-Way Mode (looks for positionIdx=0). Includes retries and enhancement.

    Args:
        exchange (ccxt.Exchange): The CCXT exchange instance.
        market_info (MarketInfo): The enhanced market info for the symbol.
        logger (logging.Logger): The logger instance.

    Returns:
        PositionInfo: An enhanced PositionInfo dictionary. Returns a default flat state
                      dictionary if fetching fails or no position is found.
    """
    lg = logger
    symbol = market_info['symbol']
    # Initialize default flat state dictionary
    default_pos: PositionInfo = { # type: ignore # Initialize with required fields, enhanced fields added later
        'symbol': symbol, 'side': 'none', 'size_decimal': Decimal("0.0"), 'info': {},
        'be_activated': False, 'tsl_activated': False
    }
    # Add None placeholders for other fields to match TypedDict structure
    for key in PositionInfo.__annotations__:
         if key not in default_pos: default_pos[key] = None # type: ignore

    max_retries = CONFIG.get('retry_count', MAX_API_RETRIES)
    retry_delay = CONFIG.get('retry_delay', RETRY_DELAY_SECONDS)
    attempts = 0
    last_exception = None

    # Determine category based on market type
    category = 'spot' if market_info.get('spot') else ('linear' if market_info.get('is_linear') else ('inverse' if market_info.get('is_inverse') else 'linear'))
    params = {'category': category}
    # Use exchange-specific market ID for filtering if available
    market_id = market_info.get('id', symbol)
    params['symbol'] = market_id

    while attempts <= max_retries:
        try:
            lg.debug(f"Fetching position data for {symbol} (Cat: {category}, ID: {market_id}) (Attempt {attempts + 1}/{max_retries + 1})...")

            # Fetch positions for the specific symbol and category
            # CCXT fetch_positions returns a list.
            positions = exchange.fetch_positions(symbols=[symbol], params=params)

            # Filter for the One-Way position (positionIdx=0)
            relevant_position_raw = None
            for p in positions:
                # Double check symbol just in case CCXT filter failed
                if p.get('symbol') != symbol: continue
                # Check for positionIdx=0 in raw info
                pos_idx_raw = p.get('info', {}).get('positionIdx')
                if pos_idx_raw is not None and str(pos_idx_raw) == '0':
                     relevant_position_raw = p
                     break # Found the One-Way position

            if relevant_position_raw:
                 enhanced_pos = enhance_position_info(relevant_position_raw, market_info)
                 if enhanced_pos['side'] != 'none': # Check derived side after enhancement
                      lg.debug(f"{NEON_GREEN}Active {enhanced_pos['side'].capitalize()} position found for {symbol}. "
                               f"Qty: {enhanced_pos['size_decimal'].normalize()}, Entry: {enhanced_pos['entryPrice_decimal'].normalize() if enhanced_pos['entryPrice_decimal'] else 'N/A'}{RESET}")
                      # Log native stops if attached
                      stop_details = []
                      if enhanced_pos.get('stopLossPrice_dec'): stop_details.append(f"SL: {enhanced_pos['stopLossPrice_dec'].normalize()}")
                      if enhanced_pos.get('takeProfitPrice_dec'): stop_details.append(f"TP: {enhanced_pos['takeProfitPrice_dec'].normalize()}")
                      if enhanced_pos.get('trailingStopPrice_dec'): stop_details.append(f"TSL Trig: {enhanced_pos['trailingStopPrice_dec'].normalize()}")
                      if enhanced_pos.get('tslActivationPrice_dec'): stop_details.append(f"TSL Act: {enhanced_pos['tslActivationPrice_dec'].normalize()}")
                      if stop_details: lg.debug(f"{NEON_CYAN}Position Stops: {' | '.join(stop_details)}{RESET}")
                      return enhanced_pos
                 else:
                     lg.debug(f"Position check for {symbol}: PositionIdx=0 found, but size is zero/negligible. Currently Flat.")
                     return default_pos
            else:
                 lg.debug(f"Position check for {symbol}: No position found matching One-Way criteria (Idx=0). Currently Flat.")
                 return default_pos

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching position for {symbol}: {e}. Retry {attempts + 1}/{max_retries + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching position for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error fetching position: {e}. Stopping fetch.{RESET}")
             return default_pos # Fatal, return flat state
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching position for {symbol}: {e}{RESET}")
            # Some exchange errors might indicate no position exists, treat as potentially non-retryable?
            # For now, retry unless it's auth error.
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching position for {symbol}: {e}{RESET}", exc_info=True)
            return default_pos # Return flat on unexpected error

        attempts += 1
        if attempts <= max_retries:
            delay = retry_delay * (2 ** (attempts - 1)) # Exponential backoff
            time.sleep(delay)

    lg.error(f"{NEON_RED}Failed to fetch position for {symbol} after {max_retries + 1} attempts. Last error: {last_exception}. Assuming flat.{RESET}")
    return default_pos

# --- Strategy Implementation ---
# Includes placeholder logic for Volumatic Trend and basic Pivot Order Blocks

def calculate_volumatic_trend(df: pd.DataFrame, vt_length: int, vt_atr_period: int, vt_vol_ema_length: int, vt_atr_multiplier: Decimal, logger: logging.Logger) -> pd.DataFrame:
    """
    Calculates Volumatic Trend indicators (Trend Line, Bands, Volume Norm).
    Uses pandas_ta for calculations, ensures Decimal output where relevant.

    Args:
        df: DataFrame with OHLCV columns (Decimal).
        vt_length: Length for the primary trend line (EMA).
        vt_atr_period: Period for ATR calculation.
        vt_vol_ema_length: Length for Volume EMA used in normalization.
        vt_atr_multiplier: Multiplier for ATR bands (Decimal).
        logger: Logger instance.

    Returns:
        DataFrame with VT columns added. Original DataFrame if calculation fails.
    """
    lg = logger
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    # Check for sufficient data length including buffer
    min_len_needed = max(vt_length, vt_atr_period, vt_vol_ema_length) + 10
    if not all(col in df.columns for col in required_cols) or len(df) < min_len_needed:
        lg.warning(f"{NEON_YELLOW}VT Calc: Insufficient data ({len(df)} rows, need ~{min_len_needed}) or missing columns. Cannot calculate VT.{RESET}")
        # Add NA columns if they don't exist to prevent downstream errors
        for col in ['trend_line', 'atr', 'upper_band', 'lower_band', 'vol_ema', 'vol_ratio', 'vol_norm_int']:
            if col not in df.columns: df[col] = pd.NA
        return df

    try:
        df_calc = df.copy() # Work on a copy
        # Ensure pandas_ta inputs are float (as it often expects/outputs floats)
        df_float = df_calc[required_cols].astype(float)

        # Calculate ATR
        df_calc['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=vt_atr_period).apply(_safe_decimal_conversion, default=Decimal(0), allow_none=True)

        # Calculate Volume EMA and Ratio
        df_calc['vol_ema'] = ta.ema(df_float['volume'], length=vt_vol_ema_length).apply(_safe_decimal_conversion, default=Decimal(0), allow_none=True)
        # Avoid division by zero for vol_ratio
        df_calc['vol_ratio'] = df_calc.apply(lambda row: row['volume'] / row['vol_ema'] if row['vol_ema'] is not None and row['vol_ema'] > POSITION_QTY_EPSILON and row['volume'] is not None else Decimal('1.0'), axis=1)

        # Normalize Volume Ratio (Example: 0-200 integer)
        df_calc['vol_norm_int'] = df_calc['vol_ratio'].apply(lambda x: min(int(x * 100), 200) if x is not None and x.is_finite() else 0)

        # Calculate Trend Line (EMA)
        df_calc['trend_line'] = ta.ema(df_float['close'], length=vt_length).apply(_safe_decimal_conversion, default=Decimal(0), allow_none=True)

        # Calculate ATR Bands (using Decimal multiplication)
        df_calc['upper_band'] = df_calc.apply(lambda row: row['trend_line'] + (row['atr'] * vt_atr_multiplier) if row['trend_line'] is not None and row['atr'] is not None and row['atr'].is_finite() and row['atr'] > 0 else row['trend_line'], axis=1)
        df_calc['lower_band'] = df_calc.apply(lambda row: row['trend_line'] - (row['atr'] * vt_atr_multiplier) if row['trend_line'] is not None and row['atr'] is not None and row['atr'].is_finite() and row['atr'] > 0 else row['trend_line'], axis=1)

        # Check for NaNs in crucial output columns (last row)
        output_cols = ['trend_line', 'atr', 'upper_band', 'lower_band', 'vol_ratio', 'vol_norm_int']
        if not df_calc.empty and df_calc[output_cols].iloc[-1].isnull().any():
            nan_cols = df_calc[output_cols].iloc[-1].isnull()
            nan_details = ', '.join([col for col in output_cols if nan_cols[col]])
            lg.warning(f"{NEON_YELLOW}VT Calc: Calculation resulted in NaN(s) for last candle in: {nan_details}. Signal generation may be affected.{RESET}")

        lg.debug(f"VT Calc: Completed. Last ATR: {df_calc['atr'].iloc[-1].normalize() if not df_calc.empty and df_calc['atr'].iloc[-1] is not None else 'N/A'}, Vol Norm: {df_calc['vol_norm_int'].iloc[-1] if not df_calc.empty else 'N/A'}")
        return df_calc

    except Exception as e:
        lg.error(f"{NEON_RED}VT Calc: Unexpected error during calculation: {e}{RESET}", exc_info=True)
        # Add NA columns on error
        for col in ['trend_line', 'atr', 'upper_band', 'lower_band', 'vol_ema', 'vol_ratio', 'vol_norm_int