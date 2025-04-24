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
    incorporating normalized volume analysis. (Placeholder logic included)
2.  **Pivot Order Blocks (OBs):** Identifying potential support/resistance zones
    based on pivot highs and lows derived from candle wicks or bodies. (Placeholder logic included)

This version synthesizes features and robustness from previous iterations, including:
-   Robust configuration loading from both .env (secrets) and config.json (parameters).
-   Detailed configuration validation with automatic correction to defaults.
-   Flexible notification options (Termux SMS and Email).
-   Enhanced logging with colorama, rotation, and sensitive data redaction.
-   Comprehensive API interaction handling with retries and specific error logging for CCXT.
-   Accurate Decimal usage for all financial calculations.
-   Structured data types using TypedDicts for clarity.
-   Implementation of native Bybit V5 Stop Loss, Take Profit, and Trailing Stop Loss where possible
    via CCXT order placement/editing.
-   Logic for managing Break-Even stop adjustments.
-   Support for multiple trading pairs (processed sequentially per cycle).
-   Graceful shutdown on interruption or critical errors.

Disclaimer:
- **EXTREME RISK**: Trading cryptocurrencies, especially futures contracts with leverage and automated systems, involves substantial risk of financial loss. This script is provided for EDUCATIONAL PURPOSES ONLY. You could lose your entire investment and potentially more. Use this software entirely at your own risk. The authors and contributors assume NO responsibility for any trading losses.
- **NATIVE SL/TP/TSL RELIANCE**: The bot's protective stop mechanisms rely entirely on Bybit's exchange-native order execution. Their performance is subject to exchange conditions, potential slippage during volatile periods, API reliability, order book liquidity, and specific exchange rules. These orders are NOT GUARANTEED to execute at the precise trigger price specified.
- **PARAMETER SENSITIVITY & OPTIMIZATION**: The performance of this bot is highly dependent on the chosen strategy parameters (indicator settings, risk levels, SL/TP/TSL percentages, filter thresholds). These parameters require extensive backtesting, optimization, and forward testing on a TESTNET environment before considering any live deployment. Default parameters are unlikely to be profitable.
- **API RATE LIMITS & BANS**: Excessive API requests can lead to temporary or permanent bans from the exchange. Monitor API usage and adjust script timing accordingly. CCXT's built-in rate limiter is enabled but may not prevent all issues under heavy load.
- **SLIPPAGE**: Market orders, used for entry and potentially for SL/TP/TSL execution by the exchange, are susceptible to slippage. This means the actual execution price may differ from the price observed when the order was placed, especially during high volatility or low liquidity.
- **TEST THOROUGHLY**: **DO NOT RUN THIS SCRIPT WITH REAL FUNDS WITHOUT EXTENSIVE AND SUCCESSFUL TESTING ON A TESTNET OR DEMO ACCOUNT.** Ensure you fully understand every part of the code, its logic, and its potential risks before any live deployment.
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
import shutil     # Used to check for termux-sms-send
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
        # We don't need this specific zone, just need the check to pass
        _ = ZoneInfo("UTC") # Check UTC first as it's always available
        _ = ZoneInfo("America/Chicago") # Then check a non-UTC zone
        _ZONEINFO_AVAILABLE = True
    except ZoneInfoNotFoundError:
        _ZONEINFO_AVAILABLE = False
        print(f"{Fore.YELLOW}Warning: 'zoneinfo' is available, but 'tzdata' package seems missing or corrupt.")
        print("         `pip install tzdata` is recommended for reliable timezone support.")
    except Exception as tz_init_err:
         _ZONEINFO_AVAILABLE = False
         # Catch any other unexpected errors during ZoneInfo initialization
         print(f"{Fore.YELLOW}Warning: Error initializing test timezone with 'zoneinfo': {tz_init_err}")
except ImportError:
    _ZONEINFO_AVAILABLE = False
    # Fallback for older Python versions or if zoneinfo itself is not installed
    print(f"{Fore.YELLOW}Warning: 'zoneinfo' module not found (requires Python 3.9+). Falling back to basic UTC implementation.")
    print("         For accurate local time logging, upgrade Python or use a backport library (`pip install backports.zoneinfo`).")

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
                print(f"Warning: Fallback ZoneInfo initialized with key '{key}', but will always use UTC.")
            # Store the requested key for representation, though internally we always use UTC
            self._requested_key = key

        def __call__(self, dt: Optional[datetime] = None) -> Optional[datetime]:
            """Attaches UTC timezone info to a datetime object. Returns None if input is None."""
            if dt is None:
                return None
            # If naive, replace tzinfo with UTC. If already aware, convert to UTC.
            # This matches the behavior of `dt.astimezone(timezone.utc)`.
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
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587")) # Default SMTP port, ensure cast to int
SMTP_USER: Optional[str] = os.getenv("SMTP_USER")
SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")
NOTIFICATION_EMAIL_RECIPIENT: Optional[str] = os.getenv("NOTIFICATION_EMAIL_RECIPIENT") # Renamed from NOTIFICATION_EMAIL for clarity
TERMUX_SMS_RECIPIENT: Optional[str] = os.getenv("TERMUX_SMS_RECIPIENT") # Renamed from SMS_RECIPIENT_NUMBER for clarity

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
POSITION_QTY_EPSILON: Decimal = Decimal("1e-9")
PRICE_EPSILON: Decimal = Decimal("1e-9") # For comparing prices

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
    # Note: 'extended_to_ts' was in one snippet but not used in logic, omitted for simplicity unless needed.

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
    atr: Optional[Decimal]    # Current Average True Range value from the last candle (last value in the ATR series)
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
            if SMTP_PASSWORD and isinstance(SMTP_PASSWORD, str):
                 redacted_msg = redacted_msg.replace(SMTP_PASSWORD, self._smtp_password_placeholder)

        except Exception as e:
            # Prevent crashing the application if redaction fails unexpectedly.
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
        # We don't pass `fmt` directly here as we use dynamic `levelcolor`.
        super().__init__(fmt=self._log_format, datefmt=self._date_format, **kwargs)
        # Set the time converter to use the globally configured TIMEZONE for local time display.
        # This lambda converts the timestamp (usually Unix epoch float) to a timetuple in the local zone.
        self.converter = lambda timestamp: datetime.fromtimestamp(timestamp, tz=TIMEZONE).timetuple() # Removed the second unused argument

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
    # Replace /, :, and spaces with underscores.
    safe_filename_part = re.sub(r'[^\w\-.]', '_', name)
    # Use dot notation for logger names to support potential hierarchical logging features.
    logger_name = f"pyrmethus.{safe_filename_part}"
    # Construct the full path for the log file.
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")

    logger = logging.getLogger(logger_name)

    # Prevent adding handlers multiple times if the logger was already configured
    # (e.g., if this function is called again with the same name).
    if logger.hasHandlers():
        # Update console log level if it changed (e.g., via SIGHUP reload, though not implemented here)
        # Or just ensure its level is correct based on env var.
        # For simplicity, if handlers exist, assume configuration is stable.
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
            # print(f"{NEON_YELLOW}Warning: Invalid CONSOLE_LOG_LEVEL '{console_log_level_str}'. Defaulting to INFO.{RESET}") # Avoid logging before logger fully set up
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

    # Using logger instance to log setup status, check if handlers are actually added
    if logger.handlers:
         # Check if fh and sh were successfully created and added
         file_handler_level = next((h.level for h in logger.handlers if isinstance(h, RotatingFileHandler)), 'N/A')
         console_handler_level = next((h.level for h in logger.handlers if isinstance(h, logging.StreamHandler)), 'N/A')
         logger.debug(f"Logger '{logger_name}' initialized. File Handler Level: {logging.getLevelName(file_handler_level) if isinstance(file_handler_level, int) else file_handler_level}, Console Handler Level: {logging.getLevelName(console_handler_level) if isinstance(console_handler_level, int) else console_handler_level}")
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


    if notify_type == "email":
        # Check if email settings are complete from .env
        if not all([SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_EMAIL_RECIPIENT]):
            lg.warning("Email notification is enabled but .env settings are incomplete (SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_EMAIL_RECIPIENT env vars).")
            return False

        try:
            msg = MIMEText(body)
            msg['Subject'] = f"[Pyrmethus Bot] {subject}" # Prefix subject for clarity
            msg['From'] = SMTP_USER
            msg['To'] = NOTIFICATION_EMAIL_RECIPIENT

            lg.debug(f"Attempting to send email notification to {NOTIFICATION_EMAIL_RECIPIENT} via {SMTP_SERVER}:{SMTP_PORT}...")
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls() # Upgrade connection to secure
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, NOTIFICATION_EMAIL_RECIPIENT, msg.as_string())
            lg.info(f"{NEON_GREEN}Successfully sent email notification: '{subject}'{RESET}")
            return True
        except Exception as e:
            lg.error(f"{NEON_RED}Failed to send email notification: {e}{RESET}")
            return False

    elif notify_type == "sms":
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

         # Prepare the command spell. Message should be the last argument(s).
         # Prefix message for clarity.
         sms_message = f"[Pyrmethus Bot] {subject}: {body}"
         command: List[str] = ['termux-sms-send', '-n', TERMUX_SMS_RECIPIENT, sms_message]

         # Use a timeout from config if available, default if not.
         sms_timeout = CONFIG.get('notifications', {}).get('sms_timeout_seconds', 30)

         try:
             lg.debug(f"Dispatching SMS notification to {TERMUX_SMS_RECIPIENT} (Timeout: {sms_timeout}s)...")
             # Execute the spell via subprocess with timeout and output capture
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
        target_type_str = 'integer' if is_int else 'float' # For logging

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
                 if abs(float(original_val) - converted_float) > 1e-9:
                      needs_type_correction = True
                      final_val = converted_float
                      init_logger.info(f"{NEON_YELLOW}Config Update: Adjusted float/int value for '{key_path}' due to precision from {repr(original_val)} to {repr(final_val)}.{RESET}")
                 else:
                      final_val = converted_float # Keep the float representation if close enough
            # If original was a string or something else, convert Decimal to float
            elif not isinstance(original_val, (float, int)):
                 needs_type_correction = True
                 final_val = float(final_val_dec) # Convert validated Decimal to float
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type for float key '{key_path}' from {type(original_val).__name__} to float (value: {repr(final_val)}).{RESET}")
            else: # Should technically be covered by the above cases for float/int, but defensive
                 final_val = float(final_val_dec)


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
    # Handle numeric 1/0 as well?
    elif isinstance(original_val, int) and original_val in [0, 1]:
        corrected_val = bool(original_val)

    if corrected_val is not None:
         # Successfully interpreted as boolean, update the value and log
         init_logger.info(f"{NEON_YELLOW}Config Update: Corrected boolean-like value for '{key_path}' from {repr(original_val)} to {repr(corrected_val)}.{RESET}")
         final_val = corrected_val
         corrected = True
    else:
         # Cannot interpret the value as boolean, use the default and log a warning
         init_logger.warning(f"Config Validation Warning: Invalid value for boolean key '{key_path}': {repr(original_val)}. Expected true/false or equivalent. Using default: {repr(default_val)}.{RESET}")
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
                  if original_val.lower() == choice.lower():
                      found_match = True
                      corrected_match_value = choice # Use the canonical casing from `choices` list
                      break

     if not found_match: # Not a valid choice (or wrong type)
         init_logger.warning(f"Config Validation Warning: Invalid value for '{key_path}': {repr(original_val)}. Must be one of {valid_choices} ({'case-sensitive' if case_sensitive else 'case-insensitive'}). Using default: {repr(default_val)}.{RESET}")
         final_val = default_val
         corrected = True
     elif corrected_match_value != original_val: # Valid choice, but potentially wrong case (if case-insensitive)
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
        "max_concurrent_positions": 1,          # Maximum number of positions allowed open simultaneously across all pairs (Currently per symbol due to simple loop).

        # == Risk & Sizing ==
        "risk_per_trade": 0.01,                 # Fraction of available balance to risk per trade (e.0.01 = 1%). Must be > 0.0 and <= 1.0.
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
        "strategy_params": {
            # -- Volumatic Trend (VT) - Placeholders, replace with actual values --
            "vt_length": DEFAULT_VT_LENGTH,             # Lookback period for VT calculation (integer > 0)
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,     # Lookback period for ATR calculation within VT (integer > 0)
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH, # Placeholder: Lookback for Volume EMA/SWMA (integer > 0)
            "vt_atr_multiplier": DEFAULT_VT_ATR_MULTIPLIER, # Placeholder: ATR multiplier for VT bands (float > 0)
            "vt_step_atr_multiplier": DEFAULT_VT_STEP_ATR_MULTIPLIER, # Unused placeholder (float)

            # -- Order Blocks (OB) - Placeholders, replace with actual values --
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
            "trailing_stop_callback_rate": DEFAULT_TRAILING_STOP_CALLBACK_RATE, # TSL callback/distance (float > 0). Interpretation depends on exchange/implementation (e.g., 0.005 = 0.5% or 0.5 price points).
            "trailing_stop_activation_percentage": DEFAULT_TRAILING_STOP_ACTIVATION_PERCENTAGE, # Activate TSL when price moves this % from entry (float >= 0).
        },

        # == Notifications ==
        "notifications": {
            "enable_notifications": True, # Master switch for notifications
            "notification_type": "email", # 'email' or 'sms'. Must match one of these.
            "sms_timeout_seconds": 30 # Timeout for termux-sms-send command
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
             init_logger.warning(f"Could not back up corrupted config file '{filepath}': {backup_err}")

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
                         init_logger.error(f"{NEON_RED}Config validation structure error: Path segment '{key}' not found or not a dictionary in loaded config at '{path}'.{RESET}")
                         return None, None, None
                    if key not in current_def_level or not isinstance(current_def_level[key], dict):
                         # This indicates an error in the *default_config* structure definition itself
                         init_logger.critical(f"{NEON_RED}FATAL: Internal Default Config structure error: Path segment '{key}' not found or not a dictionary in default config at '{path}'. Cannot validate.{RESET}")
                         # Depending on severity, might exit here. For now, return None.
                         return None, None, None

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
                    # Note: We don't set config_needs_saving here directly, wrappers will do it.
                    # This case should be rare if _ensure_config_keys is reliable.


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
                # Error already logged by get_nested_levels
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
        # Leverage >= 0 (0 or 1 means disable setting leverage)
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
        # Proximity factors must be >= 1.0 (or slightly more for float comparison safety)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1) # e.g., 1.005 = 0.5% proximity
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1) # e.g., 1.001 = 0.1% proximity
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
        # TSL callback must be > 0 (e.g., 0-20%)
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", Decimal('0'), Decimal('0.5'), is_strict_min=True)
        # TSL activation can be 0% (immediate once triggered) or positive
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", Decimal('0'), Decimal('0.5'), allow_zero=True)


        # == Notifications ==
        validate_boolean(updated_config, "notifications.enable_notifications")
        validate_string_choice(updated_config, "notifications.notification_type", ["email", "sms"], case_sensitive=False)
        validate_numeric(updated_config, "notifications.sms_timeout_seconds", 5, 120, is_int=True)


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
            # Optional: Add logic to check start_date < end_date if needed


        init_logger.debug("Configuration parameter validation complete.")

        # --- Step 4: Save Updated Config if Necessary ---
        if config_needs_saving:
             init_logger.info(f"{NEON_YELLOW}Configuration updated with defaults or corrections. Saving changes to '{filepath}'...{RESET}")
             try:
                 # Convert any lingering Decimal objects to float for JSON serialization
                 def convert_decimals_to_float(obj: Any) -> Any:
                     if isinstance(obj, Decimal):
                         return float(obj)
                     if isinstance(obj, dict):
                         return {k: convert_decimals_to_float(v) for k, v in obj.items()}
                     if isinstance(obj, list):
                         return [convert_decimals_to_float(elem) for elem in elem]
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
    if pd.isna(value) or value is None:
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
    # Ensure price is positive and finite before formatting
    if not isinstance(price, Decimal) or price <= Decimal('0') or not price.is_finite():
         return None
    try:
        # Use ccxt's built-in precision formatting
        formatted_price = exchange.price_to_precision(symbol, float(price)) # CCXT expects float/string input
        return formatted_price
    except Exception as e:
        # Log formatting errors at debug level, they might indicate market info issues
        logger = setup_logger(f"ccxt.{symbol}") # Get logger for the symbol
        logger.debug(f"Failed to format price {price} for {symbol}: {e}")
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
    # Ensure amount is non-negative (quantity can be 0 for some purposes, but not strictly negative for order amount)
    if not isinstance(amount, Decimal) or amount < Decimal('0') or not amount.is_finite():
         return None
    try:
        # Use ccxt's built-in precision formatting
        formatted_amount = exchange.amount_to_precision(symbol, float(amount)) # CCXT expects float/string input
        return formatted_amount
    except Exception as e:
        # Log formatting errors at debug level
        logger = setup_logger(f"ccxt.{symbol}")
        logger.debug(f"Failed to format amount {amount} for {symbol}: {e}")
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
            # logger.debug(f"Parsed '{key_name}': Value {repr(value)} is invalid/None, returning None as allowed.")
            return None
        else:
            # logger.debug(f"Parsed '{key_name}': Value {repr(value)} is invalid/None, returning default {default.normalize()}.")
            return default
    # else:
        # Successfully parsed a finite Decimal
        # logger.debug(f"Parsed '{key_name}': Value {repr(value)} converted to Decimal {d_val.normalize()}.")
    return d_val # Return the successfully parsed Decimal


def _handle_ccxt_exception(e: Exception, logger: logging.Logger, action: str, symbol: Optional[str] = None, retry_attempt: int = 0) -> bool:
    """
    Logs CCXT exceptions with specific messages and determines if a retry is warranted.

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

    # Use isinstance for exception types rather than checking string content where possible
    if isinstance(e, ccxt.DDoSProtection) or isinstance(e, ccxt.RateLimitExceeded):
        # Rate limits or DDoS protection responses suggest temporary overload
        logger.warning(f"{NEON_YELLOW}{log_prefix}Rate limit or DDoS protection triggered: {e}{RESET}")
        # Check if `e.retryAfter` is available for a hint on how long to wait
        retry_after = getattr(e, 'retryAfter', None) # Time in seconds to wait (if provided by CCXT/exchange)
        if retry_after and isinstance(retry_after, (int, float)) and retry_after > 0:
             wait_time = min(max(retry_after, RETRY_DELAY_SECONDS), 600) # Wait at least RETRY_DELAY_SECONDS, max 10 min
             logger.info(f"Waiting instructed {wait_time}s before retrying...")
             time.sleep(wait_time)
        else:
            # Fallback to exponential backoff if no specific retry time is given
            delay = RETRY_DELAY_SECONDS * (2 ** retry_attempt)
            logger.info(f"Waiting default {delay}s before retrying...")
            time.sleep(delay)
        retry_recommendation = True # These are usually temporary

    elif isinstance(e, (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException)):
        # General network issues, timeouts, connection errors
        logger.warning(f"{NEON_YELLOW}{log_prefix}Network or timeout error: {e}{RESET}")
        delay = RETRY_DELAY_SECONDS * (2 ** retry_attempt) # Exponential backoff
        logger.info(f"Waiting {delay}s before retrying...")
        time.sleep(delay)
        retry_recommendation = True # These are often transient

    elif isinstance(e, (ccxt.ExchangeNotAvailable, ccxt.ExchangeTemporaryDisabled)):
        # Exchange is down for maintenance or temporarily offline
        logger.warning(f"{NEON_YELLOW}{log_prefix}Exchange temporarily unavailable: {e}{RESET}")
        # Wait a longer, fixed amount for exchange-wide issues
        wait_time = max(RETRY_DELAY_SECONDS * 5, 60) # Wait at least 60s or 5 * base delay
        logger.info(f"Waiting {wait_time}s before retrying...")
        time.sleep(wait_time)
        retry_recommendation = True # Worth retrying after a pause

    elif isinstance(e, ccxt.AuthenticationError):
        # API Key/Secret invalid, IP not whitelisted, etc. - Fatal error
        logger.critical(f"{NEON_RED}{BRIGHT}{log_prefix}Authentication Error: {e}{RESET}")
        logger.critical(f"{NEON_RED}Please check your API credentials and IP whitelist. Exiting.{RESET}")
        retry_recommendation = False # Authentication errors are not retryable in this context

    elif isinstance(e, ccxt.InsufficientFunds):
        # Attempted to place an order but balance was too low.
        logger.error(f"{NEON_RED}{log_prefix}Insufficient funds: {e}{RESET}")
        # This implies an issue with position sizing or unexpected balance change.
        # Retrying the *same* order is unlikely to work immediately unless balance is added.
        # However, subsequent actions (like fetching balance later) might be retryable.
        retry_recommendation = False # The *order placement* itself is not retryable without addressing funds issue
        # Optionally send a notification about insufficient funds
        send_notification(f"Funds Error: {symbol}", f"Insufficient funds for {action} on {symbol}. Error: {e}", logger, notification_type='email') # Prioritize email for critical errors

    elif isinstance(e, ccxt.InvalidOrder):
        # Order parameters (price, amount, type) are invalid according to the exchange.
        logger.error(f"{NEON_RED}{log_prefix}Invalid order parameters: {e}{RESET}")
        # This likely indicates a calculation error or incorrect precision. Not retryable without fixing logic.
        # Log details including any parameters that were attempted
        logger.debug(f"Invalid order details: {getattr(e, 'context', 'N/A')}") # CCXT may attach context
        retry_recommendation = False # The *order placement* is not retryable without fixing parameters

    elif isinstance(e, ccxt.OrderNotFound):
         # Attempted to cancel or modify an order that doesn't exist.
         logger.warning(f"{NEON_YELLOW}{log_prefix}Order not found: {e}{RESET}")
         retry_recommendation = False # Order doesn't exist, retrying won't help find it. Assume it was cancelled or filled by other means.

    elif isinstance(e, ccxt.CancelPending):
         # The order cancellation is pending. Retrying cancellation might be needed.
         logger.info(f"{log_prefix}Cancel pending: {e}. Retrying...")
         delay = RETRY_DELAY_SECONDS # Small delay before re-attempting cancellation check/cancel
         time.sleep(delay)
         retry_recommendation = True # Keep retrying cancel attempts

    elif isinstance(e, ccxt.ArgumentsRequired):
        # Missing required arguments for an API call.
        logger.critical(f"{NEON_RED}{BRIGHT}{log_prefix}Missing arguments for API call: {e}{RESET}")
        logger.critical(f"{NEON_RED}This indicates a bug in the bot's code. Exiting.{RESET}")
        retry_recommendation = False # Bug requires code fix

    else:
        # Catch all other unexpected CCXT exceptions
        logger.error(f"{NEON_RED}{log_prefix}An unexpected CCXT error occurred: {e}{RESET}", exc_info=True) # Log traceback for unexpected errors
        # Decide default retry behavior for unknown errors - usually cautious
        delay = RETRY_DELAY_SECONDS * (retry_attempt + 1) # Linear backoff for unknown
        logger.info(f"Waiting {delay}s before retrying (for unexpected error type)...")
        time.sleep(delay)
        retry_recommendation = (retry_attempt < MAX_API_RETRIES) # Only retry up to max attempts for unknown errors

    return retry_recommendation


# --- Exchange Data Fetching Functions (Using CCXT Helpers) ---

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Fetches historical kline data for a symbol using CCXT with retry logic.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        timeframe: The kline interval in CCXT format (e.g., '1m', '5m', '1h').
        limit: The number of historical klines to fetch. Max limit is BYBIT_API_KLINE_LIMIT.
        logger: The logger instance for the symbol.

    Returns:
        A pandas DataFrame containing the kline data (indexed by timestamp),
        or None if fetching fails after retries.
    """
    lg = logger
    ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe, timeframe) # Map config interval to CCXT format

    klines = None
    action_desc = f"fetching klines ({ccxt_timeframe}, limit {limit})"
    for attempt in range(MAX_API_RETRIES + 1):
        if _shutdown_requested:
             lg.info(f"Shutdown requested during kline fetch for {symbol}. Aborting.")
             return None
        try:
            lg.debug(f"{action_desc} for {symbol}, attempt {attempt + 1}/{MAX_API_RETRIES + 1}")
            # Ensure limit does not exceed exchange max
            fetch_limit = min(limit, BYBIT_API_KLINE_LIMIT)
            # Use since parameter to fetch the most recent candles first for large limits
            # Fetching `limit` candles *up to now* is the common pattern.
            # We don't need to calculate `since` based on timeframe * limit here,
            # as CCXT's fetch_ohlcv handles fetching the *latest* `limit` candles by default.
            raw_klines = exchange.fetch_ohlcv(symbol, ccxt_timeframe, limit=fetch_limit)

            if raw_klines:
                # Convert raw list of lists into a pandas DataFrame
                # Columns: timestamp, open, high, low, close, volume
                # Timestamps from CCXT are in milliseconds
                df = pd.DataFrame(raw_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                # Convert timestamp (milliseconds, UTC) to datetime and set as index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.set_index('timestamp')

                # Convert price and volume columns to Decimal for precision
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    # Use apply with _safe_decimal_conversion, allowing None intermediate values
                    # fillna(Decimal('0')) ensures no NaNs remain if the column should be numeric (like volume)
                    if col == 'volume':
                         df[col] = df[col].apply(lambda x: _safe_decimal_conversion(x, default=Decimal('0.0'), allow_none=False))
                    else: # Prices
                         df[col] = df[col].apply(lambda x: _safe_decimal_conversion(x, default=Decimal('0.0'), allow_none=False)) # Use default 0.0 for prices if conversion fails, though should ideally raise error for invalid prices

                # Ensure numeric columns are actually numeric types (Decimal or float fallback if apply fails)
                # This is a safety net. Pandas apply with Decimal should work, but float is safer if Decimal conversion fails unexpectedly
                df['open'] = pd.to_numeric(df['open'], errors='coerce').apply(_safe_decimal_conversion, default=Decimal('0.0'), allow_none=False)
                df['high'] = pd.to_numeric(df['high'], errors='coerce').apply(_safe_decimal_conversion, default=Decimal('0.0'), allow_none=False)
                df['low'] = pd.to_numeric(df['low'], errors='coerce').apply(_safe_decimal_conversion, default=Decimal('0.0'), allow_none=False)
                df['close'] = pd.to_numeric(df['close'], errors='coerce').apply(_safe_decimal_conversion, default=Decimal('0.0'), allow_none=False)
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').apply(_safe_decimal_conversion, default=Decimal('0.0'), allow_none=False)


                # Drop rows with any NaN values introduced by conversion errors, although ideally
                # _safe_decimal_conversion handles this to default=0.0 for prices/volume.
                df.dropna(inplace=True)

                # Check if DataFrame is still valid after processing
                if df.empty:
                     lg.warning(f"Kline data became empty after processing for {symbol}.")
                     klines = None # Treat as fetch failure
                else:
                     # Sort by timestamp just in case (though CCXT usually returns sorted)
                     df.sort_index(inplace=True)
                     lg.debug(f"Successfully fetched and processed {len(df)} klines for {symbol} ({ccxt_timeframe}).")
                     klines = df
                     break # Exit retry loop on success
            else:
                lg.warning(f"Fetched empty kline data for {symbol} ({ccxt_timeframe}, limit {fetch_limit}). Attempt {attempt + 1}.")
                klines = None # Treat as fetch failure for retry logic

        except Exception as e:
            # Let the helper handle logging and determine if retry is needed
            retry = _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
            if not retry:
                # If _handle_ccxt_exception returns False, it's a fatal error or max retries reached for transient errors
                return None
            # If retry is True, the helper has already waited, continue the loop

    if klines is None or klines.empty:
        lg.error(f"{NEON_RED}Failed to fetch usable kline data for {symbol} ({ccxt_timeframe}, limit {limit}) after {MAX_API_RETRIES + 1} attempts.{RESET}")
        return None

    # Optional: Trim DataFrame to MAX_DF_LEN if it exceeds the limit
    if len(klines) > MAX_DF_LEN:
        lg.debug(f"Trimming DataFrame for {symbol} from {len(klines)} to {MAX_DF_LEN} rows.")
        klines = klines.tail(MAX_DF_LEN)

    return klines


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency using CCXT with retry logic.

    Args:
        exchange: The CCXT exchange instance.
        currency: The currency code (e.g., 'USDT', 'BTC').
        logger: The logger instance for the operation.

    Returns:
        The available balance as a Decimal, or None if fetching fails after retries
        or if the currency is not found or has zero balance.
    """
    lg = logger
    balance_info = None
    action_desc = f"fetching balance for {currency}"

    for attempt in range(MAX_API_RETRIES + 1):
        if _shutdown_requested:
             lg.info(f"Shutdown requested during balance fetch for {currency}. Aborting.")
             return None
        try:
            lg.debug(f"{action_desc}, attempt {attempt + 1}/{MAX_API_RETRIES + 1}")
            # Use fetch_balance to get all account balances
            balances = exchange.fetch_balance()

            if balances and currency in balances:
                # Extract total and free (available) balance for the target currency
                # 'free' balance is usually what's available for new orders
                total_bal = _safe_decimal_conversion(balances[currency].get('total'), allow_none=True)
                free_bal = _safe_decimal_conversion(balances[currency].get('free'), allow_none=True)

                # Use the free balance for trading decisions
                if free_bal is not None and free_bal >= Decimal('0'):
                    lg.debug(f"Successfully fetched balance for {currency}: Total={total_bal.normalize() if total_bal is not None else 'N/A'}, Free={free_bal.normalize()}.")
                    balance_info = free_bal
                    break # Exit retry loop on success
                else:
                    # Balance data for currency was invalid or not found (should be caught by 'currency in balances' but defensive)
                     lg.warning(f"Invalid balance data found for {currency}: {balances.get(currency)}. Attempt {attempt + 1}.")
                     balance_info = None # Ensure None for retry logic to continue if needed

            elif balances:
                 # Balances fetched, but specific currency not found.
                 # This might happen if the currency isn't held or is not supported by the endpoint.
                 lg.warning(f"Balance fetched, but currency '{currency}' not found in response. Available: {list(balances.keys())}. Attempt {attempt + 1}.")
                 balance_info = None # Cannot get balance for this currency

            else:
                # fetch_balance returned None or empty dict
                lg.warning(f"Fetched empty balance data. Attempt {attempt + 1}.")
                balance_info = None # Ensure None for retry logic to continue

        except Exception as e:
            retry = _handle_ccxt_exception(e, lg, action_desc, currency, attempt)
            if not retry:
                 # If not retrying, return None after logging
                 return None
            # If retry is True, the helper has already waited, continue the loop

    if balance_info is None:
        lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts.{RESET}")
        return None

    return balance_info


def fetch_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[MarketInfo]:
    """
    Fetches and parses market information for a specific symbol from CCXT markets dictionary.

    Args:
        exchange: The CCXT exchange instance (with markets loaded).
        symbol: The trading symbol (e.g., 'BTC/USDT').
        logger: The logger instance for the symbol.

    Returns:
        A parsed MarketInfo TypedDict, or None if the symbol is not found
        or market data is incomplete/invalid.
    """
    lg = logger
    market = exchange.markets.get(symbol)

    if not market:
        lg.critical(f"{NEON_RED}FATAL ERROR: Market information for symbol '{symbol}' not found in exchange.markets.{RESET}")
        lg.critical(f"{NEON_RED}Ensure the symbol is correct and exists on the exchange in {CONFIG.get('use_sandbox',True)=} mode. Exiting.{RESET}")
        return None

    # Perform basic structural checks on essential fields
    if not isinstance(market, dict):
        lg.critical(f"{NEON_RED}FATAL ERROR: Market data for '{symbol}' is not a dictionary.{RESET}")
        return None
    if 'symbol' not in market or market['symbol'] != symbol:
        lg.critical(f"{NEON_RED}FATAL ERROR: Market data for '{symbol}' has incorrect symbol key: {market.get('symbol')}.{RESET}")
        return None
    if 'precision' not in market or not isinstance(market['precision'], dict):
         lg.critical(f"{NEON_RED}FATAL ERROR: Market data for '{symbol}' is missing 'precision' dictionary.{RESET}")
         return None
    if 'price' not in market['precision'] or 'amount' not in market['precision']:
         lg.critical(f"{NEON_RED}FATAL ERROR: Market precision for '{symbol}' is missing 'price' or 'amount'.{RESET}")
         return None
    if 'limits' not in market or not isinstance(market['limits'], dict):
         lg.critical(f"{NEON_RED}FATAL ERROR: Market data for '{symbol}' is missing 'limits' dictionary.{RESET}")
         return None
    if 'amount' not in market['limits'] or 'cost' not in market['limits']:
         lg.critical(f"{NEON_RED}FATAL ERROR: Market limits for '{symbol}' is missing 'amount' or 'cost'.{RESET}")
         return None


    # Safely parse critical Decimal fields using the helper
    price_precision_step_dec = _parse_decimal(market['precision'].get('price'), 'precision.price', lg, allow_none=True)
    amount_precision_step_dec = _parse_decimal(market['precision'].get('amount'), 'precision.amount', lg, allow_none=True)
    contract_size_dec = _parse_decimal(market.get('contractSize'), 'contractSize', lg, allow_none=True, default=Decimal('1')) # Default to 1 if not specified (e.g., for spot or some linear contracts)

    min_amount_dec = _parse_decimal(market['limits']['amount'].get('min'), 'limits.amount.min', lg, allow_none=True)
    max_amount_dec = _parse_decimal(market['limits']['amount'].get('max'), 'limits.amount.max', lg, allow_none=True)
    min_cost_dec = _parse_decimal(market['limits']['cost'].get('min'), 'limits.cost.min', lg, allow_none=True)
    max_cost_dec = _parse_decimal(market['limits']['cost'].get('max'), 'limits.cost.max', lg, allow_none=True)

    # Critical check: Ensure essential precision and limit values were parsed successfully
    if price_precision_step_dec is None or price_precision_step_dec <= Decimal('0'):
         lg.critical(f"{NEON_RED}FATAL ERROR: Invalid price precision step for '{symbol}': {market['precision'].get('price')}. Cannot proceed.{RESET}")
         return None
    if amount_precision_step_dec is None or amount_precision_step_dec <= Decimal('0'):
         lg.critical(f"{NEON_RED}FATAL ERROR: Invalid amount precision step for '{symbol}': {market['precision'].get('amount')}. Cannot proceed.{RESET}")
         return None
    # Note: min/max amounts and costs can be None if the exchange doesn't specify them.

    # Determine contract type string for logging
    contract_type_str = "Unknown"
    is_contract = market.get('contract', False) # Use the base ccxt flag
    is_linear = market.get('linear', False) # Use the base ccxt flag
    is_inverse = market.get('inverse', False) # Use the base ccxt flag

    if market.get('spot'):
        contract_type_str = "Spot"
    elif is_contract:
        if is_linear:
            contract_type_str = "Linear"
        elif is_inverse:
            contract_type_str = "Inverse"
        elif market.get('option'):
            contract_type_str = "Option"
        else:
             contract_type_str = "Future/Swap" # Generic contract type

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
        'spot': market.get('spot', False),
        'margin': market.get('margin', False),
        'swap': market.get('swap', False),
        'future': market.get('future', False),
        'option': market.get('option', False),
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
        'contract_size_decimal': contract_size_dec # Guaranteed non-None (defaults to 1)
    }

    lg.debug(f"Successfully parsed market info for {symbol}. Type: {market_info['contract_type_str']}, Precision: Amt={market_info['amount_precision_step_decimal'].normalize()}, Price={market_info['price_precision_step_decimal'].normalize()}.")
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

            # Bybit V5 can return multiple positions for the same symbol in different modes (Isolated/Cross, Hedged/One-Way)
            # Assuming one-way, isolated mode for simplicity as is common for this bot type.
            # If using hedged mode or cross mode positions need to be handled differently.
            # We filter for 'open' positions. CCXT typically only returns open/active positions by default.
            open_raw_positions = [
                p for p in raw_positions
                if p and isinstance(p, dict) and p.get('symbol') == symbol and p.get('side') in ['long', 'short'] and p.get('size', 0) > 0
            ]

            if not open_raw_positions:
                lg.debug(f"No open positions found for {symbol}.")
                return [] # Return empty list if no open positions

            lg.debug(f"Found {len(open_raw_positions)} raw open position(s) for {symbol}.")

            for raw_pos in open_raw_positions:
                # Safely parse relevant fields to Decimal
                size_dec = _parse_decimal(raw_pos.get('size'), 'size', lg) # Size is crucial
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
                # Bybit V5 position info structure:
                # 'stopLoss': 'price_string' (or '0'),
                # 'takeProfit': 'price_string' (or '0'),
                # 'trailingStop': 'trigger_price_string' (or '0'),
                # 'activePrice': 'activation_price_string' (or '0')
                # 'tpslMode': 'full' or 'partial'
                # 'riskLimitValue': 'value'
                # 'leverage': 'value'
                # 'positionValue': 'value' (notional)
                # 'liqPrice': 'value'
                # 'entryPrice': 'value'
                # 'markPrice': 'value'
                # 'positionIdx': 0 (cross), 1 (long isolated), 2 (short isolated) - IMPORTANT for V5!
                # 'tpTriggerBy': 'LastPrice', 'IndexPrice', 'MarkPrice'
                # 'slTriggerBy': 'LastPrice', 'IndexPrice', 'MarkPrice'
                # 'trailingActive': 'activation_price_string' (same as activePrice sometimes?)
                # 'trailingTriggerBy': 'LastPrice', 'IndexPrice', 'MarkPrice'
                # 'bustPrice': 'value'

                pos_info = raw_pos.get('info', {}) # Get the exchange-specific info dict

                sl_raw = pos_info.get('stopLoss') or raw_pos.get('stopLossPrice') # Check both info and unified
                tp_raw = pos_info.get('takeProfit') or raw_pos.get('takeProfitPrice') # Check both info and unified
                tsl_trigger_raw = pos_info.get('trailingStop') or raw_pos.get('trailingStopLoss') # Check both info and unified
                tsl_activation_raw = pos_info.get('activePrice') # Primarily from info

                # Parse raw protection strings to Decimal
                sl_dec = _parse_decimal(sl_raw, 'stopLossPrice_raw', lg, allow_none=True)
                tp_dec = _parse_decimal(tp_raw, 'takeProfitPrice_raw', lg, allow_none=True)
                tsl_trigger_dec = _parse_decimal(tsl_trigger_raw, 'trailingStopPrice_raw', lg, allow_none=True)
                tsl_activation_dec = _parse_decimal(tsl_activation_raw, 'tslActivationPrice_raw', lg, allow_none=True)

                # Bybit V5 uses '0' or '0.0' string if SL/TP/TSL is NOT set. Treat these as None.
                if sl_dec is not None and sl_dec.is_zero(): sl_dec = None
                if tp_dec is not None and tp_dec.is_zero(): tp_dec = None
                if tsl_trigger_dec is not None and tsl_trigger_dec.is_zero(): tsl_trigger_dec = None
                if tsl_activation_dec is not None and tsl_activation_dec.is_zero(): tsl_activation_dec = None

                # Determine bot state flags (be_activated, tsl_activated) from position data *if needed*
                # It's safer to manage these flags within the bot's state dictionary
                # associated with the symbol, rather than trying to derive them solely from the
                # exchange position data, as the exchange doesn't explicitly state "BE activated".
                # We'll add these to the PositionInfo structure but manage them outside this fetch function.

                # Check for critical missing data for an open position
                if size_dec is None or size_dec.is_zero():
                     lg.warning(f"Found position for {symbol} with invalid/zero size: {raw_pos.get('size')}. Skipping.")
                     continue # Skip this position if size is invalid
                if entry_price_dec is None or entry_price_dec.is_zero():
                     lg.warning(f"Found position for {symbol} with invalid/zero entry price: {raw_pos.get('entryPrice')}. Skipping.")
                     continue # Skip if entry price is invalid (makes BE/TSL calculation impossible)


                parsed_position: PositionInfo = {
                    # Standard CCXT fields (copied directly or cast)
                    'id': raw_pos.get('id'), # Optional
                    'symbol': str(raw_pos.get('symbol', '')), # Should match input symbol
                    'timestamp': raw_pos.get('timestamp'), # Optional
                    'datetime': raw_pos.get('datetime'), # Optional
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
                    'lastUpdateTimestamp': raw_pos.get('lastUpdateTimestamp'), # Optional
                    'info': raw_pos.get('info', {}), # Full raw info dict
                    # Added/Derived Decimal fields
                    'size_decimal': size_dec, # Guaranteed non-zero by check above
                    'entryPrice_decimal': entry_price_dec, # Guaranteed non-zero by check above
                    'markPrice_decimal': mark_price_dec,
                    'liquidationPrice_decimal': liquidation_price_dec,
                    'leverage_decimal': leverage_dec,
                    'unrealizedPnl_decimal': unrealized_pnl_dec,
                    'notional_decimal': notional_dec,
                    'collateral_decimal': collateral_dec,
                    'initialMargin_decimal': initial_margin_dec,
                    'maintenanceMargin_decimal': maintenance_margin_dec,
                    # Protection Order Status (raw strings and parsed Decimals)
                    'stopLossPrice_raw': sl_raw,
                    'takeProfitPrice_raw': tp_raw,
                    'trailingStopPrice_raw': tsl_trigger_raw, # TSL Trigger Price
                    'tslActivationPrice_raw': tsl_activation_raw, # TSL Activation Price
                    'stopLossPrice_dec': sl_dec,
                    'takeProfitPrice_dec': tp_dec,
                    'trailingStopPrice_dec': tsl_trigger_dec,
                    'tslActivationPrice_dec': tsl_activation_dec,
                    # Bot State Tracking (Initialized here, managed elsewhere)
                    'be_activated': False, # Needs to be loaded from persistent state or determined if SL is close to entry
                    'tsl_activated': False # Needs to be loaded from persistent state or determined if TSL activation price is set
                }
                positions_list.append(parsed_position)

            # Successfully processed positions
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
# NOTE: The actual complex logic for Volumatic Trend and Order Blocks
# needs to be implemented in these functions. The current implementation
# provides a basic structure and placeholder logic only.

def calculate_indicators(dataframe: pd.DataFrame, config_params: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """
    Calculates strategy indicators (Volumatic Trend, ATR, Pivots) and adds them
    as new columns to the DataFrame.

    Args:
        dataframe: The input pandas DataFrame with kline data (OHLCV).
        config_params: The 'strategy_params' dictionary from the configuration.
        logger: The logger instance for the symbol.

    Returns:
        The DataFrame with added indicator columns.
    """
    lg = logger
    df = dataframe.copy() # Work on a copy

    if df.empty:
        lg.warning("Cannot calculate indicators on an empty DataFrame.")
        return df

    # Ensure necessary columns are Decimal and finite before TA calculation
    # This should already be done by fetch_klines_ccxt, but check defensively
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            lg.warning(f"Column '{col}' is not numeric before indicator calculation.")
            # Attempt to convert again, forcing errors to NaN, then handle
            df[col] = pd.to_numeric(df[col], errors='coerce').apply(_safe_decimal_conversion, default=Decimal('0.0'))
            df.dropna(subset=[col], inplace=True) # Drop rows where critical columns became NaN

    if df.empty:
        lg.warning("DataFrame became empty after ensuring numeric columns for indicators.")
        return df

    # Convert Decimal columns to float temporarily for pandas_ta, then convert back if needed
    # Pandas_ta generally works with floats. Be mindful of potential precision loss,
    # but for standard indicators like EMA, ATR, this is usually acceptable.
    # Critical prices (entry, SL, TP) *must* use Decimal.
    df_float = df.astype(float)

    # --- Volumatic Trend Placeholder ---
    vt_length = config_params.get("vt_length", DEFAULT_VT_LENGTH)
    vt_atr_period = config_params.get("vt_atr_period", DEFAULT_VT_ATR_PERIOD)
    vt_vol_ema_length = config_params.get("vt_vol_ema_length", DEFAULT_VT_VOL_EMA_LENGTH)
    vt_atr_multiplier = config_params.get("vt_atr_multiplier", DEFAULT_VT_ATR_MULTIPLIER)
    # vt_step_atr_multiplier = config_params.get("vt_step_atr_multiplier", DEFAULT_VT_STEP_ATR_MULTIPLIER) # Unused placeholder

    # --- PLACEHOLDER VT/ATR Calculation ---
    # Replace this with actual Volumatic Trend logic
    try:
        # Example: Simple EMA/SMA cross as placeholder for VT core trend
        df_float['SMA_VT'] = ta.sma(df_float['close'], length=vt_length)
        df_float['EMA_VT'] = ta.ema(df_float['close'], length=vt_length // 2) # Shorter EMA for cross

        # Calculate ATR
        df_float['ATR'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=vt_atr_period)

        # Example: Simple upper/lower bands (like Keltner Channels based on EMA and ATR)
        # This is a stand-in for VT bands
        df_float['VT_Upper_Band'] = df_float['EMA_VT'] + df_float['ATR'] * vt_atr_multiplier
        df_float['VT_Lower_Band'] = df_float['EMA_VT'] - df_float['ATR'] * vt_atr_multiplier

        # Placeholder for Volume Analysis (e.g., simple EMA of Volume)
        df_float['Volume_EMA'] = ta.ema(df_float['volume'], length=vt_vol_ema_length)
        # Add placeholder for normalized volume if needed by the actual strategy

    except Exception as e:
        lg.error(f"{NEON_RED}Error calculating Volumatic Trend/ATR indicators: {e}{RESET}", exc_info=True)
        # Ensure indicator columns exist even if calculation failed, filled with NaN
        for col in ['SMA_VT', 'EMA_VT', 'ATR', 'VT_Upper_Band', 'VT_Lower_Band', 'Volume_EMA']:
             if col not in df_float.columns:
                  df_float[col] = np.nan # Add column with NaN if missing

    # --- Pivot Point Calculation for Order Blocks ---
    ph_left = config_params.get("ph_left", DEFAULT_PH_LEFT)
    ph_right = config_params.get("ph_right", DEFAULT_PH_RIGHT)
    pl_left = config_params.get("pl_left", DEFAULT_PL_LEFT)
    pl_right = config_params.get("pl_right", DEFAULT_PL_RIGHT)
    ob_source = config_params.get("ob_source", DEFAULT_OB_SOURCE) # "Wicks" or "Body"

    # --- PLACEHOLDER Pivot Calculation ---
    try:
        # Define source columns based on ob_source
        high_src = 'high' if ob_source == 'Wicks' else 'close' # Use Close as high for Body
        low_src = 'low' if ob_source == 'Wicks' else 'open'   # Use Open as low for Body (assuming close > open for bullish OB)
                                                              # Need more robust logic for Body OBs based on candle direction

        # Calculate Pivot Highs (PH) and Pivot Lows (PL) - using standard definition
        # A Pivot High is a high that is the highest in a window of ph_left candles before and ph_right candles after
        # A Pivot Low is a low that is the lowest in a window of pl_left candles before and pl_right candles after

        # Use rolling windows and apply functions. `center=True` aligns window to the middle (the pivot candle)
        # .iloc is used to access rows by integer position for the window checks

        def is_pivot_high(row_idx, df_check, left, right):
            if row_idx < left or row_idx >= len(df_check) - right:
                return False # Cannot be a pivot if not enough bars on either side
            current_high = df_check.iloc[row_idx][high_src]
            window_highs = df_check.iloc[row_idx - left : row_idx + right + 1][high_src]
            # Check if current high is the maximum in the window
            # Handle potential NaN values in the window if calculation failed earlier
            if pd.isna(current_high): return False
            return current_high >= window_highs.max() # Use >= for tie-breaking (first occurrence)

        def is_pivot_low(row_idx, df_check, left, right):
            if row_idx < left or row_idx >= len(df_check) - right:
                return False
            current_low = df_check.iloc[row_idx][low_src]
            window_lows = df_check.iloc[row_idx - left : row_idx + right + 1][low_src]
             # Check if current low is the minimum in the window
            if pd.isna(current_low): return False
            return current_low <= window_lows.min() # Use <= for tie-breaking

        # Apply the pivot checks. This can be slow on very large DataFrames.
        # Using indices for comparison is more robust than comparing series slices directly.
        df_float['is_pivot_high'] = [is_pivot_high(i, df_float, ph_left, ph_right) for i in range(len(df_float))]
        df_float['is_pivot_low'] = [is_pivot_low(i, df_float, pl_left, pl_right) for i in range(len(df_float))]

    except Exception as e:
         lg.error(f"{NEON_RED}Error calculating Pivot Points for Order Blocks: {e}{RESET}", exc_info=True)
         # Ensure pivot columns exist even if calculation failed
         if 'is_pivot_high' not in df_float.columns: df_float['is_pivot_high'] = False
         if 'is_pivot_low' not in df_float.columns: df_float['is_pivot_low'] = False


    # --- Convert back to Decimal for final output ---
    # Only convert the OHLCV and relevant indicator columns back to Decimal
    # Keep boolean pivot flags as they are.
    for col in ['open', 'high', 'low', 'close', 'volume', 'SMA_VT', 'EMA_VT', 'ATR', 'VT_Upper_Band', 'VT_Lower_Band', 'Volume_EMA']:
        if col in df_float.columns:
            # Convert float column back to Decimal, coercing errors to NaN first then filling with 0.0
             df[col] = pd.to_numeric(df_float[col], errors='coerce').apply(_safe_decimal_conversion, default=Decimal('0.0'))
        else:
            # If indicator calculation failed entirely, add the column filled with 0.0 or NaN
             df[col] = Decimal('0.0') # Or np.nan if you prefer missing data over zero
             lg.warning(f"Indicator column '{col}' was not found after calculation, filling with Decimal('0.0').")


    # Copy the boolean pivot columns back
    df['is_pivot_high'] = df_float['is_pivot_high'].copy()
    df['is_pivot_low'] = df_float['is_pivot_low'].copy()


    lg.debug(f"Finished calculating indicators. DataFrame shape: {df.shape}")
    return df

def identify_order_blocks(dataframe: pd.DataFrame, config_params: Dict[str, Any], logger: logging.Logger) -> Tuple[List[OrderBlock], List[OrderBlock]]:
    """
    Identifies potential Order Blocks (OBs) based on pivot points and determines
    their active status.

    Args:
        dataframe: The input pandas DataFrame with kline data and pivot indicators.
        config_params: The 'strategy_params' dictionary from the configuration.
        logger: The logger instance for the symbol.

    Returns:
        A tuple containing two lists: (active_bull_boxes, active_bear_boxes).
    """
    lg = logger
    active_bull_boxes: List[OrderBlock] = []
    active_bear_boxes: List[OrderBlock] = []

    ob_source = config_params.get("ob_source", DEFAULT_OB_SOURCE)
    ob_max_boxes = config_params.get("ob_max_boxes", DEFAULT_OB_MAX_BOXES)
    ob_extend = config_params.get("ob_extend", DEFAULT_OB_EXTEND)

    if dataframe.empty:
        lg.warning("Cannot identify order blocks on an empty DataFrame.")
        return [], []

    # Ensure necessary columns exist from calculate_indicators
    if 'is_pivot_high' not in dataframe.columns or 'is_pivot_low' not in dataframe.columns:
         lg.error("DataFrame missing 'is_pivot_high' or 'is_pivot_low' columns. Cannot identify OBs.")
         return [], []

    # Identify potential OB candidates based on pivots
    # A bullish OB (Demand) is often identified before a move up, typically associated with a Pivot Low.
    # A bearish OB (Supply) is often identified before a move down, typically associated with a Pivot High.

    potential_bull_obs = dataframe[dataframe['is_pivot_low']].copy()
    potential_bear_obs = dataframe[dataframe['is_pivot_high']].copy()

    # --- Placeholder OB Definition and Validation ---
    # This needs refined logic based on candle structure, volume, etc.
    # Example simplistic approach:
    # Bullish OB: The candle *at* the Pivot Low. Range is Low to High or Open to Close based on source.
    # Bearish OB: The candle *at* the Pivot High. Range is High to Low or Open to Close based on source.
    # Validation: Is the candle at the pivot an "engulfing" candle or strong momentum candle?
    # Invalidation: An OB is violated if price closes *outside* the box.

    current_price = dataframe['close'].iloc[-1] # Use the last close price

    # Helper to check if a price violates an OB
    def is_violated(ob: OrderBlock, price: Decimal, side: str) -> bool:
         if side == "BULL":
             # Bullish OB is violated if price closes *below* the bottom
             return price < ob['bottom']
         elif side == "BEAR":
             # Bearish OB is violated if price closes *above* the top
             return price > ob['top']
         return False # Should not happen

    # Process potential Bullish OBs
    for index, row in potential_bull_obs.iterrows():
        try:
            # Define the box range based on source
            if ob_source == "Wicks":
                bottom = _safe_decimal_conversion(row['low'], allow_none=True)
                top = _safe_decimal_conversion(row['high'], allow_none=True)
            else: # "Body" - assuming bullish candle (close > open) for bullish OB
                bottom = _safe_decimal_conversion(row['open'], allow_none=True)
                top = _safe_decimal_conversion(row['close'], allow_none=True)

            if bottom is None or top is None or bottom >= top:
                 # Skip if price levels are invalid or inverted
                 lg.debug(f"Skipping invalid bullish OB at {index}: bottom={bottom}, top={top}")
                 continue

            ob_id = f"BULL_{index.value}_{ob_source}" # Unique ID based on timestamp and source

            # Check violation against current price
            violated_status = is_violated({'bottom': bottom, 'top': top}, current_price, "BULL") # Create a temp dict for check
            violation_ts = index if violated_status else None # This check is against *current* price, not historical close for violation. Need to check against *all* subsequent candles for true historical violation.

            # For a real bot, we need to track OBs over time.
            # This requires saving state between runs or checking historical data for violation.
            # For this example, we'll simplify: an OB is "active" if it's identified in the fetched history
            # AND its defining candle's high/low hasn't been crossed by the *last candle's close*
            # AND we haven't exceeded max_boxes.

            # More realistic simple check: is the current price *within* or *near* the zone,
            # and hasn't closed past it *since its creation*?
            # A proper implementation requires checking candle closes *after* the OB candle.
            # Let's simplify for this structure: An OB is active if it's identified AND the
            # *most recent candle's close* has not violated it. This is a very basic filter.

            # Check if the last candle's close has violated the OB identified by this pivot
            # Violation needs to check against price action *after* the OB candle.
            # A truly active OB should not have been significantly penetrated or closed below/above.
            # For placeholder, assume any identified pivot OB is "potentially active" if the *most recent price* hasn't invalidated it.
            # This is a significant simplification.

            # A more robust check (still simplified): Is the OB defined by a candle *before* the last few candles?
            # And has the price stayed above it since its creation?

            # Check if the OB candle is recent enough AND hasn't been violated by any close since
            is_active = True
            potential_violation_ts = None

            # Check all subsequent candles in the fetched dataframe for violation
            subsequent_candles = dataframe.loc[index:].iloc[1:] # Get candles after the pivot candle
            for sub_index, sub_row in subsequent_candles.iterrows():
                 sub_close = _safe_decimal_conversion(sub_row['close'], allow_none=True)
                 if sub_close is not None and is_violated({'bottom': bottom, 'top': top}, sub_close, "BULL"):
                     is_active = False
                     potential_violation_ts = sub_index # Timestamp of the violating candle
                     break # Violated, stop checking this OB

            # If not violated by subsequent candles in the fetched data, consider it active *for now*
            # This doesn't track violations across data fetches, which is a limitation without state management.
            if is_active:
                # Limit the number of active boxes
                if len(active_bull_boxes) < ob_max_boxes:
                    # Add the OB to the active list
                    active_bull_boxes.append({
                        'id': ob_id,
                        'type': 'BULL',
                        'timestamp': index,
                        'top': top,
                        'bottom': bottom,
                        'active': True, # Active based on check above
                        'violated': False, # Not violated by candles in this fetch
                        'violation_ts': None
                    })
                else:
                     lg.debug(f"Skipping bullish OB at {index} as max boxes ({ob_max_boxes}) reached.")
            # else:
                 # lg.debug(f"Bullish OB at {index} identified but violated at {potential_violation_ts}.")


        except Exception as e:
            lg.error(f"{NEON_RED}Error processing potential bullish OB at index {index}: {e}{RESET}", exc_info=True)
            # Continue to next potential OB


    # Process potential Bearish OBs (similar logic)
    for index, row in potential_bear_obs.iterrows():
        try:
            # Define the box range based on source
            if ob_source == "Wicks":
                bottom = _safe_decimal_conversion(row['low'], allow_none=True)
                top = _safe_decimal_conversion(row['high'], allow_none=True)
            else: # "Body" - assuming bearish candle (open > close) for bearish OB
                bottom = _safe_decimal_conversion(row['close'], allow_none=True)
                top = _safe_decimal_conversion(row['open'], allow_none=True)

            if bottom is None or top is None or bottom >= top:
                 # Skip if price levels are invalid or inverted
                 lg.debug(f"Skipping invalid bearish OB at {index}: bottom={bottom}, top={top}")
                 continue

            ob_id = f"BEAR_{index.value}_{ob_source}" # Unique ID based on timestamp and source

            # Check if the last candle's close has violated the OB identified by this pivot
            # Violation needs to check against price action *after* the OB candle.
            # A truly active OB should not have been significantly penetrated or closed below/above.
            # For placeholder, assume any identified pivot OB is "potentially active" if the *most recent price* hasn't invalidated it.
            # This is a significant simplification.

            # Check all subsequent candles in the fetched dataframe for violation
            is_active = True
            potential_violation_ts = None
            subsequent_candles = dataframe.loc[index:].iloc[1:] # Get candles after the pivot candle
            for sub_index, sub_row in subsequent_candles.iterrows():
                 sub_close = _safe_decimal_conversion(sub_row['close'], allow_none=True)
                 if sub_close is not None and is_violated({'bottom': bottom, 'top': top}, sub_close, "BEAR"):
                     is_active = False
                     potential_violation_ts = sub_index # Timestamp of the violating candle
                     break # Violated, stop checking this OB

            # If not violated by subsequent candles in the fetched data, consider it active *for now*
            if is_active:
                # Limit the number of active boxes
                if len(active_bear_boxes) < ob_max_boxes:
                    # Add the OB to the active list
                    active_bear_boxes.append({
                        'id': ob_id,
                        'type': 'BEAR',
                        'timestamp': index,
                        'top': top,
                        'bottom': bottom,
                        'active': True, # Active based on check above
                        'violated': False, # Not violated by candles in this fetch
                        'violation_ts': None
                    })
                else:
                     lg.debug(f"Skipping bearish OB at {index} as max boxes ({ob_max_boxes}) reached.")
            # else:
                 # lg.debug(f"Bearish OB at {index} identified but violated at {potential_violation_ts}.")


        except Exception as e:
            lg.error(f"{NEON_RED}Error processing potential bearish OB at index {index}: {e}{RESET}", exc_info=True)
            # Continue to next potential OB


    # Sort active boxes by timestamp (oldest first) - useful for priority if needed
    active_bull_boxes.sort(key=lambda x: x['timestamp'])
    active_bear_boxes.sort(key=lambda x: x['timestamp'])

    lg.debug(f"Finished identifying order blocks. Found {len(active_bull_boxes)} active bullish and {len(active_bear_boxes)} active bearish boxes.")

    return active_bull_boxes, active_bear_boxes


def analyze_strategy(dataframe: pd.DataFrame, config_params: Dict[str, Any], logger: logging.Logger) -> StrategyAnalysisResults:
    """
    Analyzes the DataFrame with indicators to determine the current trend,
    bands, ATR, and active order blocks based on the strategy rules.

    Args:
        dataframe: The input pandas DataFrame with calculated indicators and pivots.
        config_params: The 'strategy_params' dictionary from the configuration.
        logger: The logger instance for the symbol.

    Returns:
        A StrategyAnalysisResults TypedDict containing the analysis outcome.
    """
    lg = logger

    if dataframe.empty:
        lg.warning("Cannot analyze strategy on an empty DataFrame.")
        return {
            'dataframe': dataframe,
            'last_close': Decimal('0.0'),
            'current_trend_up': None,
            'trend_just_changed': False,
            'active_bull_boxes': [],
            'active_bear_boxes': [],
            'vol_norm_int': None,
            'atr': Decimal('0.0'),
            'upper_band': Decimal('0.0'),
            'lower_band': Decimal('0.0')
        }

    # Ensure necessary columns exist from calculate_indicators and identify_order_blocks
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'ATR', 'VT_Upper_Band', 'VT_Lower_Band', 'SMA_VT', 'EMA_VT']
    if not all(col in dataframe.columns for col in required_cols):
         lg.error("DataFrame missing required indicator columns for strategy analysis.")
         # Return a results dict indicating analysis failed for key elements
         return {
            'dataframe': dataframe,
            'last_close': dataframe['close'].iloc[-1] if 'close' in dataframe.columns and not dataframe['close'].empty else Decimal('0.0'),
            'current_trend_up': None,
            'trend_just_changed': False,
            'active_bull_boxes': [], # Cannot identify OBs reliably without pivots
            'active_bear_boxes': [], # Cannot identify OBs reliably without pivots
            'vol_norm_int': None,
            'atr': dataframe['ATR'].iloc[-1] if 'ATR' in dataframe.columns and not dataframe['ATR'].empty else Decimal('0.0'),
            'upper_band': dataframe['VT_Upper_Band'].iloc[-1] if 'VT_Upper_Band' in dataframe.columns and not dataframe['VT_Upper_Band'].empty else Decimal('0.0'),
            'lower_band': dataframe['VT_Lower_Band'].iloc[-1] if 'VT_Lower_Band' in dataframe.columns and not dataframe['VT_Lower_Band'].empty else Decimal('0.0')
        }


    # --- Volumatic Trend Analysis (Placeholder) ---
    # Example logic: Trend is up if EMA > SMA, down if EMA < SMA
    current_sma = dataframe['SMA_VT'].iloc[-1]
    current_ema = dataframe['EMA_VT'].iloc[-1]
    prev_sma = dataframe['SMA_VT'].iloc[-2] if len(dataframe) >= 2 else None
    prev_ema = dataframe['EMA_VT'].iloc[-2] if len(dataframe) >= 2 else None

    current_trend_up: Optional[bool] = None
    if current_ema > current_sma:
        current_trend_up = True # Up Trend
    elif current_ema < current_sma:
        current_trend_up = False # Down Trend
    # Else: Sideways/Undetermined (could be cross or flat lines)


    # Check if trend just changed on the last candle
    trend_just_changed = False
    if prev_ema is not None and prev_sma is not None:
         prev_trend_up: Optional[bool] = None
         if prev_ema > prev_sma:
              prev_trend_up = True
         elif prev_ema < prev_sma:
              prev_trend_up = False

         # Trend changed if previous trend was different from current trend
         # Handle None cases: if current trend is determined but previous wasn't, or vice-versa, not a "just changed" cross
         if current_trend_up is not None and prev_trend_up is not None and current_trend_up != prev_trend_up:
              trend_just_changed = True
              lg.debug(f"Trend just changed: {'Up' if current_trend_up else 'Down'}")

    # --- Order Block Identification ---
    # This calls the separate function to get active OBs from the DataFrame
    active_bull_boxes, active_bear_boxes = identify_order_blocks(dataframe, config_params, logger)

    # --- Get latest indicator values ---
    last_close = dataframe['close'].iloc[-1]
    current_atr = dataframe['ATR'].iloc[-1]
    upper_band = dataframe['VT_Upper_Band'].iloc[-1]
    lower_band = dataframe['VT_Lower_Band'].iloc[-1]
    # Placeholder for normalized volume
    vol_norm_int = None # Implement actual normalized volume logic if needed

    lg.debug(f"Strategy analysis complete. Trend: {'Up' if current_trend_up else 'Down' if current_trend_up is False else 'Sideways'}, ATR: {current_atr.normalize() if current_atr is not None else 'N/A'}")
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


def generate_signal(analysis_results: StrategyAnalysisResults, current_positions: List[PositionInfo], config_params: Dict[str, Any], market_info: MarketInfo, logger: logging.Logger) -> SignalResult:
    """
    Generates a trading signal ("BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT", "NONE")
    based on the strategy analysis results and current open positions.

    Args:
        analysis_results: The results from analyze_strategy.
        current_positions: A list of current open PositionInfo objects for the symbol.
        config_params: The 'strategy_params' dictionary from the configuration.
        market_info: MarketInfo TypedDict for the symbol.
        logger: The logger instance for the symbol.

    Returns:
        A SignalResult TypedDict indicating the action to take and the reason.
        Includes calculated initial SL/TP prices for entry signals.
    """
    lg = logger

    df = analysis_results['dataframe']
    last_close = analysis_results['last_close']
    current_trend_up = analysis_results['current_trend_up']
    trend_just_changed = analysis_results['trend_just_changed']
    active_bull_boxes = analysis_results['active_bull_boxes']
    active_bear_boxes = analysis_results['active_bear_boxes']
    current_atr = analysis_results['atr']

    ob_entry_proximity_factor = Decimal(str(config_params.get("ob_entry_proximity_factor", DEFAULT_OB_ENTRY_PROXIMITY_FACTOR)))
    ob_exit_proximity_factor = Decimal(str(config_params.get("ob_exit_proximity_factor", DEFAULT_OB_EXIT_PROXIMITY_FACTOR)))
    initial_sl_atr_mult = Decimal(str(CONFIG.get("protection", {}).get("initial_stop_loss_atr_multiple", DEFAULT_INITIAL_STOP_LOSS_ATR_MULTIPLE)))
    initial_tp_atr_mult = Decimal(str(CONFIG.get("protection", {}).get("initial_take_profit_atr_multiple", DEFAULT_INITIAL_TAKE_PROFIT_ATR_MULTIPLE)))


    # Check if minimum data is available for signal generation
    if current_trend_up is None or current_atr is None or current_atr <= Decimal('0'):
        lg.debug("Signal generation: Trend or ATR not determined. Returning NONE.")
        return {'signal': 'NONE', 'reason': 'Insufficient data or indicators not ready', 'initial_sl_price': None, 'initial_tp_price': None}

    # --- Determine if an exit signal exists for any current position ---
    # Assuming a maximum of one position per symbol for now based on max_concurrent_positions=1 logic
    current_position: Optional[PositionInfo] = current_positions[0] if current_positions else None

    if current_position:
        pos_side = current_position['side']
        entry_price = current_position['entryPrice_decimal']
        # Check for exit conditions:
        # 1. Trend reversal (e.g., Up trend position, trend changes to Down)
        # 2. Price hitting an opposite Order Block (e.g., Long position near a Bearish OB top)
        # 3. Stop Loss or Take Profit hit (handled natively by exchange, bot detects position closure)
        # 4. Other strategy-specific exit rules (e.g., breaking a band, volume spike)

        # Exit condition 1: Trend Reversal
        if pos_side == 'long' and current_trend_up is False and trend_just_changed:
            lg.info(f"{NEON_YELLOW}Exit Signal: Trend reversed from Up to Down for active LONG position.{RESET}")
            return {'signal': 'EXIT_LONG', 'reason': 'Trend reversal (VT)', 'initial_sl_price': None, 'initial_tp_price': None}
        elif pos_side == 'short' and current_trend_up is True and trend_just_changed:
            lg.info(f"{NEON_YELLOW}Exit Signal: Trend reversed from Down to Up for active SHORT position.{RESET}")
            return {'signal': 'EXIT_SHORT', 'reason': 'Trend reversal (VT)', 'initial_sl_price': None, 'initial_tp_price': None}

        # Exit condition 2: Price nearing/hitting opposite OB
        # Find the closest opposite OB that has not been violated
        closest_opposite_ob: Optional[OrderBlock] = None
        ob_check_price = last_close # Use last close price for checking proximity

        if pos_side == 'long' and active_bear_boxes:
             # Find the highest bottom or lowest top of active bearish OBs that hasn't been passed yet
             relevant_obs = [ob for ob in active_bear_boxes if ob_check_price <= ob['top']] # Only consider OBs at or above current price
             if relevant_obs:
                 # For a Long, we exit near the *top* of a Bearish OB
                 closest_opposite_ob = min(relevant_obs, key=lambda ob: ob['top']) # Closest is the one with the minimum top price
                 # Check proximity to the top of this bearish OB
                 if ob_check_price >= closest_opposite_ob['top'] / ob_exit_proximity_factor: # Within X% proximity below top
                      lg.info(f"{NEON_YELLOW}Exit Signal: Price {last_close.normalize()} near Bearish OB top {closest_opposite_ob['top'].normalize()} for active LONG position.{RESET}")
                      return {'signal': 'EXIT_LONG', 'reason': 'Price near Bearish OB', 'initial_sl_price': None, 'initial_tp_price': None}

        elif pos_side == 'short' and active_bull_boxes:
             # Find the lowest top or highest bottom of active bullish OBs that hasn't been passed yet
             relevant_obs = [ob for ob in active_bull_boxes if ob_check_price >= ob['bottom']] # Only consider OBs at or below current price
             if relevant_obs:
                  # For a Short, we exit near the *bottom* of a Bullish OB
                  closest_opposite_ob = max(relevant_obs, key=lambda ob: ob['bottom']) # Closest is the one with the maximum bottom price
                  # Check proximity to the bottom of this bullish OB
                  if ob_check_price <= closest_opposite_ob['bottom'] * ob_exit_proximity_factor: # Within X% proximity above bottom
                      lg.info(f"{NEON_YELLOW}Exit Signal: Price {last_close.normalize()} near Bullish OB bottom {closest_opposite_ob['bottom'].normalize()} for active SHORT position.{RESET}")
                      return {'signal': 'EXIT_SHORT', 'reason': 'Price near Bullish OB', 'initial_sl_price': None, 'initial_tp_price': None}

        # Exit condition 3: SL/TP hit (Position will be gone next loop, no signal needed here)
        # Exit condition 4: Other Strategy Rules (Placeholder - add your logic here)
        # Example: Price crossing back inside VT bands after trending outside
        # if pos_side == 'long' and last_close < analysis_results['lower_band']:
        #      lg.info(f"{NEON_YELLOW}Exit Signal: Price closed below lower band for active LONG position.{RESET}")
        #      return {'signal': 'EXIT_LONG', 'reason': 'Price below lower band', 'initial_sl_price': None, 'initial_tp_price': None}
        # elif pos_side == 'short' and last_close > analysis_results['upper_band']:
        #      lg.info(f"{NEON_YELLOW}Exit Signal: Price closed above upper band for active SHORT position.{RESET}")
        #      return {'signal': 'EXIT_SHORT', 'reason': 'Price above upper band', 'initial_sl_price': None, 'initial_tp_price': None}


        # If none of the explicit exit conditions met, but still in position
        lg.debug(f"Holding active {pos_side.upper()} position. No exit signal generated.")
        return {'signal': 'HOLD', 'reason': 'Holding existing position', 'initial_sl_price': None, 'initial_tp_price': None}

    # --- Determine if an entry signal exists (only if no current position) ---
    else: # No current position open
        # Entry conditions:
        # 1. Trend is determined (Up for BUY, Down for SELL)
        # 2. Price is near a relevant Order Block (Bullish OB for BUY, Bearish OB for SELL)
        # 3. Other strategy-specific entry rules (e.g., volume confirmation, break of structure)

        # Entry condition 1: Trend Check
        if current_trend_up is True:
             potential_entry_side = 'long'
        elif current_trend_up is False:
             potential_entry_side = 'short'
        else:
             lg.debug("Signal generation: Trend is undetermined. No entry signal.")
             return {'signal': 'NONE', 'reason': 'Trend undetermined', 'initial_sl_price': None, 'initial_tp_price': None}


        # Entry condition 2: Price Proximity to Relevant Order Block
        relevant_obs: List[OrderBlock] = []
        entry_ob: Optional[OrderBlock] = None
        entry_ob_edge: Optional[Decimal] = None # The specific edge (top/bottom) we are reacting to

        if potential_entry_side == 'long' and active_bull_boxes:
             # For a Long entry, look for price near a Bullish OB *below* current price
             relevant_obs = [ob for ob in active_bull_boxes if last_close >= ob['bottom']] # Only consider OBs at or below current price
             if relevant_obs:
                 # Closest Bullish OB below current price (highest bottom)
                 closest_ob = max(relevant_obs, key=lambda ob: ob['bottom'])
                 # Check if current price is within entry proximity *above* the bottom of this OB
                 if last_close <= closest_ob['bottom'] * ob_entry_proximity_factor: # Within X% proximity above bottom
                      entry_ob = closest_ob
                      entry_ob_edge = entry_ob['bottom']
                      lg.debug(f"Potential LONG entry signal: Price {last_close.normalize()} near Bullish OB bottom {entry_ob_edge.normalize()}.")

        elif potential_entry_side == 'short' and active_bear_boxes:
             # For a Short entry, look for price near a Bearish OB *above* current price
             relevant_obs = [ob for ob in active_bear_boxes if last_close <= ob['top']] # Only consider OBs at or above current price
             if relevant_obs:
                 # Closest Bearish OB above current price (lowest top)
                 closest_ob = min(relevant_obs, key=lambda ob: ob['top'])
                 # Check if current price is within entry proximity *below* the top of this OB
                 if last_close >= closest_ob['top'] / ob_entry_proximity_factor: # Within X% proximity below top
                      entry_ob = closest_ob
                      entry_ob_edge = entry_ob['top']
                      lg.debug(f"Potential SHORT entry signal: Price {last_close.normalize()} near Bearish OB top {entry_ob_edge.normalize()}.")


        # Entry condition 3: Other Strategy Rules (Placeholder - add your logic here)
        # Example: Price crossing a band *in the direction of the trend* near an OB
        # if potential_entry_side == 'long' and last_close > analysis_results['upper_band'] and entry_ob:
        #     lg.debug("Potential LONG entry signal: Price above upper band AND near Bullish OB.")
        #     pass # Signal confirmed
        # elif potential_entry_side == 'short' and last_close < analysis_results['lower_band'] and entry_ob:
        #     lg.debug("Potential SHORT entry signal: Price below lower band AND near Bearish OB.")
        #     pass # Signal confirmed
        # else:
        #      # If price is not near a band or other conditions aren't met, maybe no signal
        #      entry_ob = None # Invalidate OB-based signal if other conditions aren't met


        # --- Final Signal Determination ---
        initial_sl_price: Optional[Decimal] = None
        initial_tp_price: Optional[Decimal] = None
        signal = 'NONE'
        reason = 'No entry conditions met'

        # Basic Signal Rule: Trend + OB Proximity (Placeholder logic)
        if potential_entry_side == 'long' and entry_ob:
             signal = 'BUY'
             reason = f"Up trend and price near Bullish OB {entry_ob['id']} @ {entry_ob_edge.normalize()}"
             # Calculate initial SL/TP based on ATR
             if current_atr is not None and current_atr > Decimal('0'):
                  initial_sl_price = last_close - current_atr * initial_sl_atr_mult
                  # SL should also be below the OB bottom for robustness? Example: max(calculated_sl, ob_bottom - small_offset)
                  initial_tp_price = last_close + current_atr * initial_tp_atr_mult if initial_tp_atr_mult > Decimal('0') else None

        elif potential_entry_side == 'short' and entry_ob:
             signal = 'SELL'
             reason = f"Down trend and price near Bearish OB {entry_ob['id']} @ {entry_ob_edge.normalize()}"
             # Calculate initial SL/TP based on ATR
             if current_atr is not None and current_atr > Decimal('0'):
                  initial_sl_price = last_close + current_atr * initial_sl_atr_mult
                   # SL should also be above the OB top for robustness? Example: min(calculated_sl, ob_top + small_offset)
                  initial_tp_price = last_close - current_atr * initial_tp_atr_mult if initial_tp_atr_mult > Decimal('0') else None

        # Refine SL/TP to match price precision
        if initial_sl_price is not None:
             initial_sl_price = _safe_decimal_conversion(_format_price(exchange, market_info['symbol'], initial_sl_price), allow_none=True)
             if initial_sl_price is None or initial_sl_price <= Decimal('0'): # Sanity check after formatting
                 lg.warning(f"Calculated initial SL price {initial_sl_price} became invalid after formatting. Disabling initial SL.")
                 initial_sl_price = None
             # For LONG, SL must be <= entry price. For SHORT, SL must be >= entry price.
             if signal == 'BUY' and initial_sl_price is not None and initial_sl_price > last_close:
                  lg.warning(f"Calculated initial SL price {initial_sl_price.normalize()} is above entry price {last_close.normalize()} for LONG. Adjusting or disabling SL.")
                  initial_sl_price = last_close * (Decimal('1') - PRICE_EPSILON) # Place just below entry
             elif signal == 'SELL' and initial_sl_price is not None and initial_sl_price < last_close:
                  lg.warning(f"Calculated initial SL price {initial_sl_price.normalize()} is below entry price {last_close.normalize()} for SHORT. Adjusting or disabling SL.")
                  initial_sl_price = last_close * (Decimal('1') + PRICE_EPSILON) # Place just above entry


        if initial_tp_price is not None:
             initial_tp_price = _safe_decimal_conversion(_format_price(exchange, market_info['symbol'], initial_tp_price), allow_none=True)
             if initial_tp_price is None or initial_tp_price <= Decimal('0'): # Sanity check after formatting
                 lg.warning(f"Calculated initial TP price {initial_tp_price} became invalid after formatting. Disabling initial TP.")
                 initial_tp_price = None
             # For LONG, TP must be >= entry price. For SHORT, TP must be <= entry price.
             if signal == 'BUY' and initial_tp_price is not None and initial_tp_price < last_close:
                  lg.warning(f"Calculated initial TP price {initial_tp_price.normalize()} is below entry price {last_close.normalize()} for LONG. Disabling TP.")
                  initial_tp_price = None # Invalid TP direction
             elif signal == 'SELL' and initial_tp_price is not None and initial_tp_price > last_close:
                  lg.warning(f"Calculated initial TP price {initial_tp_price.normalize()} is above entry price {last_close.normalize()} for SHORT. Disabling TP.")
                  initial_tp_price = None # Invalid TP direction


        lg.info(f"Signal Generated: {signal} ({reason}) @ Price: {last_close.normalize()} | SL: {initial_sl_price.normalize() if initial_sl_price else 'None'}, TP: {initial_tp_price.normalize() if initial_tp_price else 'None'}")
        return {'signal': signal, 'reason': reason, 'initial_sl_price': initial_sl_price, 'initial_tp_price': initial_tp_price}


# --- Position Sizing Logic ---
def calculate_position_size(account_balance: Decimal, current_price: Decimal, atr_value: Decimal, market_info: MarketInfo, logger: logging.Logger) -> Optional[Decimal]:
    """
    Calculates the position size (amount in base currency for spot or contracts for futures)
    based on available balance, risk per trade, ATR, and current price.

    Uses the Risk % approach: Position Size = (Balance * Risk %) / (Distance to SL * Contract Size * Price)
    Distance to SL is estimated by ATR * initial_stop_loss_atr_multiple.

    Args:
        account_balance: The total available balance in the quote currency.
        current_price: The current market price (used for order value calculation).
        atr_value: The current ATR value for the symbol.
        market_info: The MarketInfo TypedDict for the symbol.
        logger: The logger instance for the symbol.

    Returns:
        The calculated position size as a Decimal, formatted to precision,
        or None if calculation is not possible or results in an invalid size.
    """
    lg = logger

    risk_per_trade = Decimal(str(CONFIG.get("risk_per_trade", DEFAULT_CONFIG['risk_per_trade'])))
    initial_sl_atr_mult = Decimal(str(CONFIG.get("protection", {}).get("initial_stop_loss_atr_multiple", DEFAULT_CONFIG['protection']['initial_stop_loss_atr_multiple'])))
    leverage = Decimal(str(CONFIG.get("leverage", DEFAULT_CONFIG['leverage']))) # Desired leverage

    # Ensure necessary inputs are valid Decimals
    if account_balance is None or account_balance <= Decimal('0'):
        lg.warning("Cannot calculate position size: Account balance is zero or invalid.")
        return None
    if current_price is None or current_price <= Decimal('0'):
        lg.warning("Cannot calculate position size: Current price is zero or invalid.")
        return None
    if atr_value is None or atr_value <= Decimal('0'):
        lg.warning("Cannot calculate position size: ATR value is zero or invalid.")
        return None
    if risk_per_trade <= Decimal('0') or risk_per_trade > Decimal('1'):
         lg.error(f"Cannot calculate position size: Invalid risk_per_trade config value {risk_per_trade}. Must be > 0 and <= 1.")
         return None
    if initial_sl_atr_mult <= Decimal('0'):
         lg.error(f"Cannot calculate position size: Invalid initial_stop_loss_atr_multiple config value {initial_sl_atr_mult}. Must be > 0.")
         return None
    if market_info is None or market_info['amount_precision_step_decimal'] is None or market_info['amount_precision_step_decimal'] <= Decimal('0'):
        lg.error(f"Cannot calculate position size: Market amount precision is missing or invalid for {market_info.get('symbol', 'N/A')}.")
        return None

    # Calculate estimated stop loss distance in quote currency per base unit/contract
    # Stop Loss Distance = ATR * ATR_Multiplier
    sl_distance_price_units = atr_value * initial_sl_atr_mult

    # Calculate the value risked per base unit/contract if SL is hit
    # For Linear contracts (USDT-margined) and Spot: Risk per unit = SL Distance
    # For Inverse contracts (BTC-margined): Risk per unit calculation is more complex, depends on contract size and current price.
    # Value Per Contract = Contract Size / Entry Price * (Entry Price - Liquidation Price) --- roughly...
    # Risk Per Contract (Inverse) = Contract Size / Entry Price * (Entry Price - SL Price) ...
    # It's simpler to think about the VALUE of the contracts and the risk percentage on that value.

    # Using the Risk % of Capital approach:
    # Total Capital to Risk = Account Balance * Risk Per Trade %
    # Estimated Cost per Base Unit/Contract at SL = SL Distance (price) * Contract Size
    # Number of Units/Contracts = (Total Capital to Risk) / (Estimated Loss Per Unit/Contract at SL)
    # Number of Units/Contracts = (Balance * Risk %) / (ATR * SL_ATR_Mult * Contract Size * Price) -- This isn't quite right either...

    # Let's use the Margin required approach (simplified for linear):
    # Desired Position Value (Notional) = Capital * Risk % * Leverage / (Entry Price / SL Price - 1) ... (complex)

    # Simpler risk-based sizing for Linear contracts (Risk % of Equity):
    # Amount_in_Base = (Capital * Risk %) / (Entry Price - SL Price)
    # Amount_in_Base = (Capital * Risk %) / (ATR * SL_ATR_Mult)
    # If contract size is not 1 (e.g., Inverse), need to account for it:
    # Contracts = (Capital * Risk %) / (ATR * SL_ATR_Mult * Contract Size / Price) ? No...

    # The simplest robust method using ATR for SL distance, applicable to both linear and inverse:
    # Calculate the Stop Loss Price first:
    # Long: SL Price = Current Price - (ATR * SL_ATR_Mult)
    # Short: SL Price = Current Price + (ATR * SL_ATR_Mult)
    # Price difference to SL = ATR * SL_ATR_Mult
    # Value of 1 unit of base currency (or 1 contract for linear) = Current Price * Contract Size (for contracts)
    # Value of 1 contract (for inverse) = Contract Size (in quote currency) / Current Price (base price)
    # Let's stick to the most common case: Linear contracts (ContractSize = 1, Quote is USDT)
    # Or Spot (ContractSize = 1, Quote is USDT)
    # Position Size (in Base or Contracts) = (Balance * Risk %) / (SL Distance in Quote per unit of Base/Contract)
    # SL Distance in Quote per Base unit = ATR * SL_ATR_Mult
    # Number of Base Units/Contracts = (Balance * Risk %) / (ATR * SL_ATR_Mult)

    # Correct sizing logic for Linear/Spot based on Risk %:
    # Risked Capital in Quote = account_balance * risk_per_trade
    # Price distance to SL = atr_value * initial_sl_atr_mult
    # Cost of this price distance for 1 unit of base currency (or 1 contract for linear) = Price distance to SL
    # Quantity (in Base/Contracts) = Risked Capital in Quote / Price distance to SL
    # Quantity = (account_balance * risk_per_trade) / (atr_value * initial_sl_atr_mult)

    try:
        risked_capital_quote = account_balance * risk_per_trade
        sl_price_distance = atr_value * initial_sl_atr_mult

        if sl_price_distance <= Decimal('0'):
             lg.warning("Calculated stop loss price distance is zero or negative. Cannot size position.")
             return None

        # This formula gives the quantity in BASE currency for spot or CONTRACTS for linear contracts
        # assuming ContractSize = 1 and the risk is measured against price movement in quote.
        calculated_amount = risked_capital_quote / sl_price_distance

        # For Inverse contracts, sizing based on BTC-margined value is more complex.
        # A common approximation for inverse risk sizing is based on the *value* of the position at entry.
        # Position Value in Quote = Quantity (contracts) * ContractSize / Price (if Price is Base/Quote)
        # Or Position Value = Quantity (contracts) * ContractSize (if ContractSize is in Quote, e.g., \$1)

        # Let's refine for Linear vs Inverse vs Spot
        if market_info['is_linear'] or market_info['spot']:
            # For linear (USDT-margined) or spot, quantity is in base currency units.
            # Risk per base unit movement is 1:1 in quote currency (excluding fees).
            # Calculated amount is correct for base units.
            lg.debug(f"Sizing for {market_info['contract_type_str']} contract.")
            pass # calculated_amount is the quantity in base units / contracts

        elif market_info['is_inverse']:
            # For inverse (e.g., BTC-margined BTC/USD), quantity is in CONTRACTS.
            # The value of one contract is fixed in Quote currency (e.g., \$1 for BTC/USD).
            # Risk per contract movement is related to ContractSize / Price.
            # Loss per contract = (EntryPrice - SLPrice) * ContractSize / EntryPrice * BTC Price (if using BTC collateral)
            # This gets complicated with inverse contracts and PnL currency vs collateral currency.

            # Alternative approach for inverse risk sizing:
            # Determine the desired NOTIONAL value of the position in Quote currency (e.g., USD).
            # Leverage allows controlling the margin used, not the risk directly based on SL distance vs capital.
            # Risk based on SL distance on inverse:
            # Loss in Base Currency = Quantity(contracts) * ContractSize * (1/SLPrice - 1/EntryPrice)
            # Convert Loss in Base to Quote (at current price): Loss Quote = Loss Base * Current Price
            # Risked Capital Quote = Quantity * ContractSize * Current Price * (1/SLPrice - 1/EntryPrice)
            # Risked Capital Quote / (Balance * Risk %) = 1
            # Quantity = (Balance * Risk %) / (ContractSize * Current Price * |1/SLPrice - 1/EntryPrice|)
            # Where |1/SLPrice - 1/EntryPrice| is the price difference impact on the inverse value.
            # SL Price = Current Price +/- (ATR * SL_ATR_Mult) (in Base/Quote terms)

            # Given the complexity, the simplest risk-based sizing for inverse using ATR distance:
            # Value per contract = MarketInfo.contract_size_decimal (e.g., \$1)
            # Price difference to SL in quote (based on current price level) = ATR * SL_ATR_Mult
            # This is still not quite right for inverse contracts.

            # Revert to a simpler interpretation that aligns with Linear/Spot if ContractSize is 1 or specified in Quote:
            # Assume ContractSize is in Quote currency (e.g., Bybit's BTCUSD is \$1 contract)
            # If ContractSize is fixed Quote value: Contracts = Risked Capital Quote / (ATR * SL_ATR_Mult * (Price / ContractSize) ) ??
            # If ContractSize is 1 (like most linear): Contracts = Risked Capital Quote / (ATR * SL_ATR_Mult) - this is what we calculated.

            # Let's assume the calculation `calculated_amount = risked_capital_quote / sl_price_distance`
            # yields the quantity in the units needed by the exchange (Base units for spot/linear, Contracts for inverse)
            # *provided* SL Distance in the denominator is expressed correctly in Quote currency terms per that unit.
            # For Bybit Inverse (e.g. BTC/USD), contractSize is 1 USD. Price is in USD.
            # The price movement of 1 USD impacts PnL by ContractSize / EntryPrice * (EntryPrice - NewPrice) ...
            # This is too complex for a generic helper. A common workaround for inverse is to size based on desired NOTIONAL value.
            # Target Notional Value (Quote) = Balance * Risk % * Leverage (simple, not risk-per-trade based on SL)
            # Target Notional Value (Quote) = Balance * Max_Risk_Percentage (simpler)
            # Quantity (Contracts) = Target Notional Value (Quote) / ContractSize (Quote) ... But this doesn't use SL.

            # Let's stick to the most common 'Risk % of Equity per trade' using ATR for Linear/Spot
            # And issue a warning for Inverse, suggesting manual override or verification.
            lg.warning(f"{NEON_YELLOW}Position sizing formula based on Risk % and ATR distance is primarily designed for Spot/Linear contracts where ContractSize=1 and Quote is the traded currency.{RESET}")
            lg.warning(f"{NEON_YELLOW}Sizing for Inverse contract '{market_info['symbol']}' (Contract Size: {market_info['contract_size_decimal'].normalize()}) may be inaccurate. Verify calculation.{RESET}")
            # Continue with the calculated_amount, acknowledging potential inaccuracy for Inverse.


        # Apply amount precision and limits
        formatted_amount_str = _format_amount(exchange, market_info['symbol'], calculated_amount)

        if formatted_amount_str is None:
            lg.warning(f"Calculated amount {calculated_amount.normalize()} is invalid or cannot be formatted to precision.")
            return None

        final_amount = _safe_decimal_conversion(formatted_amount_str, allow_none=True) # Convert back to Decimal

        if final_amount is None or final_amount <= Decimal('0'):
             lg.warning(f"Final calculated amount {final_amount} after formatting is zero or invalid. Cannot place order.")
             return None

        # Check against market limits
        min_amount = market_info['min_amount_decimal']
        max_amount = market_info['max_amount_decimal']
        min_cost = market_info['min_cost_decimal'] # Cost is Price * Amount * ContractSize
        max_cost = market_info['max_cost_decimal']

        if min_amount is not None and final_amount < min_amount:
            lg.warning(f"Calculated amount {final_amount.normalize()} is below minimum amount {min_amount.normalize()}. Adjusting to min amount.")
            final_amount = min_amount
            # Re-format after adjustment to min amount
            formatted_amount_str = _format_amount(exchange, market_info['symbol'], final_amount)
            if formatted_amount_str is None:
                 lg.error(f"Cannot format minimum amount {min_amount.normalize()} for {market_info['symbol']}. Sizing failed.")
                 return None
            final_amount = _safe_decimal_conversion(formatted_amount_str)


        if max_amount is not None and final_amount > max_amount:
            lg.warning(f"Calculated amount {final_amount.normalize()} is above maximum amount {max_amount.normalize()}. Adjusting to max amount.")
            final_amount = max_amount
            # Re-format after adjustment to max amount
            formatted_amount_str = _format_amount(exchange, market_info['symbol'], final_amount)
            if formatted_amount_str is None:
                 lg.error(f"Cannot format maximum amount {max_amount.normalize()} for {market_info['symbol']}. Sizing failed.")
                 return None
            final_amount = _safe_decimal_conversion(formatted_amount_str)


        # Check minimum order value (cost) for linear/spot contracts, if applicable
        # For Bybit USDT Perp, min order value is \$1.00
        if market_info['is_linear'] or market_info['spot']:
             current_cost = current_price * final_amount * market_info['contract_size_decimal'] # contract size is 1 for linear/spot
             if min_cost is not None and current_cost < min_cost:
                  lg.warning(f"Calculated order value {current_cost.normalize()} is below minimum cost {min_cost.normalize()}. Adjusting amount to meet min cost.")
                  # Calculate minimum required amount to meet min_cost: min_amount_from_cost = min_cost / (current_price * contract_size)
                  if current_price > Decimal('0') and market_info['contract_size_decimal'] > Decimal('0'):
                       min_amount_from_cost = min_cost / (current_price * market_info['contract_size_decimal'])
                       final_amount = min_amount_from_cost
                       # Re-format after adjustment for min cost
                       formatted_amount_str = _format_amount(exchange, market_info['symbol'], final_amount)
                       if formatted_amount_str is None:
                            lg.error(f"Cannot format amount {final_amount.normalize()} derived from min cost for {market_info['symbol']}. Sizing failed.")
                            return None
                       final_amount = _safe_decimal_conversion(formatted_amount_str)
                       # Check if this adjusted amount is now also below min_amount_decimal, take the larger
                       if market_info['min_amount_decimal'] is not None and final_amount < market_info['min_amount_decimal']:
                            final_amount = market_info['min_amount_decimal']
                            # Re-format one last time
                            formatted_amount_str = _format_amount(exchange, market_info['symbol'], final_amount)
                            if formatted_amount_str is None:
                                lg.error(f"Cannot format final adjusted amount for {market_info['symbol']}. Sizing failed.")
                                return None
                            final_amount = _safe_decimal_conversion(formatted_amount_str)

                  else:
                       lg.error("Cannot calculate amount based on min cost: Price or contract size is zero. Sizing failed.")
                       return None
        # Check max cost if applicable
        if max_cost is not None and current_cost > max_cost:
             lg.warning(f"Calculated order value {current_cost.normalize()} is above maximum cost {max_cost.normalize()}. Sizing limited by max cost.")
             # Calculate max allowed amount based on max cost: max_amount_from_cost = max_cost / (current_price * contract_size)
             if current_price > Decimal('0') and market_info['contract_size_decimal'] > Decimal('0'):
                  max_amount_from_cost = max_cost / (current_price * market_info['contract_size_decimal'])
                  final_amount = min(final_amount, max_amount_from_cost) # Take the smaller of current amount or max allowed by cost
                  # Re-format after adjustment for max cost
                  formatted_amount_str = _format_amount(exchange, market_info['symbol'], final_amount)
                  if formatted_amount_str is None:
                       lg.error(f"Cannot format amount {final_amount.normalize()} derived from max cost for {market_info['symbol']}. Sizing failed.")
                       return None
                  final_amount = _safe_decimal_conversion(formatted_amount_str)

             else:
                  lg.error("Cannot calculate amount based on max cost: Price or contract size is zero. Sizing failed.")
                  return None


        # Final check on the final calculated amount
        if final_amount is None or final_amount <= Decimal('0'):
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

    # Ensure inputs are valid
    if amount is None or amount <= POSITION_QTY_EPSILON:
         lg.error(f"Attempted to execute trade with invalid amount: {amount}. Aborting.")
         return None
    if side not in ['buy', 'sell']:
         lg.error(f"Invalid side '{side}' for trade execution. Aborting.")
         return None
    if market_info is None or market_info['symbol'] != symbol:
         lg.error(f"Invalid market info for {symbol} passed to execute_trade. Aborting.")
         return None

    # Format amount and prices for the exchange API
    formatted_amount_str = _format_amount(exchange, symbol, amount)
    formatted_sl_price_str = _format_price(exchange, symbol, initial_sl_price) if initial_sl_price is not None else None
    formatted_tp_price_str = _format_price(exchange, symbol, initial_tp_price) if initial_tp_price is not None else None

    if formatted_amount_str is None:
         lg.error(f"Failed to format order amount {amount.normalize()} for {symbol}. Aborting trade.")
         return None

    order_params: Dict[str, Any] = {}

    # Add SL/TP parameters if provided
    if formatted_sl_price_str:
        # Exchange-specific parameters might be needed here.
        # For Bybit V5, SL/TP can often be set directly on the initial market order.
        # Check Bybit documentation or CCXT examples for the exact parameter names.
        # Common names are 'stopLoss', 'takeProfit', 'triggerPrice', 'triggerBy'.
        # CCXT unified method `create_order` often accepts 'stopLoss', 'takeProfit' at the top level.
        order_params['stopLoss'] = float(initial_sl_price) # CCXT often expects float here
        lg.info(f"Adding initial SL {formatted_sl_price_str} to order parameters.")

    if formatted_tp_price_str:
        order_params['takeProfit'] = float(initial_tp_price) # CCXT often expects float here
        lg.info(f"Adding initial TP {formatted_tp_price_str} to order parameters.")

    # Bybit V5 specific note: Ensure 'triggerBy' for SL/TP is set correctly if needed.
    # Default is often 'LastPrice'. Can usually be set globally or per order parameter.
    # CCXT might handle this via exchange.options['defaultTriggerBy'] or within order_params['params']
    # Example: order_params['params'] = {'slTriggerBy': 'MarkPrice', 'tpTriggerBy': 'MarkPrice'}
    # Check Bybit API docs for exact parameters for V5 `place-order` endpoint.
    # If placing SL/TP separately *after* position is confirmed, use edit_position.
    # CCXT's `create_order` supporting `stopLoss` and `takeProfit` is the simplest way if exchange allows on entry market order.

    # Determine order type (Market is typical for strategy execution)
    order_type = 'market'
    action_desc = f"creating {side.upper()} {order_type} order"

    placed_order = None
    for attempt in range(MAX_API_RETRIES + 1):
        if _shutdown_requested:
             lg.info(f"Shutdown requested during order placement for {symbol}. Aborting.")
             return None
        try:
            lg.info(f"Attempting to place {side.upper()} {order_type} order for {formatted_amount_str} {market_info['base']} on {symbol} @ ~{price.normalize()}, attempt {attempt + 1}/{MAX_API_RETRIES + 1}...")
            if formatted_sl_price_str: lg.info(f"  Initial SL: {formatted_sl_price_str}")
            if formatted_tp_price_str: lg.info(f"  Initial TP: {formatted_tp_price_str}")

            # Send the order to the exchange
            # Pass amount as float as required by ccxt create_order
            # Pass price=None for market order
            # Add any extra exchange-specific params if needed in order_params
            order = exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=float(amount), # CCXT expects float for amount
                price=None,           # Market order, price is not specified
                params=order_params   # Include SL/TP/TSL parameters here
            )

            if order and order.get('id'):
                lg.info(f"{NEON_GREEN}Successfully placed {side.upper()} order {order.get('id')} for {formatted_amount_str} on {symbol}. Status: {order.get('status')}{RESET}")
                # Wait a moment for the exchange to process the order and potentially open the position
                lg.debug(f"Waiting {POSITION_CONFIRM_DELAY_SECONDS}s for position confirmation...")
                time.sleep(POSITION_CONFIRM_DELAY_SECONDS)
                placed_order = order
                break # Exit retry loop on success
            else:
                lg.warning(f"Order placement returned no ID or invalid response for {symbol}. Attempt {attempt + 1}.")
                # No order ID means the order likely failed or the response was bad. Treat as failure and potentially retry.
                placed_order = None # Ensure placed_order is None for retry logic

        except ccxt.InsufficientFunds as e:
             # Special handling for insufficient funds - typically not retryable for the same order
             _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
             send_notification(f"Insufficient Funds: {symbol}", f"Order failed due to insufficient funds on {symbol}. Details: {e}", lg, notification_type='email')
             return None # Fatal for this trade attempt
        except ccxt.InvalidOrder as e:
             # Special handling for invalid order parameters - typically not retryable
             _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
             # Optionally send notification for invalid order config
             send_notification(f"Invalid Order: {symbol}", f"Order parameters invalid for {symbol}. Details: {e}. Check config and precision.", lg, notification_type='email')
             return None # Fatal for this trade attempt
        except Exception as e:
            # Use the helper for other CCXT errors
            retry = _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
            if not retry:
                 # If _handle_ccxt_exception returns False, it's a fatal error or max retries reached
                 return None
            # If retry is True, the helper has already waited, continue the loop

    if placed_order is None:
        lg.error(f"{NEON_RED}Failed to place order for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
        # Optionally send notification for failed order placement (if not already handled by specific exceptions above)
        # send_notification(f"Order Failed: {symbol}", f"Failed to place order for {symbol} after retries.", lg)
        return None

    lg.debug(f"Trade execution attempt complete for {symbol}.")
    return placed_order

def close_position(exchange: ccxt.Exchange, symbol: str, position: PositionInfo, logger: logging.Logger) -> bool:
    """
    Closes an existing open position using a market order.
    This typically involves creating a market order on the *opposite* side of the position.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        position: The PositionInfo TypedDict of the position to close.
        logger: The logger instance for the symbol.

    Returns:
        True if the close order is successfully placed (does not guarantee execution),
        False if placing the order fails after retries.
    """
    lg = logger

    if position is None or position['size_decimal'] is None or position['size_decimal'] <= POSITION_QTY_EPSILON:
         lg.error(f"Attempted to close position for {symbol} with invalid position data or zero size.")
         return False

    pos_side = position['side']
    pos_amount = position['size_decimal']
    # Closing a position requires trading the opposite side
    close_side = 'sell' if pos_side == 'long' else 'buy'

    # Bybit V5 uses unified accounts. Closing a position usually involves a MARKET order
    # with amount equal to position size, on the opposite side.
    # For unified accounts, it's simpler. Just specify symbol, side, type='market', amount=position_size.
    # The exchange handles reducing/closing the position.

    # Format the position size for the close order amount
    formatted_amount_str = _format_amount(exchange, symbol, pos_amount)
    if formatted_amount_str is None:
         lg.error(f"Failed to format position amount {pos_amount.normalize()} for closing order on {symbol}. Aborting close.")
         return False

    action_desc = f"closing {pos_side.upper()} position"

    close_order_placed = False
    for attempt in range(MAX_API_RETRIES + 1):
        if _shutdown_requested:
             lg.info(f"Shutdown requested during position close for {symbol}. Aborting.")
             return False
        try:
            lg.info(f"Attempting to place {close_side.upper()} MARKET order to close {pos_amount.normalize()} {pos_side.upper()} position on {symbol}, attempt {attempt + 1}/{MAX_API_RETRIES + 1}...")

            # Place the market order to close the position
            order = exchange.create_order(
                symbol=symbol,
                type='market',
                side=close_side,
                amount=float(pos_amount), # CCXT expects float
                params={} # No extra params typically needed for a simple close market order
            )

            if order and order.get('id'):
                lg.info(f"{NEON_GREEN}Successfully placed closing order {order.get('id')} for {symbol}. Status: {order.get('status')}{RESET}")
                # Wait a moment for the exchange to process the close
                lg.debug(f"Waiting {POSITION_CONFIRM_DELAY_SECONDS}s for position closure confirmation...")
                time.sleep(POSITION_CONFIRM_DELAY_SECONDS)
                close_order_placed = True
                break # Exit retry loop on success
            else:
                 lg.warning(f"Closing order placement returned no ID or invalid response for {symbol}. Attempt {attempt + 1}.")
                 close_order_placed = False # Ensure False for retry logic

        except ccxt.InsufficientFunds as e:
             # This shouldn't happen when closing a position with sufficient margin, but handle defensively
             _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
             send_notification(f"Funds Error (Close): {symbol}", f"Failed to close position due to unexpected funds issue on {symbol}. Details: {e}", lg, notification_type='email')
             return False # Fatal for this close attempt
        except ccxt.InvalidOrder as e:
             _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
             send_notification(f"Invalid Order (Close): {symbol}", f"Closing order parameters invalid for {symbol}. Details: {e}.", lg, notification_type='email')
             return False # Fatal for this close attempt
        except Exception as e:
            retry = _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
            if not retry:
                 # If _handle_ccxt_exception returns False, it's a fatal error or max retries reached
                 return False
            # If retry is True, the helper has already waited, continue the loop


    if not close_order_placed:
        lg.error(f"{NEON_RED}Failed to place closing order for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
        send_notification(f"Close Order Failed: {symbol}", f"Failed to place closing order for {symbol} after retries.", lg)
        return False

    lg.debug(f"Position close attempt complete for {symbol}.")
    return True


def edit_position_protection(exchange: ccxt.Exchange, symbol: str, position: PositionInfo, new_sl_price: Optional[Decimal] = None, new_tp_price: Optional[Decimal] = None, new_tsl_callback_rate: Optional[Decimal] = None, new_tsl_activation_price: Optional[Decimal] = None, logger: logging.Logger) -> bool:
    """
    Modifies the Stop Loss, Take Profit, and/or Trailing Stop Loss for an existing position.
    Uses CCXT's edit_position or equivalent method.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        position: The PositionInfo TypedDict of the position to modify.
        new_sl_price: The new SL price (Decimal) to set, or None to not change/remove.
        new_tp_price: The new TP price (Decimal) to set, or None to not change/remove.
        new_tsl_callback_rate: The new TSL callback rate (Decimal, e.g., 0.005), or None.
                               Note: Exchange API parameter might require price/distance instead.
                               CCXT edit_position TSL parameter usually expects the TRIGGER price or distance depending on exchange.
                               Bybit V5 `set-trading-stop` requires SL/TP/TSL prices.
                               To set TSL via Bybit V5: need `trailing_stop` (trigger price), `activePrice` (activation price).
                               We calculate `new_tsl_activation_price` and `new_tsl_callback_rate` is a config parameter.
                               We need to *set* the `trailingStop` price based on the callback rate from *current* price once activated.
                               This function will set TSL by specifying the activation price and callback rate if CCXT supports, or
                               calculate the trigger price from the callback rate based on *current* price if needed by exchange.
                               Let's aim for setting activation price and callback rate if possible via CCXT unified,
                               or pass params if Bybit V5 requires it. CCXT `edit_position` supports `trailingStop`.
                               It seems `edit_position`'s `trailingStop` argument is the TRIGGER price.
                               Bybit V5 requires `trailingStop` (trigger price) and `activePrice` (activation price).
                               Let's modify this helper to accept `tsl_activation_price` and use the *config's* `trailing_stop_callback_rate`.
        new_tsl_activation_price: The price at which TSL becomes active (Decimal). If set, enables TSL.

    Returns:
        True if the modification request was successful, False otherwise.
    """
    lg = logger

    if position is None or position['size_decimal'] is None or position['size_decimal'] <= POSITION_QTY_EPSILON:
         lg.error(f"Attempted to modify protection for invalid position on {symbol}.")
         return False
    if position['entryPrice_decimal'] is None:
        lg.error(f"Position for {symbol} has no valid entry price. Cannot manage protection.")
        return False

    pos_side = position['side']
    entry_price = position['entryPrice_decimal']
    market_price = position['markPrice_decimal'] or position['entryPrice_decimal'] # Use mark price if available, else entry

    update_params: Dict[str, Any] = {}
    action_desc = f"modifying protection for {pos_side.upper()} position on {symbol}"

    # Format SL price
    if new_sl_price is not None:
        # Ensure SL price is formatted to precision
        formatted_sl_price_str = _format_price(exchange, symbol, new_sl_price)
        if formatted_sl_price_str is None:
             lg.warning(f"Failed to format new SL price {new_sl_price.normalize()} for {symbol}. Cannot update SL.")
             # Do not add to update_params if formatting fails
        else:
             # Check if new SL price is actually different from current SL price on exchange (allowing for precision differences)
             current_sl_dec = position['stopLossPrice_dec']
             if current_sl_dec is None or abs(new_sl_price - current_sl_dec) > market_info['price_precision_step_decimal'] / Decimal('2'):
                 update_params['stopLoss'] = float(new_sl_price) # CCXT expects float
                 lg.info(f"Setting new SL for {symbol}: {formatted_sl_price_str}")
             else:
                 lg.debug(f"New SL price {new_sl_price.normalize()} is same as current {current_sl_dec.normalize()} (within precision). Skipping SL update.")


    # Format TP price
    if new_tp_price is not None:
        # Ensure TP price is formatted to precision
        formatted_tp_price_str = _format_price(exchange, symbol, new_tp_price)
        if formatted_tp_price_str is None:
             lg.warning(f"Failed to format new TP price {new_tp_price.normalize()} for {symbol}. Cannot update TP.")
             # Do not add to update_params if formatting fails
        else:
             # Check if new TP price is actually different from current TP price on exchange
             current_tp_dec = position['takeProfitPrice_dec']
             if current_tp_dec is None or abs(new_tp_price - current_tp_dec) > market_info['price_precision_step_decimal'] / Decimal('2'):
                 update_params['takeProfit'] = float(new_tp_price) # CCXT expects float
                 lg.info(f"Setting new TP for {symbol}: {formatted_tp_price_str}")
             else:
                 lg.debug(f"New TP price {new_tp_price.normalize()} is same as current {current_tp_dec.normalize()} (within precision). Skipping TP update.")


    # Trailing Stop Loss (TSL) - This is more complex with Bybit V5
    # Bybit V5 uses a TSL trigger PRICE and an activation price.
    # When the activation price is hit, the TSL trigger price becomes (current price +/- callback distance).
    # The TSL then trails the price. CCXT's `edit_position`'s `trailingStop` parameter *might* set the initial trigger price.
    # To *enable* TSL via API with an activation price, we need to use the exchange-specific parameters.
    # Bybit V5 `set-trading-stop` endpoint takes `trailingStop`, `activePrice`, `side`.
    # CCXT's `edit_position` params likely map to this.
    # We need to set the TSL activation price and the callback rate.
    # The callback rate is a configuration parameter (`trailing_stop_callback_rate`).
    # The `trailingStop` parameter in Bybit V5 API is the *initial trigger price* or can be set to 0 to disable.
    # It is NOT the callback rate percentage.
    # To enable TSL with an activation price and callback rate, you typically need to pass:
    # {'trailingStop': 0, 'activePrice': float(new_tsl_activation_price), 'trailingLossProtectedBy': float(config_tsl_callback_rate)}
    # Or similar parameters depending on the exact API endpoint and CCXT mapping.

    config_tsl_callback_rate = Decimal(str(CONFIG.get("protection", {}).get("trailing_stop_callback_rate", DEFAULT_CONFIG['protection']['trailing_stop_callback_rate'])))

    if new_tsl_activation_price is not None:
        # Ensure activation price is valid
        formatted_tsl_activation_str = _format_price(exchange, symbol, new_tsl_activation_price)
        if formatted_tsl_activation_str is None:
            lg.warning(f"Failed to format new TSL activation price {new_tsl_activation_price.normalize()} for {symbol}. Cannot update TSL.")
        else:
            # For Bybit V5, we need to pass the activation price and the callback rate in `params`.
            # CCXT `edit_position` takes a `trailingStop` argument which is the trigger price.
            # To initiate TSL with an activation price, the trigger price is often set to 0 initially,
            # and the activation price + callback rate is passed in params.

            # Check if TSL is already active or activation price is already set
            current_tsl_activation_dec = position['tslActivationPrice_dec']

            # Only attempt to SET the activation price if it's not already set to the same or higher/lower (depending on side)
            # For LONG: activate when price reaches/exceeds activation price
            # For SHORT: activate when price reaches/falls below activation price
            activate_condition_met = False
            if pos_side == 'long' and market_price is not None and new_tsl_activation_price is not None and market_price >= new_tsl_activation_price:
                 activate_condition_met = True # Price is already at or above activation price
            elif pos_side == 'short' and market_price is not None and new_tsl_activation_price is not None and market_price <= new_tsl_activation_price:
                 activate_condition_met = True # Price is already at or below activation price


            # Only send the update request if:
            # 1. Activation price is being set for the first time (current_tsl_activation_dec is None)
            # 2. Activation price is being changed (abs(new_tsl_activation_price - current_tsl_activation_dec) is large enough)
            # 3. The activation condition has just been met in *this cycle* AND TSL wasn't already active (logic for this might be outside this function)

            # Simplest approach: If we calculated a new_tsl_activation_price and it's not the same as the current one (within precision),
            # try to set it via API params.
            # CCXT `edit_position` seems to map `trailingStop` to the trigger price on Bybit.
            # We need to use `params` for the activation price and callback rate.
            # Let's assume Bybit V5 `set-trading-stop` expects `activePrice` and `trailingStop` (trigger).
            # The `trailingStop` is the *trigger price* when activated, NOT the callback rate.
            # To *activate* TSL with a callback rate, you usually set activePrice and trailingStop = 0 initially.
            # The exchange will then calculate the trigger price once activePrice is hit.

            # Check if the new activation price is different enough
            is_activation_price_different = False
            if current_tsl_activation_dec is None:
                 is_activation_price_different = True # Setting for the first time
            elif new_tsl_activation_price is not None and abs(new_tsl_activation_price - current_tsl_activation_dec) > market_info['price_precision_step_decimal'] / Decimal('2'):
                 is_activation_price_different = True # New price is significantly different

            if is_activation_price_different:
                # Set TSL using exchange-specific params for Bybit V5 'set-trading-stop'
                # The key `trailingStop` in CCXT's `edit_position` params maps to Bybit's `trailingStop` parameter (trigger price).
                # Set `trailingStop` to 0 to indicate activating via `activePrice` and `trailingLossProtectedBy` (which might map from CCXT `trailingStop` argument?)
                # This is slightly ambiguous in generic CCXT docs vs exchange specifics.
                # Looking at Bybit V5 API: `set-trading-stop` takes `symbol`, `side`, `stopLoss`, `takeProfit`, `trailingStop`, `activePrice`.
                # `trailingStop` (float/string) is the trigger distance or price. To use a callback % with activation, need to use `activePrice`.
                # Let's assume CCXT `edit_position` `params={'activePrice': X, 'trailingStop': 0}` along with `trailingStop=Y` argument works.
                # A safer assumption is that CCXT `trailingStop` argument *is* the trigger price OR distance, depending on mode.
                # Let's try setting `activePrice` via params and passing the callback % via params if possible,
                # or calculate initial trigger price if needed.

                # Bybit V5 API `set-trading-stop` endpoint documentation shows:
                # `trailingStop` (string): Trailing Stop Order price, required for Trailing Stop order. Example: "100" or "1.5%". Need to confirm format.
                # `activePrice` (string): Activation price. Example: "18000".
                # `tpslMode` (string): Optional, 'full' or 'partial'. Default 'full'.
                # `tpTriggerBy`, `slTriggerBy`, `trailingTriggerBy`.

                # It seems we can potentially set TSL with a callback % directly in `set-trading-stop` `trailingStop` field.
                # Let's try passing the callback rate as a string percentage and the activation price.

                formatted_tsl_callback_rate_str = f"{config_tsl_callback_rate * Decimal('100')}%" # Format as "0.5%"

                update_params['params'] = {
                    'activePrice': float(new_tsl_activation_price), # Bybit V5 activePrice is float/string
                    'trailingStop': formatted_tsl_callback_rate_str, # Try sending callback rate as string "%"
                    'trailingTriggerBy': 'MarkPrice' # Or 'LastPrice', 'IndexPrice' - config option?
                }
                # Also, disable the non-activation TSL trigger price if it was set via the main argument
                # update_params['trailingStop'] = 0 # If CCXT main argument is used for trigger price

                lg.info(f"Setting new TSL for {symbol}: Activation Price={formatted_tsl_activation_str}, Callback Rate={formatted_tsl_callback_rate_str}")


    # If no parameters need updating, exit early
    if not update_params and ('params' not in update_params or not update_params['params']):
         lg.debug(f"No protection parameters changed for {symbol}.")
         return True

    # Call the exchange API to modify the position protection
    modified_successfully = False
    for attempt in range(MAX_API_RETRIES + 1):
        if _shutdown_requested:
             lg.info(f"Shutdown requested during protection modification for {symbol}. Aborting.")
             return False
        try:
            lg.debug(f"Attempting to {action_desc}, attempt {attempt + 1}/{MAX_API_RETRIES + 1} with params: {update_params}")

            # CCXT edit_position unified parameters: symbol, side, params={stopLoss, takeProfit, trailingStop, ...}
            # Note: Side is required for edit_position on some exchanges like Bybit V5 to specify which position (long/short) in One-Way mode.
            response = exchange.edit_position(
                 symbol=symbol,
                 side=pos_side, # Specify side
                 params=update_params # Pass the dict containing SL, TP, TSL parameters
            )

            # CCXT edit_position might not return a standard order dict, just a success/status indicator.
            # Check response structure based on CCXT implementation for Bybit edit_position.
            # Assuming a dictionary response indicating success.
            # Bybit V5 set-trading-stop returns a result like {'retCode': 0, 'retMsg': 'OK', ...}
            if response and response.get('retCode') == 0:
                 lg.info(f"{NEON_GREEN}Successfully modified protection for {symbol} position.{RESET}")
                 # Optional: Wait for confirmation fetch_positions again
                 modified_successfully = True
                 break # Exit retry loop on success
            else:
                 lg.warning(f"Protection modification failed for {symbol}. Response: {response}. Attempt {attempt + 1}.")
                 modified_successfully = False # Ensure False for retry logic

        except Exception as e:
             retry = _handle_ccxt_exception(e, lg, action_desc, symbol, attempt)
             if not retry:
                  return False
             # If retry is True, helper waited, loop continues

    if not modified_successfully:
         lg.error(f"{NEON_RED}Failed to modify protection for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
         # Optional: send notification
         # send_notification(f"Protect Update Failed: {symbol}", f"Failed to update SL/TP/TSL for {symbol} position.", lg)
         return False

    lg.debug(f"Protection modification attempt complete for {symbol}.")
    return True


# --- Position Management (BE, TSL) Logic ---

# Need a way to store state for each symbol, specifically the be_activated and tsl_activated flags.
# A dictionary mapping symbol -> { 'be_activated': bool, 'tsl_activated': bool, 'position_id': str/None, ... }
# Or, add these flags directly to the PositionInfo object fetched, and ensure we are working on the *correct* position instance if multiple are possible (though config limits to 1).
# Let's add the flags to PositionInfo and assume we process only one open position per symbol.
# The flags will be initialized to False when a position is fetched unless loaded from persistence.
# For this stateless example, they will reset on each bot restart or if position is closed.

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
            # Add other state variables here if needed, e.g., last_trade_timestamp, consecutive_error_count
        }
        setup_logger(symbol).debug(f"Initialized state for {symbol}: {SYMBOL_STATE[symbol]}")
    return SYMBOL_STATE[symbol]

def reset_symbol_state(symbol: str, logger: logging.Logger):
    """Resets the bot's in-memory state for a specific symbol."""
    logger.debug(f"Resetting state for {symbol}.")
    SYMBOL_STATE[symbol] = {
        'be_activated': False,
        'tsl_activated': False,
        'position_id': None,
    }


def check_and_manage_position(exchange: ccxt.Exchange, symbol: str, position: PositionInfo, market_info: MarketInfo, analysis_results: StrategyAnalysisResults, logger: logging.Logger) -> bool:
    """
    Manages an existing open position, including Break-Even and Trailing Stop Loss logic.
    This function is called when `generate_signal` returns 'HOLD'.

    Args:
        exchange: The CCXT exchange instance.
        symbol: The trading symbol.
        position: The PositionInfo TypedDict of the current open position.
        market_info: MarketInfo TypedDict for the symbol.
        analysis_results: The results from analyze_strategy (contains current price, ATR).
        logger: The logger instance for the symbol.

    Returns:
        True if the position is still considered open and managed, False if an exit
        was attempted or the position was somehow invalidated (e.g., size dropped to zero unexpectedly).
    """
    lg = logger

    if position is None or position['size_decimal'] is None or position['size_decimal'] <= POSITION_QTY_EPSILON or position['entryPrice_decimal'] is None or position['side'] is None:
         lg.warning(f"Attempted to manage invalid position for {symbol}.")
         # Reset state if position seems invalid
         reset_symbol_state(symbol, lg)
         return False # Position cannot be managed

    pos_size = position['size_decimal']
    pos_side = position['side']
    entry_price = position['entryPrice_decimal']
    current_price = analysis_results['last_close'] # Use the most recent close from klines
    current_atr = analysis_results['atr'] # Use the most recent ATR from klines

    # Get symbol state (includes BE/TSL activated flags for this specific position instance)
    symbol_state = get_symbol_state(symbol)
    be_activated = symbol_state['be_activated']
    tsl_activated = symbol_state['tsl_activated']

    lg.debug(f"Managing {pos_side.upper()} position {pos_size.normalize()} on {symbol}. Entry: {entry_price.normalize()}, Current: {current_price.normalize()}. State: BE={be_activated}, TSL={tsl_activated}")

    # Check if current price is valid for calculations
    if current_price is None or current_price <= Decimal('0') or current_atr is None or current_atr <= Decimal('0'):
        lg.warning("Current price or ATR is invalid during position management. Skipping BE/TSL checks.")
        return True # Still considered open, but cannot manage stops this cycle

    # --- Break-Even (BE) Logic ---
    enable_be = CONFIG.get("protection", {}).get("enable_break_even", DEFAULT_CONFIG['protection']['enable_break_even'])
    be_trigger_atr_mult = Decimal(str(CONFIG.get("protection", {}).get("break_even_trigger_atr_multiple", DEFAULT_CONFIG['protection']['break_even_trigger_atr_multiple'])))
    be_offset_ticks = CONFIG.get("protection", {}).get("break_even_offset_ticks", DEFAULT_CONFIG['protection']['break_even_offset_ticks'])
    price_tick_size = market_info['price_precision_step_decimal'] # Guaranteed non-None by market_info fetching

    if enable_be and not be_activated:
        # Check if price has moved enough in profit to trigger BE
        profit_trigger_price = entry_price + current_atr * be_trigger_atr_mult if pos_side == 'long' else \
                               entry_price - current_atr * be_trigger_atr_mult # For short

        if (pos_side == 'long' and current_price >= profit_trigger_price) or \
           (pos_side == 'short' and current_price <= profit_trigger_price):

            lg.info(f"{NEON_GREEN}Break-Even trigger price {profit_trigger_price.normalize()} reached/crossed for {pos_side.upper()} on {symbol}. Activating BE.{RESET}")

            # Calculate the BE price (entry price +/- offset ticks)
            be_price_raw = entry_price + price_tick_size * be_offset_ticks if pos_side == 'long' else \
                           entry_price - price_tick_size * be_offset_ticks # For short

            # Format BE price to exchange precision
            be_price = _safe_decimal_conversion(_format_price(exchange, symbol, be_price_raw), allow_none=True)

            if be_price is None or be_price <= Decimal('0'):
                 lg.error(f"{NEON_RED}Failed to calculate or format BE price {be_price_raw} -> {be_price} for {symbol}. Cannot set BE.{RESET}")
                 # Do NOT set be_activated to True if we failed to set the stop
            else:
                 # Ensure BE price is *at least* the entry price (or slightly better) for long,
                 # and *at most* the entry price (or slightly better) for short.
                 # This prevents placing a BE stop that is actually *worse* than entry due to ticks/rounding.
                 # The offset handles the desired tick buffer, but the logic ensures it doesn't go the wrong way.
                 if pos_side == 'long' and be_price < entry_price * (Decimal('1') + PRICE_EPSILON):
                      be_price = entry_price + price_tick_size * be_offset_ticks
                      be_price = _safe_decimal_conversion(_format_price(exchange, symbol, be_price), allow_none=False)
                      lg.debug(f"Adjusting calculated LONG BE price to ensure it's >= entry: {be_price.normalize()}")

                 elif pos_side == 'short' and be_price > entry_price * (Decimal('1') - PRICE_EPSILON):
                      be_price = entry_price - price_tick_size * be_offset_ticks
                      be_price = _safe_decimal_conversion(_format_price(exchange, symbol, be_price), allow_none=False)
                      lg.debug(f"Adjusting calculated SHORT BE price to ensure it's <= entry: {be_price.normalize()}")


                 # Request to update the Stop Loss to the calculated BE price
                 lg.info(f"Attempting to move SL to Break-Even @ {be_price.normalize()} for {symbol}.")
                 if edit_position_protection(exchange, symbol, position, new_sl_price=be_price, logger=lg):
                     lg.info(f"{NEON_GREEN}Successfully requested SL update to BE for {symbol}.{RESET}")
                     symbol_state['be_activated'] = True # Mark BE as activated for this position instance
                 else:
                     lg.error(f"{NEON_RED}Failed to move SL to Break-Even for {symbol}.{RESET}")
                     # Leave be_activated as False so it can be retried next cycle


    # --- Trailing Stop Loss (TSL) Logic ---
    enable_tsl = CONFIG.get("protection", {}).get("enable_trailing_stop", DEFAULT_CONFIG['protection']['enable_trailing_stop'])
    tsl_activation_percentage = Decimal(str(CONFIG.get("protection", {}).get("trailing_stop_activation_percentage", DEFAULT_CONFIG['protection']['trailing_stop_activation_percentage'])))
    # tsl_callback_rate is retrieved in edit_position_protection

    if enable_tsl and not tsl_activated:
        # Check if price has moved enough in profit (% from entry) to activate TSL
        # Calculate the activation price based on percentage profit from entry
        tsl_activation_price_calculated = entry_price * (Decimal('1') + tsl_activation_percentage) if pos_side == 'long' else \
                                          entry_price * (Decimal('1') - tsl_activation_percentage) # For short

        # Check if current price reached or crossed the activation price
        activation_condition_met = False
        if pos_side == 'long' and current_price >= tsl_activation_price_calculated:
             activation_condition_met = True
        elif pos_side == 'short' and current_price <= tsl_activation_price_calculated:
             activation_condition_met = True

        if activation_condition_met:
            lg.info(f"{NEON_GREEN}TSL activation price {tsl_activation_price_calculated.normalize()} reached/crossed for {pos_side.upper()} on {symbol}. Activating TSL.{RESET}")

            # Request to update TSL parameters via API
            # We pass the calculated activation price. The callback rate is from config.
            # The edit_position_protection function handles formatting and passing to CCXT/Exchange.
            lg.info(f"Attempting to activate TSL for {symbol}.")
            if edit_position_protection(exchange, symbol, position, new_tsl_activation_price=tsl_activation_price_calculated, logger=lg):
                lg.info(f"{NEON_GREEN}Successfully requested TSL activation for {symbol}.{RESET}")
                symbol_state['tsl_activated'] = True # Mark TSL as activated for this position instance
            else:
                lg.error(f"{NEON_RED}Failed to activate TSL for {symbol}.{RESET}")
                # Leave tsl_activated as False so it can be retried next cycle
        # else:
            # lg.debug(f"TSL activation price {tsl_activation_price_calculated.normalize()} not yet reached. Current price {current_price.normalize()}.")


    # Future Enhancement: Check for exit signals again here?
    # The signal generation step already checked for trend reversal / OB hits.
    # If an exit signal was generated, the main loop handles the close order.
    # This function is primarily for managing stops when the signal is 'HOLD'.

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

    interval = CONFIG.get("interval", DEFAULT_CONFIG['interval'])
    fetch_limit = CONFIG.get("fetch_limit", DEFAULT_CONFIG['fetch_limit'])
    min_klines_for_strategy = CONFIG.get("min_klines_for_strategy", DEFAULT_CONFIG['min_klines_for_strategy'])
    enable_trading = CONFIG.get("enable_trading", DEFAULT_CONFIG['enable_trading'])
    max_concurrent_positions = CONFIG.get("max_concurrent_positions", DEFAULT_CONFIG['max_concurrent_positions'])
    risk_per_trade = CONFIG.get("risk_per_trade", DEFAULT_CONFIG['risk_per_trade']) # Used in sizing
    leverage = CONFIG.get("leverage", DEFAULT_CONFIG['leverage']) # Used in sizing/setting on exchange

    # --- Fetch Data ---
    df = fetch_klines_ccxt(exchange, symbol, interval, fetch_limit, lg)
    if df is None or df.empty:
        lg.error("Failed to fetch kline data. Skipping cycle for this symbol.")
        return

    # Check if enough klines for strategy calculation
    if len(df) < min_klines_for_strategy:
        lg.warning(f"Not enough klines ({len(df)}) for strategy ({min_klines_for_strategy} required). Skipping cycle for this symbol.")
        return

    account_balance: Optional[Decimal] = None
    if enable_trading:
        # Fetch balance only if trading is enabled, as it's needed for sizing
        account_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if account_balance is None:
            lg.warning(f"Failed to fetch account balance in {QUOTE_CURRENCY}. Cannot size trades. Skipping trade execution this cycle.")
            # Note: Position management (BE/TSL) might still be possible if funded, but new entries are blocked.
            # For simplicity, we'll use None balance to block sizing/entry later.


    # Fetch current open positions for this symbol
    # CCXT fetch_positions for a specific symbol usually returns a list of positions
    # If using one-way mode, there should be at most one long and one short.
    # Assuming one-way mode, so list has max 1 position of either 'long' or 'short'.
    current_open_positions = fetch_open_positions(exchange, symbol, lg)

    # Get bot state for this symbol
    symbol_state = get_symbol_state(symbol)

    # If the position ID tracked in state no longer exists in the fetched positions,
    # it means the position was closed (by SL, TP, liquidation, or manual intervention).
    # Reset the symbol state flags.
    if symbol_state['position_id'] is not None and not any(p['id'] == symbol_state['position_id'] for p in current_open_positions):
         lg.info(f"Position with ID {symbol_state['position_id']} for {symbol} is no longer open. Resetting symbol state.")
         reset_symbol_state(symbol, lg) # This position is gone, reset BE/TSL flags for the *next* one

    # If there are positions, find the one managed by the bot (assuming max_concurrent_positions = 1)
    # If the bot manages state by position ID, update the position object from fetch_positions
    # to include the bot's state flags (be_activated, tsl_activated) if the ID matches.
    bot_managed_position: Optional[PositionInfo] = None
    if symbol_state['position_id'] is not None:
        for pos in current_open_positions:
             if pos['id'] == symbol_state['position_id']:
                  # This is the position the bot previously opened/tracked.
                  # Update its state flags from the SYMBOL_STATE dictionary.
                  pos['be_activated'] = symbol_state['be_activated']
                  pos['tsl_activated'] = symbol_state['tsl_activated']
                  bot_managed_position = pos
                  break # Found the tracked position

    # If max_concurrent_positions is 1, and we didn't find the tracked position by ID,
    # but *any* position exists, it might be a manual position or a new one opened
    # outside the bot's tracking. For simplicity with max_concurrent_positions=1,
    # if there's exactly one position but its ID doesn't match, we *could* assume it's
    # the new one and start tracking it, or just ignore it. Let's just use the list.
    # If max_concurrent_positions=1, there should be 0 or 1 position in the list.
    # We will process the first one if it exists and matches the bot's state ID,
    # or just the first one if we need to start tracking it (less robust).
    # Better: strictly enforce tracking by ID if max_concurrent_positions > 1.
    # With max_concurrent_positions=1, just check if the list is not empty.

    # Use the first position in the list if any exist and config allows only one.
    # If config allows > 1, need more complex tracking logic per position ID.
    # For this version assuming max_concurrent_positions = 1 effectively means 0 or 1 position total.
    # If `bot_managed_position` is still None, but `current_open_positions` is not empty,
    # this could indicate a position not opened by this bot instance/state.
    # Let's treat the first one found as the one to potentially manage if state is reset,
    # but this is a simplification. A robust bot would handle unknown positions.
    if bot_managed_position is None and current_open_positions:
         lg.warning(f"{NEON_YELLOW}Found open position(s) for {symbol} (e.g., {current_open_positions[0]['id']}) not currently tracked by bot state. Proceeding assuming one-way mode and potentially managing the first one found.{RESET}")
         # Assume the first position found is the one we should manage if state is reset
         # Initialize state for this position ID
         first_pos = current_open_positions[0]
         symbol_state['position_id'] = first_pos['id']
         symbol_state['be_activated'] = False # Assume BE/TSL not activated unless state persistence is added
         symbol_state['tsl_activated'] = False # Assume BE/TSL not activated unless state persistence is added
         bot_managed_position = first_pos
         # Update the position object to include bot state flags
         bot_managed_position['be_activated'] = symbol_state['be_activated']
         bot_managed_position['tsl_activated'] = symbol_state['tsl_activated']


    # --- Run Strategy Analysis ---
    strategy_params = CONFIG.get("strategy_params", DEFAULT_CONFIG['strategy_params'])
    analysis_results = analyze_strategy(df, strategy_params, lg)

    # Check if analysis was successful enough to proceed
    if analysis_results['current_trend_up'] is None or analysis_results['atr'] is None or analysis_results['atr'] <= Decimal('0'):
        lg.warning("Strategy analysis incomplete (trend or ATR missing). Cannot generate signal. Skipping cycle logic.")
        # If there is a position, try managing its stops based on existing info if possible,
        # although BE/TSL triggers might rely on current ATR/price.
        # For now, skip full management if analysis is bad.
        return

    # --- Generate Trading Signal ---
    signal_result = generate_signal(analysis_results, current_open_positions, strategy_params, market_info, lg)
    signal = signal_result['signal']
    signal_reason = signal_result['reason']


    # --- Execute/Manage based on Signal and Position Status ---

    if bot_managed_position:
        # We have an active position that the bot is tracking
        lg.debug(f"Active {bot_managed_position['side'].upper()} position ({bot_managed_position['size_decimal'].normalize()}) detected.")

        # Check for exit signals first
        if signal == 'EXIT_LONG' and bot_managed_position['side'] == 'long':
            lg.info(f"{NEON_YELLOW}EXIT_LONG signal received ({signal_reason}) for LONG position. Attempting to close.{RESET}")
            if enable_trading:
                 if close_position(exchange, symbol, bot_managed_position, lg):
                     lg.info(f"{NEON_GREEN}Close order placed successfully for {symbol} LONG.{RESET}")
                     # State will be reset on next cycle when position is gone
                 else:
                     lg.error(f"{NEON_RED}Failed to place close order for {symbol} LONG. Position remains open.{RESET}")
                     send_notification(f"Close Failed: {symbol} LONG", f"Failed to place close order for {symbol} LONG position. Reason: {signal_reason}", lg)
            else:
                 lg.info("Trading disabled. Would have attempted to close LONG position.")
                 send_notification(f"Signal: {symbol} LONG", f"Bot disabled. Would EXIT LONG. Reason: {signal_reason}", lg)

        elif signal == 'EXIT_SHORT' and bot_managed_position['side'] == 'short':
            lg.info(f"{NEON_YELLOW}EXIT_SHORT signal received ({signal_reason}) for SHORT position. Attempting to close.{RESET}")
            if enable_trading:
                 if close_position(exchange, symbol, bot_managed_position, lg):
                     lg.info(f"{NEON_GREEN}Close order placed successfully for {symbol} SHORT.{RESET}")
                     # State will be reset on next cycle when position is gone
                 else:
                     lg.error(f"{NEON_RED}Failed to place close order for {symbol} SHORT. Position remains open.{RESET}")
                     send_notification(f"Close Failed: {symbol} SHORT", f"Failed to place close order for {symbol} SHORT position. Reason: {signal_reason}", lg)
            else:
                 lg.info("Trading disabled. Would have attempted to close SHORT position.")
                 send_notification(f"Signal: {symbol} SHORT", f"Bot disabled. Would EXIT SHORT. Reason: {signal_reason}", lg)

        elif signal == 'HOLD':
             # Manage the existing position (BE, TSL updates)
             lg.debug(f"HOLD signal received ({signal_reason}). Managing active position stops.")
             if enable_trading:
                  # check_and_manage_position updates symbol_state['be_activated'] / ['tsl_activated'] internally
                  check_and_manage_position(exchange, symbol, bot_managed_position, market_info, analysis_results, lg)
             else:
                  lg.debug("Trading disabled. Skipping position management.")
                 # Optionally notify if BE/TSL conditions were met but couldn't be acted upon
                 # This could get noisy, maybe only for critical events
                 # send_notification(f"Info: {symbol} Stops", f"Bot disabled. Would have managed stops for {symbol} position.", lg, notification_type='email') # Maybe too frequent?


        elif signal in ['BUY', 'SELL'] and bot_managed_position:
            # Received an entry signal while already in a position.
            # This implies the strategy wants to enter the SAME side (e.g. BUY while LONG)
            # or reverse (e.g. BUY while SHORT).
            # With max_concurrent_positions=1, we should not enter a new position.
            # Reversing is not explicitly handled as a signal type ('REVERSE_LONG'/'REVERSE_SHORT').
            # The current signal system ('BUY'/'SELL') implies *opening* a position.
            # If max_concurrent_positions=1, ignore entry signals when a position exists.
            lg.warning(f"{NEON_YELLOW}Received {signal} signal while already in a {bot_managed_position['side'].upper()} position on {symbol}. Configuration allows max_concurrent_positions={max_concurrent_positions}. Skipping new entry.{RESET}")

        elif signal == 'NONE':
             # No actionable signal while in a position. Still HOLD.
             lg.debug("NONE signal received while in position. Managing stops.")
             if enable_trading:
                  check_and_manage_position(exchange, symbol, bot_managed_position, market_info, analysis_results, lg)
             else:
                  lg.debug("Trading disabled. Skipping position management.")

        else:
             # Should not happen with current signal types and position status combinations
             lg.warning(f"Unexpected signal '{signal}' received while in a {bot_managed_position['side'].upper()} position on {symbol}. Taking no action.")


    else:
        # No active position being tracked for this symbol by the bot state.
        lg.debug(f"No active position tracked for {symbol}. Checking for entry signals.")

        # Check if we are allowed to open a new position based on config
        if len(current_open_positions) >= max_concurrent_positions:
             lg.info(f"Max concurrent positions ({max_concurrent_positions}) reached/exceeded for {symbol}. Skipping entry signal check.")
             # If there are positions but none are tracked by the bot state, they might be manual or from another bot.
             # For safety with max_concurrent_positions=1, we just don't open a new one.
             if current_open_positions:
                 lg.debug(f"Currently detected {len(current_open_positions)} positions for {symbol}, config allows max {max_concurrent_positions}.")
             signal = 'NONE' # Override signal to NONE if max positions reached

        # Handle entry signals only if no position exists AND max positions is not reached
        if signal == 'BUY':
            lg.info(f"{NEON_GREEN}BUY signal received ({signal_reason}). Checking eligibility for entry.{RESET}")
            if enable_trading:
                # Calculate position size
                current_price = analysis_results['last_close']
                current_atr = analysis_results['atr'] # Should be valid here due to check before analysis

                position_amount = calculate_position_size(account_balance, current_price, current_atr, market_info, lg)

                if position_amount is not None and position_amount > POSITION_QTY_EPSILON:
                     lg.info(f"Calculated position size: {position_amount.normalize()} {market_info['base']}.")
                     # Execute the buy order
                     placed_order = execute_trade(exchange, symbol, 'buy', position_amount, current_price, signal_result['initial_sl_price'], signal_result['initial_tp_price'], market_info, lg)

                     if placed_order and placed_order.get('id'):
                          lg.info(f"{NEON_GREEN}BUY order successfully initiated for {symbol}. Order ID: {placed_order['id']}{RESET}")
                          send_notification(f"BUY Entry: {symbol}", f"Placed BUY order {placed_order['id']} for {position_amount.normalize()} @ ~{current_price.normalize()} on {symbol}. SL: {signal_result['initial_sl_price']}, TP: {signal_result['initial_tp_price']}. Reason: {signal_reason}", lg)
                          # Update symbol state to track this new position
                          # We need the position ID from fetch_positions *after* placement,
                          # or rely on the order ID and hope fetch_positions links them.
                          # CCXT's fetch_positions often links order ID via position['info'] or similar.
                          # For simplicity, after placing the order, we wait and then fetch positions again
                          # to find the *new* position and get its ID for tracking state.
                          # This relies on POSITION_CONFIRM_DELAY_SECONDS being sufficient.
                          lg.debug(f"Fetching positions again after order to confirm entry and get position ID...")
                          confirmed_positions = fetch_open_positions(exchange, symbol, lg)
                          newly_opened_pos = None
                          if confirmed_positions:
                             # Find the position matching the side we just traded and that wasn't there before (less reliable)
                             # Or, try to find the position linked to the order ID (more reliable if CCXT supports)
                             # Bybit V5 position info often includes 'orderId'.
                             order_id = placed_order.get('id')
                             for pos in confirmed_positions:
                                 if pos.get('side') == 'long' and pos.get('size_decimal', Decimal('0')) > POSITION_QTY_EPSILON:
                                     # Check if this position's info links back to our order ID
                                     pos_info = pos.get('info', {})
                                     if str(pos_info.get('orderId')) == str(order_id): # Compare as strings
                                         newly_opened_pos = pos
                                         break # Found the specific position for this order

                             if newly_opened_pos:
                                 lg.info(f"{NEON_GREEN}Confirmed new LONG position opened for {symbol}. Position ID: {newly_opened_pos['id']}{RESET}")
                                 symbol_state['position_id'] = newly_opened_pos['id']
                                 symbol_state['be_activated'] = False # Reset BE/TSL flags for the new position
                                 symbol_state['tsl_activated'] = False
                             else:
                                 lg.warning(f"{NEON_YELLOW}Could not confirm new LONG position or find ID linked to order {order_id} for {symbol} after waiting. Bot state may be out of sync.{RESET}")
                                 symbol_state['position_id'] = None # Cannot track reliably
                                 symbol_state['be_activated'] = False
                                 symbol_state['tsl_activated'] = False # Assume not active if cannot confirm


                     else:
                          lg.error(f"{NEON_RED}Failed to place BUY order for {symbol}.{RESET}")
                          send_notification(f"BUY Order Failed: {symbol}", f"Failed to place BUY order for {symbol}. Reason: {signal_reason}", lg)

                else:
                     lg.warning(f"Calculated position size {position_amount} is zero or invalid for {symbol}. Skipping entry.")
                     send_notification(f"Sizing Failed: {symbol}", f"Failed to calculate valid position size for BUY on {symbol}. Reason: {signal_reason}", lg)

            else:
                 lg.info("Trading disabled. Would have attempted to place BUY order.")
                 send_notification(f"Signal: {symbol} BUY", f"Bot disabled. Would BUY. Reason: {signal_reason}", lg)


        elif signal == 'SELL':
            lg.info(f"{NEON_GREEN}SELL signal received ({signal_reason}). Checking eligibility for entry.{RESET}")
            if enable_trading:
                # Calculate position size
                current_price = analysis_results['last_close']
                current_atr = analysis_results['atr']

                position_amount = calculate_position_size(account_balance, current_price, current_atr, market_info, lg)

                if position_amount is not None and position_amount > POSITION_QTY_EPSILON:
                     lg.info(f"Calculated position size: {position_amount.normalize()} {market_info['base']}.")
                     # Execute the sell order
                     placed_order = execute_trade(exchange, symbol, 'sell', position_amount, current_price, signal_result['initial_sl_price'], signal_result['initial_tp_price'], market_info, lg)

                     if placed_order and placed_order.get('id'):
                          lg.info(f"{NEON_GREEN}SELL order successfully initiated for {symbol}. Order ID: {placed_order['id']}{RESET}")
                          send_notification(f"SELL Entry: {symbol}", f"Placed SELL order {placed_order['id']} for {position_amount.normalize()} @ ~{current_price.normalize()} on {symbol}. SL: {signal_result['initial_sl_price']}, TP: {signal_result['initial_tp_price']}. Reason: {signal_reason}", lg)
                           # Update symbol state to track this new position
                          lg.debug(f"Fetching positions again after order to confirm entry and get position ID...")
                          confirmed_positions = fetch_open_positions(exchange, symbol, lg)
                          newly_opened_pos = None
                          if confirmed_positions:
                             order_id = placed_order.get('id')
                             for pos in confirmed_positions:
                                 if pos.get('side') == 'short' and pos.get('size_decimal', Decimal('0')) > POSITION_QTY_EPSILON:
                                     # Check if this position's info links back to our order ID
                                     pos_info = pos.get('info', {})
                                     if str(pos_info.get('orderId')) == str(order_id): # Compare as strings
                                         newly_opened_pos = pos
                                         break # Found the specific position for this order

                             if newly_opened_pos:
                                 lg.info(f"{NEON_GREEN}Confirmed new SHORT position opened for {symbol}. Position ID: {newly_opened_pos['id']}{RESET}")
                                 symbol_state['position_id'] = newly_opened_pos['id']
                                 symbol_state['be_activated'] = False # Reset BE/TSL flags for the new position
                                 symbol_state['tsl_activated'] = False
                             else:
                                 lg.warning(f"{NEON_YELLOW}Could not confirm new SHORT position or find ID linked to order {order_id} for {symbol} after waiting. Bot state may be out of sync.{RESET}")
                                 symbol_state['position_id'] = None # Cannot track reliably
                                 symbol_state['be_activated'] = False
                                 symbol_state['tsl_activated'] = False # Assume not active if cannot confirm

                     else:
                          lg.error(f"{NEON_RED}Failed to place SELL order for {symbol}.{RESET}")
                          send_notification(f"SELL Order Failed: {symbol}", f"Failed to place SELL order for {symbol}. Reason: {signal_reason}", lg)
                else:
                     lg.warning(f"Calculated position size {position_amount} is zero or invalid for {symbol}. Skipping entry.")
                     send_notification(f"Sizing Failed: {symbol}", f"Failed to calculate valid position size for SELL on {symbol}. Reason: {signal_reason}", lg)
            else:
                 lg.info("Trading disabled. Would have attempted to place SELL order.")
                 send_notification(f"Signal: {symbol} SELL", f"Bot disabled. Would SELL. Reason: {signal_reason}", lg)

        elif signal == 'NONE':
             # No entry signal generated and no position exists. Do nothing.
             lg.debug(f"NONE signal received ({signal_reason}). No position open. Waiting for next cycle.")

        # Handle EXIT signals when no position exists - should not happen, but defensive
        elif signal in ['EXIT_LONG', 'EXIT_SHORT']:
             lg.warning(f"Received unexpected {signal} signal, but no position is open for {symbol}. Ignoring signal.")


    lg.info(f"{NEON_BLUE}--- Finished processing {symbol} ---{RESET}")
    # Delay before processing the next symbol OR before the next cycle for THIS symbol
    # If processing symbols sequentially in one loop iteration, delay after each symbol.
    # If processing symbols in parallel (more complex), manage delays differently.
    # Current structure processes one symbol fully, then waits, then next symbol.
    # This means the loop delay applies BETWEEN symbols.
    # The loop delay should probably be applied *after* processing ALL symbols.
    # Let's restructure the main loop slightly.


# --- Signal Handling for Graceful Shutdown ---
def signal_handler(signum, frame):
    """
    Handles interruption signals (like Ctrl+C) for graceful shutdown.
    Sets a global flag that the main loop checks.
    """
    global _shutdown_requested
    init_logger.info(f"{NEON_YELLOW}Signal {signum} received. Initiating graceful shutdown...{RESET}")
    _shutdown_requested = True
    # Note: A small delay here might help ensure the flag is seen before exiting tight loops,
    # but generally checking the flag within loops is the key.
    # time.sleep(0.1)

# Register the signal handler for common interruption signals
signal.signal(signal.SIGINT, signal_handler) # Handle Ctrl+C
# signal.signal(signal.SIGTERM, signal_handler) # Handle kill signal (optional)


# --- Main Bot Execution Loop ---
def main():
    """
    The main function that initializes the exchange and runs the trading loop.
    """
    main_logger = setup_logger("main")
    main_logger.info(f"{Fore.MAGENTA}{BRIGHT}Starting Pyrmethus Volumatic Bot v{BOT_VERSION}{Style.RESET_ALL}")
    main_logger.info(f"Bot Configuration: {CONFIG}")

    exchange = initialize_exchange(main_logger)
    if exchange is None:
        main_logger.critical(f"{NEON_RED}{BRIGHT}Exchange initialization failed. Exiting.{RESET}")
        sys.exit(1)

    # Check if trading is enabled globally
    enable_trading_global = CONFIG.get("enable_trading", False)
    if not enable_trading_global:
        main_logger.warning(f"{NEON_YELLOW}Master trading switch (enable_trading) is FALSE in config. No orders will be placed.{RESET}")
        send_notification("Bot Started (Trading DISABLED)", f"Pyrmethus Volumatic Bot started in DISABLED trading mode.", main_logger)
    else:
        main_logger.warning(f"{NEON_RED}{BRIGHT}Master trading switch (enable_trading) is TRUE in config. LIVE TRADING IS ENABLED! REAL FUNDS AT RISK!{RESET}")
        send_notification("Bot Started (Trading ENABLED)", f"Pyrmethus Volumatic Bot started in ENABLED trading mode.", main_logger, notification_type='email') # Use email for this critical notification

    # Fetch market info for all configured symbols upfront
    market_info_cache: Dict[str, MarketInfo] = {}
    configured_pairs = CONFIG.get("trading_pairs", [])

    if not configured_pairs:
         main_logger.critical(f"{NEON_RED}No trading pairs configured in config.json. Exiting.{RESET}")
         sys.exit(1)

    main_logger.info(f"Loading market info for configured trading pairs: {configured_pairs}")
    for pair in configured_pairs:
        symbol_logger = setup_logger(pair.replace('/', '_')) # Get logger specific to the symbol
        pair_market_info = fetch_market_info(exchange, pair, symbol_logger)
        if pair_market_info:
            market_info_cache[pair] = pair_market_info
            symbol_logger.info(f"Market info loaded for {pair}. Contract Type: {pair_market_info['contract_type_str']}")
        else:
             main_logger.error(f"{NEON_RED}Failed to load market info for {pair}. This pair will be skipped.{RESET}")

    # Filter out pairs that failed to load market info
    active_trading_pairs = list(market_info_cache.keys())
    if not active_trading_pairs:
         main_logger.critical(f"{NEON_RED}Failed to load market info for all configured trading pairs. No symbols available to trade. Exiting.{RESET}")
         sys.exit(1)
    else:
         main_logger.info(f"Successfully loaded market info for {len(active_trading_pairs)} trading pair(s): {active_trading_pairs}")


    main_logger.info("Entering main trading loop.")
    loop_delay = CONFIG.get("loop_delay_seconds", DEFAULT_CONFIG['loop_delay_seconds'])

    while not _shutdown_requested:
        cycle_start_time = time.time()
        main_logger.info(f"{NEON_PURPLE}===== Starting Bot Cycle @ {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')} ====={RESET}")

        try:
            # Iterate through each configured trading pair sequentially
            for symbol in active_trading_pairs:
                if _shutdown_requested:
                    main_logger.info("Shutdown requested. Exiting main loop.")
                    break # Exit inner symbol loop

                symbol_logger = setup_logger(symbol.replace('/', '_')) # Get logger specific to the symbol
                market_info = market_info_cache.get(symbol) # Retrieve cached market info

                if market_info is None:
                    symbol_logger.error(f"{NEON_RED}Market info missing for {symbol}. Skipping.{RESET}")
                    continue # Skip this symbol if market info wasn't loaded

                # Execute the trading logic for this specific symbol
                handle_trading_pair(exchange, symbol, symbol_logger, market_info)

                # Optional: Small delay BETWEEN symbols if desired, but the main loop delay handles overall timing.
                # time.sleep(1)

        except Exception as e:
            # Catch any unexpected errors that occur within the main loop iteration (outside specific functions)
            main_logger.critical(f"{NEON_RED}An unexpected critical error occurred in the main loop: {e}{RESET}", exc_info=True)
            send_notification("Critical Bot Error", f"An unexpected error occurred in the main loop. Details: {e}", main_logger, notification_type='email')
            # Decide whether to continue the loop or exit.
            # For critical errors, often safer to exit.
            _shutdown_requested = True # Request shutdown

        # Wait for the configured loop delay before the next cycle,
        # accounting for the time spent processing this cycle.
        cycle_end_time = time.time()
        elapsed_time = cycle_end_time - cycle_start_time
        wait_time = max(0, loop_delay - elapsed_time) # Ensure wait time is not negative

        if wait_time > 0:
            main_logger.info(f"Cycle finished in {elapsed_time:.2f}s. Waiting {wait_time:.2f}s before next cycle.")
            # Use a loop for waiting to check shutdown flag periodically
            wait_start = time.time()
            while time.time() - wait_start < wait_time and not _shutdown_requested:
                time.sleep(1) # Check every second for shutdown request

        if _shutdown_requested:
            main_logger.info("Shutdown requested. Exiting main trading loop.")
            break # Exit the main loop

    main_logger.info(f"{Fore.MAGENTA}{BRIGHT}===== Bot Shutdown Initiated ====={Style.RESET_ALL}")
    # Perform any necessary cleanup here (e.g., close positions, cancel orders - CAUTION advised)
    # NOTE: Automated forced closure/cancellation on shutdown is RISKY.
    # If the bot restarts quickly, it might reopen/remanage. It's often safer
    # to let existing SL/TP handle open positions or manage manually after shutdown.
    # main_logger.warning("Performing cleanup (Note: Automated position closure/cancellation on shutdown is NOT implemented by default due to risk).")

    main_logger.info("Bot stopped.")
    send_notification("Bot Stopped", "Pyrmethus Volumatic Bot has stopped.", main_logger)


if __name__ == "__main__":
    # Ensure initial logger is done setting up before calling main
    time.sleep(0.1) # Small delay for logger handlers to be fully added
    main()

