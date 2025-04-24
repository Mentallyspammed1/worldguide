#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pyrmethus_volumatic_bot.py
# Unified and Enhanced Trading Bot incorporating Volumatic Trend, Pivot Order Blocks,
# and advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Merged and improved from multiple source files.
# Version 1.5.0: Consolidated config, logging, types, strategy structure, trading logic, and notifications.

"""
Pyrmethus Volumatic Bot: A Python Trading Bot for Bybit V5

This bot implements a trading strategy based on the combination of:
1.  **Volumatic Trend:** An EMA/SWMA crossover system with ATR-based bands,
    incorporating normalized volume analysis.
2.  **Pivot Order Blocks (OBs):** Identifying potential support/resistance zones
    based on pivot highs and lows derived from candle wicks or bodies.

This version synthesizes features and robustness from previous iterations, including:
-   Robust configuration loading from both .env (secrets) and config.json (parameters).
-   Detailed configuration validation with automatic correction to defaults.
-   Flexible notification options (Termux SMS and Email).
-   Enhanced logging with colorama, rotation, and sensitive data redaction.
-   Comprehensive API interaction handling with retries and error logging for CCXT.
-   Accurate Decimal usage for all financial calculations.
-   Structured data types using TypedDicts for clarity.
-   Implementation of native Bybit V5 Stop Loss, Take Profit, and Trailing Stop Loss.
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
import subprocess # Used for Termux SMS
import shutil     # Used to check for termux-sms-send
import smtplib    # Used for Email notifications
from email.mime.text import MIMEText # Used for Email notifications
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
        # Attempt to load a common non-UTC zone to check if tzdata is available
        # We don't need this specific zone, just need the check to pass
        _ = ZoneInfo("America/Chicago")
    except ZoneInfoNotFoundError:
        # Handle the case where zoneinfo is available but tzdata is not found
        print(f"{Fore.YELLOW}Warning: 'zoneinfo' is available, but 'tzdata' package seems missing or corrupt.")
        print("         `pip install tzdata` is recommended for reliable timezone support.")
        # Continue with zoneinfo, but it might fail for non-UTC zones requested by the user
    except Exception as tz_init_err:
         # Catch any other unexpected errors during ZoneInfo initialization
         print(f"{Fore.YELLOW}Warning: Error initializing test timezone with 'zoneinfo': {tz_init_err}")
         # Continue cautiously
except ImportError:
    # Fallback for older Python versions or if zoneinfo itself is not installed
    print(f"{Fore.YELLOW}Warning: 'zoneinfo' module not found (requires Python 3.9+). Falling back to basic UTC implementation.")
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
            # Standard ZoneInfo usually replaces tzinfo. Mimic that, converting to UTC if necessary.
            return dt.astimezone(timezone.utc)


        def fromutc(self, dt: datetime) -> datetime:
            """Converts a UTC datetime to this timezone (which is always UTC in the fallback)."""
            if not isinstance(dt, datetime):
                raise TypeError("fromutc() requires a datetime argument")
            if dt.tzinfo is None:
                # Standard library raises ValueError for naive datetime in fromutc
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
BOT_VERSION = "1.5.0"

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
# This is a common constraint on Bybit V5 for USDT perpetuals.
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
    stopLossPrice_raw: Optional[str] # Current stop loss price set on the exchange (as string, often '0' or '0.0' if not set)
    takeProfitPrice_raw: Optional[str] # Current take profit price set on the exchange (as string, often '0' or '0.0' if not set)
    # Bybit V5 'trailingStop' is the trigger price once active, not the distance/callback rate.
    # We need to check the raw info structure for the actual percentage or distance if available.
    # CCXT often maps 'trailingStop' to the trigger price in its unified position structure.
    trailingStopPrice_raw: Optional[str] # Current trailing stop trigger price (as string, e.g., Bybit V5 'trailingStop')
    tslActivationPrice_raw: Optional[str] # Trailing stop activation price (as string, if available/set, e.g., Bybit V5 'activePrice')
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
         logger.debug(f"Logger '{logger_name}' initialized. File Handler: {fh.level if 'fh' in locals() else 'N/A'}, Console Handler: {sh.level if 'sh' in locals() else 'N/A'}")
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
def send_notification(subject: str, body: str, logger: logging.Logger, notification_type: str = "email") -> bool:
    """
    Sends a notification via email or Termux SMS if configured and enabled.

    Args:
        subject (str): The subject line for the notification (used for email and prefixed to SMS).
        body (str): The main content of the notification.
        logger (logging.Logger): Logger instance to use for logging notification status/errors.
        notification_type (str): 'email' or 'sms'. Determines the method used.

    Returns:
        bool: True if the notification was attempted successfully (not necessarily received), False otherwise.
    """
    lg = logger # Alias for convenience

    # Check master notification enable/disable from config (assuming CONFIG is loaded globally)
    # This check should happen before calling this function, but added defensively here.
    if 'CONFIG' not in globals() or not CONFIG.get("notifications", {}).get("enable_notifications", False):
        lg.debug(f"Notifications are disabled by config. Skipping '{subject}'.")
        return False

    if notification_type.lower() == "email":
        # Check if email settings are complete from .env
        if not all([SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_EMAIL_RECIPIENT]):
            lg.warning("Email notification is enabled but settings are incomplete (SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_EMAIL_RECIPIENT env vars).")
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

    elif notification_type.lower() == "sms":
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
        lg.error(f"{NEON_RED}Invalid notification_type specified: '{notification_type}'. Must be 'email' or 'sms'.{RESET}")
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

def _validate_and_correct_numeric(cfg: Dict, key_path: str, min_val: Union[int, float, Decimal], max_val: Union[int, float, Decimal],
                             is_strict_min: bool = False, is_int: bool = False, allow_zero: bool = False) -> bool:
    """
    Validates a numeric config value at `key_path` (e.g., "protection.leverage").

    Checks type (int/float/str numeric), range [min_val, max_val] or (min_val, max_val] if strict.
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
    # Access the default config loaded globally
    global DEFAULT_CONFIG # Access the global default config structure

    nonlocal config_needs_saving # Allow modification of the outer scope variable
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
        return False # Path itself is wrong, cannot proceed with validation for this key

    if original_val is None:
        # This case should theoretically be handled by _ensure_config_keys before validation,
        # but check defensively.
        init_logger.warning(f"{NEON_YELLOW}Config validation: Key missing at '{key_path}' during numeric check. Using default: {repr(default_val)}{RESET}")
        current_level[leaf_key] = default_val
        config_needs_saving = True
        return True

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
            if isinstance(original_val, (float, int)):
                 converted_float = float(num_val) # Convert validated Decimal to float
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
                 final_val = float(num_val) # Convert validated Decimal to float
                 init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type for float key '{key_path}' from {type(original_val).__name__} to float (value: {repr(final_val)}).{RESET}")
            else: # Should technically be covered by the above cases for float/int, but defensive
                 final_val = float(num_val)


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
        # Need to traverse the path again to update the correct nested dictionary
        current_level = cfg
        for key in keys[:-1]:
            current_level = current_level[key]
        current_level[leaf_key] = final_val
        config_needs_saving = True # Mark the configuration as needing to be saved


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
    global DEFAULT_CONFIG # Make the default config accessible globally for validation helper

    init_logger.info(f"{Fore.CYAN}# Loading configuration from '{filepath}'...{Style.RESET_ALL}")

    # --- Define Default Configuration Structure ---
    # This dictionary serves as the template and provides default values for missing keys
    # and fallback values during validation.
    DEFAULT_CONFIG = {
        # == Trading Core ==
        "trading_pairs": ["BTC/USDT"],          # List of market symbols to trade (e.g., ["BTC/USDT", "ETH/USDT"])
        "interval": "5",                        # Kline timeframe (must be one of VALID_INTERVALS strings)
        "enable_trading": False,                # Master switch: MUST BE true for live order placement. Safety default: false.
        "use_sandbox": True,                    # Use exchange's sandbox/testnet environment? Safety default: true.
        "quote_currency": "USDT",               # Primary currency for balance, PnL, risk calculations (e.g., USDT, BUSD). Case-sensitive.
        "max_concurrent_positions": 1,          # Maximum number of positions allowed open simultaneously across all pairs (Currently per symbol due to simple loop).

        # == Risk & Sizing ==
        "risk_per_trade": 0.01,                 # Fraction of available balance to risk per trade (e.g., 0.01 = 1%). Must be > 0.0 and <= 1.0.
        "leverage": 20,                         # Desired leverage for contract trading (integer). 0 or 1 typically means spot/no leverage. Exchange limits apply.

        # == API & Timing ==
        "retry_delay": RETRY_DELAY_SECONDS,             # Base delay in seconds between API retry attempts (integer)
        "loop_delay_seconds": LOOP_DELAY_SECONDS,       # Delay in seconds between processing cycles for each symbol (integer)
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Delay after placing order before checking position status (integer)

        # == Data Fetching ==
        "fetch_limit": DEFAULT_FETCH_LIMIT,             # Default number of historical klines to fetch (integer)
        "orderbook_limit": 25,                          # (Currently Unused) Limit for order book depth fetching (integer, if feature implemented later)

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
            "enable_trailing_stop": DEFAULT_ENABLE_TRAILING_STOP, # Enable Trailing Stop Loss? (boolean)
            "trailing_stop_callback_rate": DEFAULT_TRAILING_STOP_CALLBACK_RATE, # TSL callback/distance (float > 0). Interpretation depends on exchange/implementation (e.g., 0.005 = 0.5% or 0.5 price points).
            "trailing_stop_activation_percentage": DEFAULT_TRAILING_STOP_ACTIVATION_PERCENTAGE, # Activate TSL when price moves this % from entry (float >= 0).
            "enable_break_even": DEFAULT_ENABLE_BREAK_EVEN,         # Enable moving SL to break-even? (boolean)
            "break_even_trigger_atr_multiple": DEFAULT_BREAK_EVEN_TRIGGER_ATR_MULTIPLE, # Move SL to BE when price moves ATR * multiple in profit (float > 0)
            "break_even_offset_ticks": DEFAULT_BREAK_EVEN_OFFSET_TICKS, # Offset SL from entry by this many price ticks for BE (integer >= 0)
             # -- Initial SL/TP (often ATR-based) --
            "initial_stop_loss_atr_multiple": DEFAULT_INITIAL_STOP_LOSS_ATR_MULTIPLE, # Initial SL distance = ATR * this multiple (float > 0)
            "initial_take_profit_atr_multiple": DEFAULT_INITIAL_TAKE_PROFIT_ATR_MULTIPLE # Initial TP distance = ATR * this multiple (float >= 0, 0 means no initial TP)
        },

        # == Notifications ==
        "notifications": {
            "enable_notifications": True, # Master switch for email notifications
            "notification_type": "email", # 'email' or 'sms'
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
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully created default config file: {filepath}{RESET}")
            # Use the defaults since the file was just created
            loaded_config = DEFAULT_CONFIG
            config_needs_saving = False # No need to save again immediately
            # Update global QUOTE_CURRENCY from the default we just wrote
            QUOTE_CURRENCY = DEFAULT_CONFIG.get("quote_currency", "USDT")
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return DEFAULT_CONFIG # Return defaults directly as the file is now correct

        except IOError as e:
            init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Could not create config file '{filepath}': {e}.{RESET}")
            init_logger.critical(f"{NEON_RED}Please check directory permissions. Using internal defaults as fallback.{RESET}")
            # Fallback to using internal defaults in memory if file creation fails
            QUOTE_CURRENCY = DEFAULT_CONFIG.get("quote_currency", "USDT")
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return DEFAULT_CONFIG

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
                json.dump(DEFAULT_CONFIG, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Successfully recreated default config file: {filepath}{RESET}")
            QUOTE_CURRENCY = DEFAULT_CONFIG.get("quote_currency", "USDT")
            init_logger.info(f"Using default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return DEFAULT_CONFIG # Return the defaults
        except IOError as e_create:
            init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Error recreating config file after corruption: {e_create}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = DEFAULT_CONFIG.get("quote_currency", "USDT")
            init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
            return DEFAULT_CONFIG # Fallback to internal defaults
    except Exception as e:
        # Catch any other unexpected errors during file loading or initial parsing
        init_logger.critical(f"{NEON_RED}{BRIGHT}FATAL ERROR: Unexpected error loading config file '{filepath}': {e}{RESET}", exc_info=True)
        init_logger.critical(f"{NEON_RED}Using internal defaults as fallback.{RESET}")
        QUOTE_CURRENCY = DEFAULT_CONFIG.get("quote_currency", "USDT")
        init_logger.info(f"Using internal default quote currency: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        return DEFAULT_CONFIG # Fallback to internal defaults

    # --- Step 3: Ensure Keys and Validate Parameters ---
    try:
        # Ensure all default keys exist in the loaded config, add missing ones
        updated_config, keys_added = _ensure_config_keys(loaded_config, DEFAULT_CONFIG)
        if keys_added:
            config_needs_saving = True # Mark for saving later if keys were added

        # --- Validation Logic ---
        init_logger.debug("Starting configuration parameter validation...")

        # Helper function to validate boolean keys (exists and is bool)
        def validate_boolean(cfg_level: Dict[str, Any], default_level: Dict[str, Any], leaf_key: str, key_path: str) -> bool:
             nonlocal config_needs_saving
             original_val = cfg_level.get(leaf_key)
             default_val = default_level.get(leaf_key)
             if not isinstance(original_val, bool):
                 init_logger.warning(f"{NEON_YELLOW}Config Warning: '{key_path}' must be true or false. Provided: {repr(original_val)} (Type: {type(original_val).__name__}). Using default: {repr(default_val)}{RESET}")
                 cfg_level[leaf_key] = default_val
                 config_needs_saving = True
                 return True
             return False # No correction needed

        # Helper function to validate string choice keys
        def validate_string_choice(cfg_level: Dict[str, Any], default_level: Dict[str, Any], leaf_key: str, key_path: str, valid_choices: List[str]) -> bool:
             nonlocal config_needs_saving
             original_val = cfg_level.get(leaf_key)
             default_val = default_level.get(leaf_key)
             if not isinstance(original_val, str) or original_val not in valid_choices:
                 init_logger.warning(f"{NEON_YELLOW}Config Warning: '{key_path}' ('{original_val}') is invalid. Must be one of {valid_choices}. Using default: '{default_val}'{RESET}")
                 cfg_level[leaf_key] = default_val
                 config_needs_saving = True
                 return True
             return False # No correction needed


        # --- Apply Validations to Specific Config Keys ---
        changes = [] # List to track if any validation corrections occurred

        # General
        # Note: max_concurrent_positions is not strictly enforced per symbol in the current simple loop structure
        changes.append(_validate_and_correct_numeric(updated_config, "max_concurrent_positions", 1, 100, is_int=True))
        # Risk per trade must be > 0% and <= 100% (Decimal comparison)
        changes.append(_validate_and_correct_numeric(updated_config, "risk_per_trade", Decimal('0'), Decimal('1'), is_strict_min=True))
        # Leverage >= 0 (0 means disable setting)
        changes.append(_validate_and_correct_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True))
        changes.append(_validate_and_correct_numeric(updated_config, "retry_delay", 1, 60, is_int=True))
        changes.append(_validate_and_correct_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True))
        changes.append(_validate_and_correct_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True))
        # Fetch limit should be reasonable and <= exchange limit (BYBIT_API_KLINE_LIMIT)
        changes.append(_validate_and_correct_numeric(updated_config, "fetch_limit", 50, BYBIT_API_KLINE_LIMIT, is_int=True))
        # Order book limit check (if used)
        changes.append(_validate_and_correct_numeric(updated_config, "orderbook_limit", 1, 1000, is_int=True))


        # Trading Pairs Validation (Basic existence and format check)
        trading_pairs_val = updated_config.get("trading_pairs")
        if not isinstance(trading_pairs_val, list) or not trading_pairs_val: # Must be a non-empty list
            init_logger.warning(f"{NEON_YELLOW}Config Warning: 'trading_pairs' value '{trading_pairs_val}' is invalid. Must be a non-empty list of strings. Using default: {DEFAULT_CONFIG['trading_pairs']}{RESET}")
            updated_config["trading_pairs"] = DEFAULT_CONFIG["trading_pairs"]
            config_needs_saving = True
        else:
             # Check each item is a non-empty string and contains '/' (basic symbol format)
             cleaned_pairs = [p for p in trading_pairs_val if isinstance(p, str) and p and '/' in p]
             if len(cleaned_pairs) != len(trading_pairs_val):
                  init_logger.warning(f"{NEON_YELLOW}Config Warning: Some 'trading_pairs' entries are invalid (not string or empty or missing '/'). Removed invalid entries. Original: {trading_pairs_val}{RESET}")
                  updated_config["trading_pairs"] = cleaned_pairs if cleaned_pairs else DEFAULT_CONFIG["trading_pairs"]
                  config_needs_saving = True
             elif not cleaned_pairs: # If after cleaning, the list is empty
                  init_logger.warning(f"{NEON_YELLOW}Config Warning: 'trading_pairs' contains only invalid entries. Using default: {DEFAULT_CONFIG['trading_pairs']}{RESET}")
                  updated_config["trading_pairs"] = DEFAULT_CONFIG["trading_pairs"]
                  config_needs_saving = True


        # Interval Validation
        interval_val = str(updated_config.get("interval")) # Ensure string comparison
        if interval_val not in VALID_INTERVALS:
            init_logger.warning(f"{NEON_YELLOW}Config Warning: 'interval' ('{interval_val}') is invalid. Must be one of {VALID_INTERVALS}. Using default: '{DEFAULT_CONFIG['interval']}'{RESET}")
            updated_config["interval"] = DEFAULT_CONFIG["interval"]
            config_needs_saving = True

        # Boolean Validations (Top Level)
        changes.append(validate_boolean(updated_config, DEFAULT_CONFIG, "enable_trading", "enable_trading"))
        changes.append(validate_boolean(updated_config, DEFAULT_CONFIG, "use_sandbox", "use_sandbox"))

        # Quote Currency Validation
        quote_currency_val = updated_config.get("quote_currency")
        if not isinstance(quote_currency_val, str) or not quote_currency_val.strip():
             init_logger.warning(f"{NEON_YELLOW}Config Warning: Invalid 'quote_currency' value '{quote_currency_val}'. Must be a non-empty string. Using default '{DEFAULT_CONFIG['quote_currency']}'.{RESET}")
             updated_config["quote_currency"] = DEFAULT_CONFIG["quote_currency"]
             config_needs_saving = True
        else:
             # Store uppercase for consistency
             updated_config["quote_currency"] = quote_currency_val.strip().upper()
             if updated_config["quote_currency"] != quote_currency_val.strip():
                  init_logger.info(f"{NEON_YELLOW}Config Update: Capitalized 'quote_currency' to '{updated_config['quote_currency']}'.{RESET}")
                  config_needs_saving = True


        # Strategy Params (Assuming the nested structure exists due to _ensure_config_keys)
        sp = updated_config["strategy_params"]
        def_sp = DEFAULT_CONFIG["strategy_params"] # Use default nested dict for validation
        changes.append(_validate_and_correct_numeric(sp, "vt_length", 1, 500, is_int=True))
        changes.append(_validate_and_correct_numeric(sp, "vt_atr_period", 1, MAX_DF_LEN, is_int=True)) # Allow long ATR period
        changes.append(_validate_and_correct_numeric(sp, "vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True)) # Allow long Vol EMA
        changes.append(_validate_and_correct_numeric(sp, "vt_atr_multiplier", 0.1, 20.0))
        # vt_step_atr_multiplier is currently unused, minimal validation
        changes.append(_validate_and_correct_numeric(sp, "vt_step_atr_multiplier", 0.1, 20.0))
        changes.append(_validate_and_correct_numeric(sp, "ph_left", 1, 100, is_int=True))
        changes.append(_validate_and_correct_numeric(sp, "ph_right", 1, 100, is_int=True))
        changes.append(_validate_and_correct_numeric(sp, "pl_left", 1, 100, is_int=True))
        changes.append(_validate_and_correct_numeric(sp, "pl_right", 1, 100, is_int=True))
        changes.append(_validate_and_correct_numeric(sp, "ob_max_boxes", 1, 200, is_int=True))
        # Proximity factors must be >= 1.0 (or slightly more for float comparison safety)
        changes.append(_validate_and_correct_numeric(sp, "ob_entry_proximity_factor", 1.0, 1.1)) # e.g., 1.005 = 0.5% proximity
        changes.append(_validate_and_correct_numeric(sp, "ob_exit_proximity_factor", 1.0, 1.1)) # e.g., 1.001 = 0.1% proximity
        changes.append(validate_string_choice(sp, def_sp, "ob_source", "strategy_params.ob_source", ["Wicks", "Body"]))
        changes.append(validate_boolean(sp, def_sp, "ob_extend", "strategy_params.ob_extend"))


        # Protection Params (Assuming nested structure exists)
        pp = updated_config["protection"]
        def_pp = DEFAULT_CONFIG["protection"] # Use default nested dict for validation
        changes.append(validate_boolean(pp, def_pp, "enable_trailing_stop", "protection.enable_trailing_stop"))
        changes.append(validate_boolean(pp, def_pp, "enable_break_even", "protection.enable_break_even"))
        # Callback rate > 0
        changes.append(_validate_and_correct_numeric(pp, "trailing_stop_callback_rate", Decimal('0.0001'), Decimal('0.5'), is_strict_min=True)) # e.g., 0.01% to 50%
        # Activation percentage >= 0 (0 means activate immediately)
        changes.append(_validate_and_correct_numeric(pp, "trailing_stop_activation_percentage", Decimal('0'), Decimal('0.5'), allow_zero=True))
        # Break even trigger ATR multiple > 0
        changes.append(_validate_and_correct_numeric(pp, "break_even_trigger_atr_multiple", Decimal('0.1'), Decimal('10.0')))
        # Break even offset ticks >= 0 (0 means move SL exactly to entry)
        changes.append(_validate_and_correct_numeric(pp, "break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True))
        # Initial SL ATR multiple > 0
        changes.append(_validate_and_correct_numeric(pp, "initial_stop_loss_atr_multiple", Decimal('0.1'), Decimal('100.0'), is_strict_min=True))
        # Initial TP ATR multiple >= 0 (0 disables initial TP)
        changes.append(_validate_and_correct_numeric(pp, "initial_take_profit_atr_multiple", Decimal('0'), Decimal('100.0'), allow_zero=True))


        # Notifications (Assuming nested structure exists)
        np_cfg = updated_config["notifications"]
        def_np = DEFAULT_CONFIG["notifications"] # Use default nested dict for validation
        changes.append(validate_boolean(np_cfg, def_np, "enable_notifications", "notifications.enable_notifications"))
        changes.append(validate_string_choice(np_cfg, def_np, "notification_type", "notifications.notification_type", ["email", "sms"]))
        changes.append(_validate_and_correct_numeric(np_cfg, "sms_timeout_seconds", 5, 120, is_int=True))


        # Backtesting (Placeholder)
        bp = updated_config["backtesting"]
        def_bp = DEFAULT_CONFIG["backtesting"] # Use default nested dict for validation
        changes.append(validate_boolean(bp, def_bp, "enabled", "backtesting.enabled"))
        # Basic date format check YYYY-MM-DD
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for date_key in ["start_date", "end_date"]:
            date_val = bp.get(date_key)
            if not isinstance(date_val, str) or not date_pattern.match(date_val):
                 init_logger.warning(f"{NEON_YELLOW}Config Warning: 'backtesting.{date_key}' ('{date_val}') is invalid. Expected YYYY-MM-DD format. Using default: '{def_bp[date_key]}'{RESET}")
                 bp[date_key] = def_bp[date_key]
                 config_needs_saving = True
            # Optional: Add logic to check start_date < end_date if needed


        # --- Check if any validation corrections were made ---
        if any(changes):
            config_needs_saving = True


        # --- Save Updated Config if Necessary ---
        if config_needs_saving:
             init_logger.info("Configuration has been updated or corrected.")
             try:
                 # Convert potentially corrected values back to standard types for JSON
                 # (e.g., if defaults were used which might be Decimals internally, convert to float)
                 # json.dumps handles basic types (int, float, str, bool, list, dict)
                 # Need to ensure Decimal objects are converted to float for JSON serialization.
                 # A simple way is to traverse the dict and convert Decimals.
                 def convert_decimals_to_float(obj):
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
                 init_logger.info(f"{NEON_GREEN}Saved updated configuration with defaults/corrections to: {filepath}{RESET}")
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
        return DEFAULT_CONFIG # Fallback to defaults on unexpected error

# --- Load Global Configuration ---
# This loads the configuration, performs validation, and updates the global QUOTE_CURRENCY
CONFIG = load_config(CONFIG_FILE)
# DEFAULT_CONFIG is also now globally available from within load_config scope
# QUOTE_CURRENCY is updated inside load_config()

# --- Utility Functions ---
def _safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0"), allow_none: bool = False) -> Optional[Decimal]:
    """
    Safely converts a value to a Decimal, handling None, pandas NA, and potential errors.

    Args:
        value: The value to convert (can be string, float, int, Decimal, None, pandas NA, etc.).
        default: The Decimal value to return if conversion fails or input is None/NA.
        allow_none: If True and input value is None/NA, returns None instead of default.

    Returns:
        The converted Decimal value, the default, or None if allow_none is True and input is None/NA.
    """
    # Use pandas.isna to handle numpy.nan, None, and pandas.NA
    if pd.isna(value) or value is None:
        return None if allow_none else default

    try:
        # Using str(value) handles various input types more reliably before Decimal conversion
        # Strip whitespace in case value is a string from env var or similar
        str_value = str(value).strip()
        if str_value == "": # Handle empty strings explicitly
             raise InvalidOperation("Cannot convert empty string to Decimal.")
        return Decimal(str_value)
    except (InvalidOperation, TypeError, ValueError) as e:
        # Log a warning, but only if the value was not None/NA initially
        # Use init_logger or get a logger instance here? Using init_logger for simplicity in helpers
        init_logger.warning(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}. Error: {e}")
        return default

def _safe_market_decimal(
    value: Any,
    field_name: str,
    allow_zero: bool = False,
    allow_negative: bool = False,
    default: Optional[Decimal] = None
) -> Optional[Decimal]:
    """
    Safely convert market data (limits, precision) to Decimal. More specific
    error logging for market data parsing.

    Args:
        value: The value from market info (float, int, str, None).
        field_name: Name of the field being parsed (for logging).
        allow_zero: If True, 0 is considered a valid value.
        allow_negative: If True, negative values are allowed.
        default: Default value to return on failure.

    Returns:
        The value as Decimal, or default on failure/invalid.
    """
    if value is None:
        return default
    try:
        # Handle potential string representations like 'inf' or scientific notation
        str_val = str(value).strip().lower()
        if str_val in ['inf', '+inf', '-inf', 'nan']:
             init_logger.warning(f"Market data parsing: Non-finite value '{value}' for {field_name}. Using default: {default}")
             return default

        dec_val = Decimal(str_val)

        if not dec_val.is_finite(): # Double check after conversion
            init_logger.warning(f"Market data parsing: Converted value for {field_name} is non-finite: {value}. Using default: {default}")
            return default
        if not allow_zero and dec_val.is_zero():
            # Log as debug because '0' might be a valid default or min cost
            init_logger.debug(f"Market data parsing: Zero value encountered for {field_name}: {value}. Returning default/None.")
            return default # Return default if zero isn't allowed
        if not allow_negative and dec_val < 0:
            init_logger.warning(f"Market data parsing: Negative value not allowed for {field_name}: {value}. Using default: {default}")
            return default

        # Quantize to a reasonable precision to avoid floating point issues from source
        return dec_val.quantize(Decimal('1e-15'), rounding=ROUND_DOWN)

    except (ValueError, TypeError, InvalidOperation) as e:
        init_logger.warning(f"Market data parsing: Invalid value for {field_name}: '{value}' ({type(value).__name__}) - Error: {e}. Using default: {default}")
        return default

def _format_price(exchange: ccxt.Exchange, market_info: MarketInfo, price: Decimal) -> Optional[str]:
    """
    Formats a price according to the market's precision rules using enhanced MarketInfo.

    Args:
        exchange: The CCXT exchange instance.
        market_info: Enhanced MarketInfo for the symbol.
        price: The price value (Decimal).

    Returns:
        The price formatted as a string according to market precision, or None on error.
    """
    if not isinstance(price, Decimal):
        price = _safe_decimal_conversion(price) # Attempt conversion if not Decimal

    if price is None or not price.is_finite():
         # Use init_logger or get a logger instance? Use init_logger for helpers for now.
         init_logger.error(f"Format Price: Cannot format non-finite or None price: {price}")
         return None

    symbol = market_info['symbol']
    price_step = market_info.get('price_precision_step_decimal')

    if price_step is None or not price_step.is_finite() or price_step <= POSITION_QTY_EPSILON:
        # Fallback to CCXT's price_to_precision if step is invalid or missing, or log error
        init_logger.warning(f"Format Price: Invalid or missing price precision step for {symbol} ({price_step}). Falling back to CCXT's price_to_precision.")
        try:
             # CCXT formatter expects float, need to be careful with precision loss here
             return exchange.price_to_precision(symbol, float(price))
        except Exception as e:
             init_logger.error(f"Format Price: CCXT fallback failed for {symbol}: {e}. Returning raw decimal string.", exc_info=True)
             return str(price.normalize()) # Last resort: raw decimal string
    try:
        # Quantize the price to the market's price step, typically ROUND_DOWN for stops/take profit
        # Or should it be ROUND_HALF_UP? Depends on desired behavior. ROUND_DOWN is safer for stops,
        # but might slightly reduce TP profit. Let's stick to ROUND_DOWN.
        formatted_price = price.quantize(price_step, rounding=ROUND_DOWN)
        return str(formatted_price.normalize()) # Normalize to remove trailing zeros if any
    except (ValueError, InvalidOperation) as e:
        init_logger.error(f"Format Price: Error quantizing price {price.normalize()} for {symbol} with step {price_step.normalize()}: {e}. Returning raw decimal string.", exc_info=True)
        return str(price.normalize()) # Fallback to raw decimal string

def _format_amount(exchange: ccxt.Exchange, market_info: MarketInfo, amount: Decimal) -> Optional[str]:
    """
    Formats an amount (quantity) according to the market's precision rules using enhanced MarketInfo.

    Args:
        exchange: The CCXT exchange instance.
        market_info: Enhanced MarketInfo for the symbol.
        amount: The amount value (Decimal).

    Returns:
        The amount formatted as a string according to market precision, or None on error.
    """
    if not isinstance(amount, Decimal):
        amount = _safe_decimal_conversion(amount) # Attempt conversion if not Decimal

    if amount is None or not amount.is_finite():
         init_logger.error(f"Format Amount: Cannot format non-finite or None amount: {amount}")
         return None

    symbol = market_info['symbol']
    amount_step = market_info.get('amount_precision_step_decimal')

    if amount_step is None or not amount_step.is_finite() or amount_step <= POSITION_QTY_EPSILON:
        # Fallback to CCXT's amount_to_precision if step is invalid or missing
        init_logger.warning(f"Format Amount: Invalid or missing amount precision step for {symbol} ({amount_step}). Falling back to CCXT's amount_to_precision.")
        try:
            # CCXT formatter expects float
            return exchange.amount_to_precision(symbol, float(amount))
        except Exception as e:
             init_logger.error(f"Format Amount: CCXT fallback failed for {symbol}: {e}. Returning raw decimal string.", exc_info=True)
             return str(amount.normalize()) # Last resort: raw decimal string
    try:
        # Quantize the amount to the market's amount step, typically ROUND_DOWN for buy, ROUND_UP for sell?
        # Or always ROUND_DOWN? Standard practice is often ROUND_DOWN to not exceed max amount or minimum steps.
        # Let's stick to ROUND_DOWN for calculated quantities.
        formatted_amount = amount.quantize(amount_step, rounding=ROUND_DOWN)
        # Ensure resulting quantity isn't zero if the input wasn't
        if formatted_amount.is_zero() and not amount.is_zero():
             init_logger.warning(f"Format Amount: Quantization resulted in zero ({formatted_amount}) for non-zero input ({amount.normalize()}) with step ({amount_step.normalize()}). This might indicate amount is less than step. Using raw decimal string.")
             return str(amount.normalize()) # Fallback if quantization results in zero
        return str(formatted_amount.normalize()) # Normalize to remove trailing zeros
    except (ValueError, InvalidOperation) as e:
        init_logger.error(f"Format Amount: Error quantizing amount {amount.normalize()} for {symbol} with step {amount_step.normalize()}: {e}. Returning raw decimal string.", exc_info=True)
        return str(amount.normalize()) # Fallback to raw decimal string


# --- Market and Position Data Enhancement ---
def enhance_market_info(market: Dict[str, Any]) -> MarketInfo:
    """
    Adds custom fields and Decimal types to ccxt market dict for easier access and calculation.

    Args:
        market (Dict[str, Any]): The raw market dictionary from ccxt.exchange.market().

    Returns:
        MarketInfo: An enhanced dictionary with parsed decimal values and convenience flags.
    """
    # Start with a copy of the original dictionary
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
    price_limits = limits.get('price', {}) # Added price limits check
    precision = market.get('precision', {})

    # Determine step size (tick size) from precision. CCXT precision can be integer (decimal places) or float (tick size)
    amount_step = None
    amount_prec_val = precision.get('amount')
    if isinstance(amount_prec_val, int): amount_step = Decimal('1e-' + str(amount_prec_val))
    elif amount_prec_val is not None: amount_step = _safe_market_decimal(amount_prec_val, f"{market['symbol']}.precision.amount", allow_zero=False)

    price_step = None
    price_prec_val = precision.get('price')
    if isinstance(price_prec_val, int): price_step = Decimal('1e-' + str(price_prec_val))
    elif price_prec_val is not None: price_step = _safe_market_decimal(price_prec_val, f"{market['symbol']}.precision.price", allow_zero=False)

    contract_size = market.get('contractSize', 1.0) # Default to 1.0 if not specified

    # Assign enhanced fields - using setdefault is often cleaner for TypedDicts
    enhanced.setdefault('is_contract', is_contract)
    enhanced.setdefault('is_linear', is_linear)
    enhanced.setdefault('is_inverse', is_inverse)
    enhanced.setdefault('contract_type_str', contract_type)

    # Safely parse limits using the helper, providing field names for logging
    enhanced.setdefault('min_amount_decimal', _safe_market_decimal(amount_limits.get('min'), f"{market['symbol']}.limits.amount.min", allow_zero=True))
    enhanced.setdefault('max_amount_decimal', _safe_market_decimal(amount_limits.get('max'), f"{market['symbol']}.limits.amount.max"))
    enhanced.setdefault('min_cost_decimal', _safe_market_decimal(cost_limits.get('min'), f"{market['symbol']}.limits.cost.min", allow_zero=True))
    enhanced.setdefault('max_cost_decimal', _safe_market_decimal(cost_limits.get('max'), f"{market['symbol']}.limits.cost.max"))
    enhanced.setdefault('min_price_decimal', _safe_market_decimal(price_limits.get('min'), f"{market['symbol']}.limits.price.min", allow_zero=True)) # Added price limits
    enhanced.setdefault('max_price_decimal', _safe_market_decimal(price_limits.get('max'), f"{market['symbol']}.limits.price.max")) # Added price limits


    enhanced.setdefault('amount_precision_step_decimal', amount_step)
    enhanced.setdefault('price_precision_step_decimal', price_step)

    # Contract size might be missing or None for spot markets, default to 1.0 (Decimal)
    enhanced.setdefault('contract_size_decimal', _safe_market_decimal(contract_size, f"{market['symbol']}.contractSize", default=Decimal('1.0'), allow_zero=True)) # Allow 0 contract size? Probably not, default 1 if 0.
    if enhanced['contract_size_decimal'] is None or enhanced['contract_size_decimal'].is_zero():
         enhanced['contract_size_decimal'] = Decimal('1.0')
         # init_logger.warning(f"Market data parsing: Invalid or zero contract size for {market['symbol']}. Defaulting to 1.0.")


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
    # Start with a copy of the original dictionary
    enhanced: PositionInfo = position.copy() # type: ignore # Allow copy despite TypedDict

    symbol = market_info['symbol']

    # Convert key numeric fields to Decimal using safe helper
    # Note: position size can be zero if flat, PnL/collateral/margin can be zero or negative
    enhanced.setdefault('size_decimal', _safe_market_decimal(position.get('contracts'), f"{symbol}.position.contracts", allow_zero=True, allow_negative=True, default=Decimal('0')))
    # Some exchanges use 'info.size' or similar for contracts. Check the raw info too.
    # For Bybit V5, 'info.size' is the quantity. Use that if 'contracts' is zero/None.
    if enhanced['size_decimal'] is None or enhanced['size_decimal'].is_zero():
         enhanced['size_decimal'] = _safe_market_decimal(position.get('info', {}).get('size'), f"{symbol}.position.info.size", allow_zero=True, allow_negative=True, default=Decimal('0'))

    # Determine side based on size_decimal if standard 'side' is missing or unreliable
    if enhanced['size_decimal'] > POSITION_QTY_EPSILON and position.get('side') != 'long':
         enhanced['side'] = 'long'
    elif enhanced['size_decimal'] < -POSITION_QTY_EPSILON and position.get('side') != 'short':
         enhanced['side'] = 'short'
    elif abs(enhanced['size_decimal']) <= POSITION_QTY_EPSILON:
         enhanced['side'] = 'none' # Use 'none' internally for flat state
         enhanced['size_decimal'] = Decimal('0') # Ensure size is exactly 0 if flat


    enhanced.setdefault('entryPrice_decimal', _safe_market_decimal(position.get('entryPrice'), f"{symbol}.position.entryPrice"))
    enhanced.setdefault('markPrice_decimal', _safe_market_decimal(position.get('markPrice'), f"{symbol}.position.markPrice"))
    enhanced.setdefault('liquidationPrice_decimal', _safe_market_decimal(position.get('liquidationPrice'), f"{symbol}.position.liquidationPrice"))
    enhanced.setdefault('leverage_decimal', _safe_market_decimal(position.get('leverage'), f"{symbol}.position.leverage", default=Decimal('1.0'))) # Default leverage to 1 if missing
    enhanced.setdefault('unrealizedPnl_decimal', _safe_market_decimal(position.get('unrealizedPnl'), f"{symbol}.position.unrealizedPnl", allow_zero=True, allow_negative=True))
    enhanced.setdefault('notional_decimal', _safe_market_decimal(position.get('notional'), f"{symbol}.position.notional", allow_zero=True, allow_negative=True))
    enhanced.setdefault('collateral_decimal', _safe_market_decimal(position.get('collateral'), f"{symbol}.position.collateral", allow_zero=True, allow_negative=True))
    enhanced.setdefault('initialMargin_decimal', _safe_market_decimal(position.get('initialMargin'), f"{symbol}.position.initialMargin", allow_zero=True))
    enhanced.setdefault('maintenanceMargin_decimal', _safe_market_decimal(position.get('maintenanceMargin'), f"{symbol}.position.maintenanceMargin", allow_zero=True))


    # --- Extract Raw and Parsed Native Protection Info (Bybit V5 specifics from 'info') ---
    info = position.get('info', {})
    # Bybit V5 uses keys like 'stopLoss', 'takeProfit', 'trailingStop', 'activePrice' in position 'info'
    enhanced.setdefault('stopLossPrice_raw', info.get('stopLoss'))
    enhanced.setdefault('takeProfitPrice_raw', info.get('takeProfit'))
    # Note: Bybit V5 'trailingStop' in position info is the current trigger price, not the % distance.
    enhanced.setdefault('trailingStopPrice_raw', info.get('trailingStop'))
    enhanced.setdefault('tslActivationPrice_raw', info.get('activePrice')) # Bybit V5 activation price

    # Safely parse raw stop prices to Decimal
    # Treat '0' or '0.0' strings from API as None/0 Decimal
    enhanced.setdefault('stopLossPrice_dec', _safe_decimal_conversion(enhanced['stopLossPrice_raw'], allow_none=True))
    enhanced.setdefault('takeProfitPrice_dec', _safe_decimal_conversion(enhanced['takeProfitPrice_raw'], allow_none=True))
    enhanced.setdefault('trailingStopPrice_dec', _safe_decimal_conversion(enhanced['trailingStopPrice_raw'], allow_none=True))
    enhanced.setdefault('tslActivationPrice_dec', _safe_decimal_conversion(enhanced['tslActivationPrice_raw'], allow_none=True))

    # Initialize state flags (these would be managed by the bot's state machine per symbol)
    # These flags should ideally persist across cycles for a given symbol.
    # A simple approach is to add them to a per-symbol state dictionary in the main loop.
    # For now, initialize them here as False; the main logic will need to update/store them.
    enhanced.setdefault('be_activated', False)
    enhanced.setdefault('tsl_activated', False) # Flag whether TSL has been *activated* by price movement (client-side logic)


    return enhanced # type: ignore # Return as PositionInfo TypedDict

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
    try:
        # Common CCXT exchange options
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': 'linear',         # Assume linear contracts by default for V5
                'adjustForTimeDifference': True, # Auto-adjust for clock skew
                'recvWindow': CONFIG.get('api_timing', {}).get('recv_window', 10000), # Bybit V5 recvWindow (default 5000)
                # Timeouts for various operations (in milliseconds) from config if available, else hardcoded reasonable defaults
                'fetchTickerTimeout': CONFIG.get('api_timing', {}).get('fetch_ticker_timeout', 15000),
                'fetchBalanceTimeout': CONFIG.get('api_timing', {}).get('fetch_balance_timeout', 20000),
                'createOrderTimeout': CONFIG.get('api_timing', {}).get('create_order_timeout', 30000),
                'cancelOrderTimeout': CONFIG.get('api_timing', {}).get('cancel_order_timeout', 20000),
                'fetchPositionsTimeout': CONFIG.get('api_timing', {}).get('fetch_positions_timeout', 20000),
                'fetchOHLCVTimeout': CONFIG.get('api_timing', {}).get('fetch_ohlcv_timeout', 60000), # Longer timeout for potentially large kline fetches
            }
        }
        # Instantiate the Bybit exchange object
        exchange = ccxt.bybit(exchange_options)

        # Configure Sandbox Mode
        is_sandbox = CONFIG.get('use_sandbox', True)
        exchange.set_sandbox_mode(is_sandbox)
        if is_sandbox:
            lg.warning(f"{NEON_YELLOW}<<< USING SANDBOX MODE (Testnet Environment) >>>{RESET}")
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}!!! <<< USING LIVE TRADING ENVIRONMENT - REAL FUNDS AT RISK >>> !!!{RESET}")

        # Load Markets with Retries
        lg.info(f"Attempting to load markets for {exchange.id}...")
        markets_loaded = False
        last_market_error = None
        # Get retry settings from config, fallback to constants
        max_retries = CONFIG.get('retry_count', MAX_API_RETRIES)
        retry_delay = CONFIG.get('retry_delay', RETRY_DELAY_SECONDS)

        for attempt in range(max_retries + 1):
            try:
                # Force reload on retries to ensure fresh market data
                exchange.load_markets(reload=(attempt > 0))
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
                if attempt >= max_retries:
                    lg.critical(f"{NEON_RED}Maximum retries exceeded while loading markets due to network errors. Last error: {last_market_error}. Exiting.{RESET}")
                    return None
            except ccxt.AuthenticationError as e:
                 last_market_error = e
                 lg.critical(f"{NEON_RED}Authentication error loading markets: {e}. Check API Key/Secret/Permissions. Exiting.{RESET}")
                 return None
            except Exception as e:
                last_market_error = e
                lg.critical(f"{NEON_RED}Unexpected error loading markets: {e}. Exiting.{RESET}", exc_info=True)
                return None

            # Apply delay before retrying (only if not the last attempt)
            if not markets_loaded and attempt < max_retries:
                 delay = retry_delay * (attempt + 1) # Exponential backoff
                 lg.warning(f"Retrying market load in {delay} seconds...")
                 time.sleep(delay)

        if not markets_loaded:
            lg.critical(f"{NEON_RED}Failed to load markets for {exchange.id} after all retries. Last error: {last_market_error}. Exiting.{RESET}")
            return None

        lg.info(f"CCXT exchange initialized: {exchange.id} | Sandbox: {is_sandbox}")

        # Initial Balance Check
        lg.info(f"Attempting initial balance fetch for quote currency ({QUOTE_CURRENCY})...")
        initial_balance: Optional[Decimal] = None
        try:
            # Fetch balance using V5 specific parameters for linear accounts on Bybit
            # CCXT maps 'total' and 'free' to the unified structure.
            balance_data = fetch_balance(exchange, logger)
            if balance_data and QUOTE_CURRENCY in balance_data.get('total', {}):
                 initial_balance = _safe_decimal_conversion(balance_data['total'][QUOTE_CURRENCY])
            # fetch_balance helper already logs errors
        except ccxt.AuthenticationError as auth_err:
            # Handle auth errors specifically here as they are critical during balance check
            lg.critical(f"{NEON_RED}Authentication Error during initial balance fetch: {auth_err}. Check API Key/Secret/Permissions. Exiting.{RESET}")
            return None
        except Exception as balance_err:
             # Catch other potential errors during the initial balance check
             lg.warning(f"{NEON_YELLOW}Initial balance fetch encountered an error: {balance_err}.{RESET}", exc_info=True)
             # Let the logic below decide based on trading enabled status

        # Evaluate balance check result based on trading mode
        if initial_balance is not None:
            lg.info(f"{NEON_GREEN}Initial available balance: {initial_balance.normalize()} {QUOTE_CURRENCY}{RESET}")
            # Send notification about startup success
            if CONFIG.get("notifications", {}).get("enable_notifications", False):
                 send_notification(f"Bot Started ({exchange.id} {'Sandbox' if is_sandbox else 'LIVE'})",
                                   f"Bot version {BOT_VERSION} initialized successfully for {CONFIG.get('trading_pairs', [])}. Initial Balance ({QUOTE_CURRENCY}): {initial_balance.normalize()}",
                                   lg,
                                   CONFIG.get('notifications', {}).get('notification_type', 'email'))

            return exchange # Success!
        else:
            # Balance fetch failed (fetch_balance logs the failure reason)
            lg.error(f"{NEON_RED}Initial balance fetch FAILED for {QUOTE_CURRENCY}.{RESET}")
            if CONFIG.get('enable_trading', False):
                lg.critical(f"{NEON_RED}Trading is enabled, but initial balance check failed. Cannot proceed safely. Exiting.{RESET}")
                # Send notification about fatal error
                if CONFIG.get("notifications", {}).get("enable_notifications", False):
                    send_notification(f"Bot FATAL Error ({exchange.id} {'Sandbox' if is_sandbox else 'LIVE'})",
                                       f"Bot version {BOT_VERSION} failed to initialize. Critical: Initial balance fetch failed.",
                                       lg,
                                       CONFIG.get('notifications', {}).get('notification_type', 'email'))
                return None
            else:
                lg.warning(f"{NEON_YELLOW}Trading is disabled. Proceeding without confirmed initial balance, but errors might occur later.{RESET}")
                # Send warning notification
                if CONFIG.get("notifications", {}).get("enable_notifications", False):
                    send_notification(f"Bot Warning ({exchange.id} {'Sandbox' if is_sandbox else 'LIVE'})",
                                       f"Bot version {BOT_VERSION} initialized in dry-run mode, but initial balance fetch failed. Trading disabled, but investigate.",
                                       lg,
                                       CONFIG.get('notifications', {}).get('notification_type', 'email'))
                return exchange # Allow proceeding in non-trading mode

    except Exception as e:
        # Catch-all for errors during the initialization process itself
        lg.critical(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        # Send notification about fatal initialization error
        is_sandbox_notify = CONFIG.get('use_sandbox', True) # Use config value even if init failed early
        notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             try: # Wrap notification attempt in case of error during error handling
                 send_notification(f"Bot FATAL Error ({'Sandbox' if is_sandbox_notify else 'LIVE'})",
                                   f"Bot version {BOT_VERSION} failed to initialize. Unexpected error: {type(e).__name__} - {e}",
                                   lg,
                                   notify_type)
             except Exception as notify_err:
                  print(f"{NEON_RED}FATAL: Failed to send notification about initialization error: {notify_err}{RESET}", file=sys.stderr)

        return None

# --- CCXT Data Fetching Helpers ---
def fetch_balance(exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Fetches account balance using CCXT's fetch_balance.

    Handles Bybit V5 'category' parameter automatically for linear accounts.
    Includes retry logic for network errors and rate limits.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        logger (logging.Logger): The logger instance.

    Returns:
        Optional[Dict[str, Any]]: The balance dictionary, or None on failure.
    """
    lg = logger
    attempts = 0
    last_exception = None
    # Get retry settings from config, fallback to constants
    max_retries = CONFIG.get('retry_count', MAX_API_RETRIES)
    retry_delay = CONFIG.get('retry_delay', RETRY_DELAY_SECONDS)

    while attempts <= max_retries:
        try:
            lg.debug(f"Fetching balance data (Attempt {attempts + 1}/{max_retries + 1})...")
            # Bybit V5 fetchBalance requires category in params for unified account
            params = {'category': 'linear'} # Assuming linear account structure for this bot
            balance = exchange.fetch_balance(params=params)
            # Basic check if balance structure looks valid
            if isinstance(balance, dict) and ('total' in balance or 'free' in balance):
                 lg.debug("Balance data fetched successfully.")
                 return balance
            else:
                 last_exception = ValueError("Fetch balance returned invalid structure.")
                 lg.warning(f"Fetched balance has unexpected structure (Attempt {attempts + 1}). Retrying...")

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching balance: {e}. Retry {attempts + 1}/{max_retries + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * 3 # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            # Don't increment attempts for rate limits, just wait and retry the same attempt number logically
            continue # Skip attempt increment and retry delay below
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.error(f"{NEON_RED}Authentication Error fetching balance: {e}. Check API Key/Secret/Permissions. Stopping fetch.{RESET}")
             return None # Fatal error for this operation
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}")
            # Could add checks for specific non-retryable error codes here if needed
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
            return None # Exit on unexpected errors

        # Increment attempt counter and apply delay (only if not a rate limit wait)
        attempts += 1
        if attempts <= max_retries:
            time.sleep(retry_delay * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to fetch balance after {max_retries + 1} attempts. Last error: {last_exception}{RESET}")
    return None

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using `fetch_ticker`.

    Prioritizes 'last' price. Falls back progressively:
    1. Mid-price ((bid + ask) / 2) if both bid and ask are valid.
    2. 'ask' price if only ask is valid.
    3. 'bid' price if only bid is valid.

    Includes retry logic for network errors and rate limits.

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
    # Get retry settings from config, fallback to constants
    max_retries = CONFIG.get('retry_count', MAX_API_RETRIES)
    retry_delay = CONFIG.get('retry_delay', RETRY_DELAY_SECONDS)

    while attempts <= max_retries:
        try:
            lg.debug(f"Fetching ticker data for {symbol} (Attempt {attempts + 1}/{max_retries + 1})")
            ticker = exchange.fetch_ticker(symbol)
            price: Optional[Decimal] = None

            # Helper to safely convert ticker values to Decimal
            def safe_decimal_from_ticker(value: Optional[Any], field_name: str) -> Optional[Decimal]:
                """Safely converts ticker field to positive Decimal."""
                if value is None: return None
                try:
                    s_val = str(value).strip()
                    if not s_val: return None
                    dec_val = Decimal(s_val)
                    return dec_val if dec_val.is_finite() and dec_val > POSITION_QTY_EPSILON else None # Ensure finite and > epsilon
                except (ValueError, InvalidOperation, TypeError):
                    lg.debug(f"Could not parse ticker field '{field_name}' value '{value}' to Decimal.")
                    return None

            # 1. Try 'last' price
            price = safe_decimal_from_ticker(ticker.get('last'), 'last')

            # 2. Fallback to mid-price if 'last' is invalid
            if price is None:
                bid = safe_decimal_from_ticker(ticker.get('bid'), 'bid')
                ask = safe_decimal_from_ticker(ticker.get('ask'), 'ask')
                if bid is not None and ask is not None: # Check for None after safe conversion
                    price = (bid + ask) / Decimal('2')
                    lg.debug(f"Using mid-price (Bid: {bid.normalize()}, Ask: {ask.normalize()}) -> {price.normalize()}")
                # 3. Fallback to 'ask' if only ask is valid
                elif ask is not None:
                    price = ask
                    lg.debug(f"Using 'ask' price as fallback: {price.normalize()}")
                # 4. Fallback to 'bid' if only bid is valid
                elif bid is not None:
                    price = bid
                    lg.debug(f"Using 'bid' price as fallback: {price.normalize()}")

            # Check if a valid price was obtained (non-None and finite)
            if price is not None and price.is_finite() and price > POSITION_QTY_EPSILON:
                lg.debug(f"Current price successfully fetched for {symbol}: {price.normalize()}")
                return price.normalize() # Ensure normalization
            else:
                last_exception = ValueError(f"No valid price ('last', 'mid', 'ask', 'bid') found in ticker data. Ticker: {ticker}")
                lg.warning(f"No valid price found in ticker (Attempt {attempts + 1}). Retrying...")

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retry {attempts + 1}/{max_retries + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * 3 # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            # Don't increment attempts, just retry after waiting
            continue
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error fetching price: {e}. Check API Key/Secret/Permissions. Stopping fetch.{RESET}")
             return None # Fatal error for this operation
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Could add checks for specific non-retryable error codes here if needed
            # For now, assume potentially retryable unless it's an auth error
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None # Exit on unexpected errors

        # Increment attempt counter and apply delay (only if not a rate limit wait)
        attempts += 1
        if attempts <= max_retries:
            time.sleep(retry_delay * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to fetch current price for {symbol} after {max_retries + 1} attempts. Last error: {last_exception}{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, market_info: MarketInfo, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches OHLCV (kline) data using CCXT's `fetch_ohlcv` method with enhancements.

    - Handles Bybit V5 'category' parameter automatically based on market info.
    - Implements robust retry logic for network errors and rate limits.
    - Validates fetched data timestamp lag to detect potential staleness.
    - Processes data into a Pandas DataFrame with Decimal types for precision.
    - Cleans data (drops rows with NaNs in key columns, zero prices/volumes).
    - Trims DataFrame to `MAX_DF_LEN` to manage memory usage.
    - Ensures DataFrame is sorted by timestamp.

    Args:
        exchange (ccxt.Exchange): The initialized ccxt.Exchange object.
        market_info (MarketInfo): The enhanced market info for the symbol.
        timeframe (str): The CCXT timeframe string (e.g., "5m", "1h", "1d").
        limit (int): The desired number of klines. Will be capped by `BYBIT_API_KLINE_LIMIT`
                     per API request, but the function aims to fetch the `limit` specified
                     (currently single request, multi-request fetching for large limits is TBD).
        logger (logging.Logger): The logger instance.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the OHLCV data, or an empty DataFrame on failure.
    """
    lg = logger
    symbol = market_info['symbol']
    attempts = 0
    last_exception = None
    ohlcv_data: List[List[Union[int, float]]] = [] # List of lists for OHLCV
    # Get retry settings from config, fallback to constants
    max_retries = CONFIG.get('retry_count', MAX_API_RETRIES)
    retry_delay = CONFIG.get('retry_delay', RETRY_DELAY_SECONDS)
    # Get fetch limit from config, fallback to default or limit to exchange max
    fetch_limit_cfg = CONFIG.get('fetch_limit', DEFAULT_FETCH_LIMIT)
    fetch_limit_actual = min(limit, fetch_limit_cfg, BYBIT_API_KLINE_LIMIT) # Respect all limits

    if fetch_limit_actual <= 0:
         lg.error(f"{NEON_RED}Kline fetch limit is zero or negative ({fetch_limit_actual}). Cannot fetch data.{RESET}")
         return pd.DataFrame()

    while attempts <= max_retries:
        try:
            lg.debug(f"Fetching {fetch_limit_actual} klines for {symbol} ({timeframe}) (Attempt {attempts + 1}/{max_retries + 1})...")
            # Bybit V5 fetchOHLCV requires category in params for perpetual futures
            params = {'category': market_info.get('info', {}).get('category', 'linear')} # Use category from market info if available, default to linear

            # Use exchange.fetch_ohlcv - CCXT handles pagination internally for many exchanges,
            # but Bybit V5 fetch_ohlcv seems to respect the 'limit' parameter directly up to its max.
            # If we need > 1000 klines, we'd need manual pagination loops here.
            # For now, assume the requested 'limit' is within exchange limits or we accept less.
            # CCXT fetch_ohlcv limit parameter is usually the number of candles requested.
            # The exchange limit (BYBIT_API_KLINE_LIMIT) is the max it returns per single request.
            # So, we request `limit` but may only get up to `BYBIT_API_KLINE_LIMIT`.
            # To get exactly `limit` up to `MAX_DF_LEN`, we'd need manual pagination if limit > BYBIT_API_KLINE_LIMIT.
            # Let's simplify for now and just request `limit` (from function arg) and warn if we get less than expected.
            # Or, even better, request `min(limit, BYBIT_API_KLINE_LIMIT)` and adjust strategy needs accordingly.
            # Let's request `limit` and log how many we got.
            ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit, params=params)

            if not ohlcv_data:
                last_exception = ValueError("Fetch OHLCV returned empty data.")
                lg.warning(f"Fetched OHLCV data is empty for {symbol} ({timeframe}) (Attempt {attempts + 1}). Retrying...")
            else:
                lg.debug(f"Successfully fetched {len(ohlcv_data)} klines for {symbol} ({timeframe}).")
                break # Exit retry loop on successful fetch

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching klines for {symbol} ({timeframe}): {e}. Retry {attempts + 1}/{max_retries + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * 3 # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching klines for {symbol} ({timeframe}): {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Skip attempt increment and retry delay below
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error fetching klines: {e}. Check API Key/Secret/Permissions. Stopping fetch.{RESET}")
             return pd.DataFrame() # Fatal error
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol} ({timeframe}): {e}{RESET}")
            # Could add checks for specific non-retryable error codes here
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching klines for {symbol} ({timeframe}): {e}{RESET}", exc_info=True)
            return pd.DataFrame() # Exit on unexpected errors

        # Increment attempt counter and apply delay (only if not a rate limit wait)
        attempts += 1
        if attempts <= max_retries:
            time.sleep(retry_delay * attempts) # Exponential backoff

    if not ohlcv_data:
        lg.error(f"{NEON_RED}Failed to fetch klines for {symbol} ({timeframe}) after {max_retries + 1} attempts. Last error: {last_exception}{RESET}")
        return pd.DataFrame() # Return empty DataFrame on final failure

    # --- Data Processing & Cleaning ---
    df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Ensure necessary columns are numeric (handle potential string/None values from exchange)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        # Use errors='coerce' to turn unparseable values into NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert timestamp to datetime objects (UTC) and set as index
    # Ensure unit is correct (usually 'ms' for CCXT timestamps)
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True, drop=False) # Keep timestamp column as well
    except Exception as time_e:
         lg.error(f"{NEON_RED}Data Fetch: Error converting timestamp column for {symbol}: {time_e}{RESET}")
         return pd.DataFrame() # Cannot proceed without valid timestamps

    # --- Robust NaN Handling ---
    # Check for NaNs in critical columns (OHLCV)
    initial_nan_count = df[numeric_cols].isnull().sum().sum()
    if initial_nan_count > 0:
        nan_counts_per_col = df[numeric_cols].isnull().sum()
        lg.warning(f"{NEON_YELLOW}Data Fetch: Found {initial_nan_count} NaN values in OHLCV data for {symbol} after conversion:\n"
                       f"{nan_counts_per_col[nan_counts_per_col > 0]}\nAttempting forward fill (ffill)...{RESET}")
        df.ffill(inplace=True) # Fill NaNs with the previous valid observation

        # Check if NaNs remain (likely at the beginning of the series if data history is short)
        remaining_nan_count = df[numeric_cols].isnull().sum().sum()
        if remaining_nan_count > 0:
            lg.warning(f"{NEON_YELLOW}NaNs remain after ffill ({remaining_nan_count}). Attempting backward fill (bfill)...{RESET}")
            df.bfill(inplace=True) # Fill remaining NaNs with the next valid observation

            # Final check: if NaNs still exist, data is likely too gappy at start/end or completely invalid
            final_nan_count = df[numeric_cols].isnull().sum().sum()
            if final_nan_count > 0:
                lg.error(f"{NEON_RED}Data Fetch: Unfillable NaN values ({final_nan_count}) remain after ffill and bfill for {symbol}. "
                             f"Data quality insufficient. Columns with NaNs:\n{df[numeric_cols].isnull().sum()[df[numeric_cols].isnull().sum() > 0]}\nReturning empty DataFrame.{RESET}")
                return pd.DataFrame() # Cannot proceed with unreliable data

    # Ensure Decimal types for OHLCV columns after cleaning
    for col in numeric_cols:
        # Using .apply(safe_decimal_conversion) on the cleaned column
        df[col] = df[col].apply(_safe_decimal_conversion) # Default 0.0 for any remaining issues

    # Drop rows where essential Decimal columns are NaN (after conversion attempts)
    df.dropna(subset=numeric_cols, inplace=True)

    # Ensure no zero prices or non-positive volumes remain after conversion/cleaning
    if df.empty or (df[['open', 'high', 'low', 'close']] <= POSITION_QTY_EPSILON).any().any() or (df['volume'] < 0).any():
         lg.warning(f"{NEON_YELLOW}Data Fetch: Found zero/negative prices or negative volumes after cleaning for {symbol}. Returning empty DataFrame.{RESET}")
         return pd.DataFrame()
    if (df['volume'] == 0).any():
         lg.debug(f"Data Fetch: Found zero volume bars for {symbol}.") # Zero volume is sometimes expected but note it

    # Ensure DataFrame is sorted by timestamp ascending
    df.sort_index(inplace=True)

    # Trim DataFrame to MAX_DF_LEN to control memory usage
    if len(df) > MAX_DF_LEN:
        original_len = len(df)
        df = df.tail(MAX_DF_LEN).copy() # Use .copy() to avoid SettingWithCopyWarning
        lg.debug(f"Data Fetch: Trimmed DataFrame for {symbol} from {original_len} to {MAX_DF_LEN} rows.")

    # Check if the last candle is complete (timestamp is slightly in the past)
    # This is an approximation; exact candle close time is better but harder across exchanges
    if len(df) > 0:
        last_candle_time_utc = df.index[-1]
        now_utc = pd.Timestamp.now(tz='UTC')
        # Convert interval string to seconds
        interval_seconds = exchange.parse_timeframe(timeframe) if exchange.parse_timeframe(timeframe) is not None else 60 # Default to 60s if parse fails (unlikely)
        # A completed candle's timestamp should be before the current time,
        # typically by at least the interval duration. Allow a small buffer (e.g., 10% of interval).
        # This check is heuristic and depends on exchange timestamp precision and server load.
        # For high-frequency scalping, we often use the last available candle regardless, but it's worth noting.
        time_diff_seconds = (now_utc - last_candle_time_utc).total_seconds()
        if time_diff_seconds < interval_seconds * 0.1: # If last candle is less than 10% of interval ago
            lg.warning(f"{NEON_YELLOW}Data Fetch: Last candle timestamp ({last_candle_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}) for {symbol} ({timeframe}) "
                           f"is very recent ({time_diff_seconds:.1f}s ago, interval is {interval_seconds}s). "
                           f"It might be incomplete. Using it anyway as typical for scalping, but be aware.{RESET}")
        elif time_diff_seconds >= interval_seconds * 2: # If last candle is more than 2 intervals ago
             lg.warning(f"{NEON_YELLOW}Data Fetch: Last candle timestamp ({last_candle_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}) for {symbol} ({timeframe}) "
                           f"is old ({time_diff_seconds:.1f}s ago). Data might be stale.{RESET}")


    lg.debug(f"Data Fetch: Successfully processed {len(df)} OHLCV candles for {symbol}.")
    return df

def get_current_position(exchange: ccxt.Exchange, market_info: MarketInfo, logger: logging.Logger) -> PositionInfo:
    """
    Fetches current position details for a specific symbol using Bybit V5 API specifics.
    Assumes One-Way Mode (looks for positionIdx=0). Includes parsing attached SL/TP/TSL.

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
    # Initialize default flat state dictionary with necessary keys
    default_pos: PositionInfo = {
        'id': None,
        'symbol': symbol,
        'timestamp': None,
        'datetime': None,
        'contracts': None, # Legacy field
        'contractSize': market_info['contractSize'], # Use market contract size default
        'side': 'none', # Use 'none' for flat state
        'notional': None,
        'leverage': None,
        'unrealizedPnl': None,
        'realizedPnl': None,
        'collateral': None,
        'entryPrice': None,
        'markPrice': None,
        'liquidationPrice': None,
        'marginMode': None,
        'hedged': None,
        'maintenanceMargin': None,
        'maintenanceMarginPercentage': None,
        'initialMargin': None,
        'initialMarginPercentage': None,
        'marginRatio': None,
        'lastUpdateTimestamp': None,
        'info': {}, # Empty raw info
        # Enhanced/Parsed fields
        'size_decimal': Decimal("0.0"), # Explicitly zero for flat
        'stopLossPrice_raw': None, 'takeProfitPrice_raw': None, 'trailingStopPrice_raw': None, 'tslActivationPrice_raw': None,
        'stopLossPrice_dec': None, 'takeProfitPrice_dec': None, 'trailingStopPrice_dec': None, 'tslActivationPrice_dec': None,
        # Bot state flags (default to False for a fresh check)
        'be_activated': False, 'tsl_activated': False
    }

    # Get retry settings from config, fallback to constants
    max_retries = CONFIG.get('retry_count', MAX_API_RETRIES)
    retry_delay = CONFIG.get('retry_delay', RETRY_DELAY_SECONDS)
    attempts = 0
    last_exception = None

    while attempts <= max_retries:
        try:
            lg.debug(f"Fetching position data for {symbol} (Attempt {attempts + 1}/{max_retries + 1})...")

            # Bybit V5 fetchPositions requires params={'category': 'linear'} and optionally symbol
            # We only care about the One-Way position (positionIdx=0) for the target symbol
            # Use category from market info if available, default to linear
            params = {'category': market_info.get('info', {}).get('category', 'linear')}
            # Filter by exchange-specific market ID if available
            if market_info.get('id'):
                 params['symbol'] = market_info['id']

            # CCXT fetchPositions returns a list of positions. For One-Way, we expect max one per symbol.
            # Use the unified symbol for the CCXT call itself
            positions = exchange.fetch_positions(symbols=[symbol], params=params)

            # Filter for the relevant position in One-Way mode (positionIdx=0)
            # Also check for positionIdx=1 and positionIdx=2 in case of hedge mode if side is Buy/Sell
            relevant_position = None
            for p in positions:
                # Check if it's the correct symbol (CCXT filters by this, but good to be sure)
                if p.get('symbol') != symbol: continue
                # For One-Way mode, positionIdx is 0 and the size should be the total size.
                # The 'side' field indicates the direction (Long/Short).
                # If the 'positionIdx' field exists and is '0', this is the One-Way position.
                # If positionIdx is missing or not '0', it might be hedge mode (positionIdx 1/2 for Long/Short side)
                # Let's assume One-Way mode as per bot design and check for positionIdx=0.
                # Bybit V5 uses positionIdx 0 for both Long and Short in One-Way mode.
                pos_idx_raw = p.get('info', {}).get('positionIdx')
                if pos_idx_raw is not None and str(pos_idx_raw) == '0':
                     relevant_position = p
                     break # Found the One-Way position

            if relevant_position:
                 # Enhance the found position data
                 enhanced_pos = enhance_position_info(relevant_position, market_info)
                 # Check if the enhanced position has non-zero size
                 if abs(enhanced_pos['size_decimal']) > POSITION_QTY_EPSILON:
                      lg.debug(f"{NEON_GREEN}Active {enhanced_pos.get('side', 'N/A').capitalize()} position found for {symbol}. "
                               f"Qty: {enhanced_pos['size_decimal'].normalize()}, Entry: {enhanced_pos['entryPrice_decimal'].normalize() if enhanced_pos['entryPrice_decimal'] else 'N/A'}{RESET}")
                      # Log native stops if attached (check parsed decimal values)
                      stop_details = []
                      if enhanced_pos.get('stopLossPrice_dec') is not None and enhanced_pos['stopLossPrice_dec'] > POSITION_QTY_EPSILON:
                           stop_details.append(f"SL: {enhanced_pos['stopLossPrice_dec'].normalize()}")
                      if enhanced_pos.get('takeProfitPrice_dec') is not None and enhanced_pos['takeProfitPrice_dec'] > POSITION_QTY_EPSILON:
                           stop_details.append(f"TP: {enhanced_pos['takeProfitPrice_dec'].normalize()}")
                      if enhanced_pos.get('trailingStopPrice_dec') is not None and enhanced_pos['trailingStopPrice_dec'] > POSITION_QTY_EPSILON:
                           stop_details.append(f"TSL Trigger: {enhanced_pos['trailingStopPrice_dec'].normalize()}")
                      if stop_details:
                           lg.debug(f"{NEON_CYAN}Position Stops: {' | '.join(stop_details)}{RESET}")

                      # Return the enhanced position data
                      return enhanced_pos
                 else:
                     # The relevant positionIdx=0 was found, but size is zero. This means flat.
                     lg.debug(f"Position check for {symbol}: PositionIdx=0 found, but size is zero. Currently Flat.")
                     return default_pos # Return default flat state

            else:
                 # No relevant position found in the list (either empty list or no positionIdx=0)
                 lg.debug(f"Position check for {symbol}: No position found matching One-Way criteria. Currently Flat.")
                 return default_pos # Return default flat state

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching position for {symbol}: {e}. Retry {attempts + 1}/{max_retries + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = retry_delay * 3 # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching position for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Skip attempt increment and retry delay below
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error fetching position: {e}. Check API Key/Secret/Permissions. Stopping fetch.{RESET}")
             return default_pos # Fatal error, return flat state
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching position for {symbol}: {e}{RESET}")
            # Check for specific errors that might mean the position is already closed/never existed
            # e.g. Bybit V5 might return something like 'position not found' or similar.
            # For now, assume retryable unless it's an auth error.
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching position for {symbol}: {e}{RESET}", exc_info=True)
            return default_pos # Exit on unexpected errors

        # Increment attempt counter and apply delay (only if not a rate limit wait)
        attempts += 1
        if attempts <= max_retries:
            time.sleep(retry_delay * attempts) # Exponential backoff


    lg.error(f"{NEON_RED}Failed to fetch position for {symbol} after {max_retries + 1} attempts. Last error: {last_exception}. Assuming flat.{RESET}")
    return default_pos # Return default flat state after all retries fail

# --- Strategy Implementation (Placeholder) ---
# Note: The provided snippets had basic/placeholder logic for VT and OB.
# This implementation provides the structure and integrates the required inputs
# (DataFrame, OBs, etc.), but the core trading conditions will be simplified
# based on the snippets or left as explicit placeholders for extension.

def calculate_volumatic_trend(df: pd.DataFrame, vt_length: int, vt_atr_period: int, vt_vol_ema_length: int, vt_atr_multiplier: Decimal, logger: logging.Logger) -> pd.DataFrame:
    """
    Calculates Volumatic Trend indicators (Trend Line, Bands, Volume Norm).

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns (Decimal).
        vt_length: Length for the primary trend line (EMA).
        vt_atr_period: Period for ATR calculation.
        vt_vol_ema_length: Length for Volume EMA used in normalization.
        vt_atr_multiplier: Multiplier for ATR bands.
        logger: Logger instance.

    Returns:
        DataFrame with 'trend_line', 'atr', 'upper_band', 'lower_band', 'vol_ratio', 'vol_norm_int' columns added (Decimal).
        Returns the original DataFrame if calculation fails or data is insufficient.
    """
    lg = logger
    # Ensure necessary columns exist and are Decimal
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols) or \
       not all(pd.api.types.is_extension_array_dtype(df[col].dtype) for col in required_cols) or \
       len(df) < max(vt_length, vt_atr_period, vt_vol_ema_length) + 10: # Add buffer for indicator stability
        lg.warning(f"{NEON_YELLOW}VT Calc: Insufficient or invalid data ({len(df)} rows, missing/non-decimal columns). Cannot calculate VT.{RESET}")
        return df # Return original df if inputs are bad

    try:
        # Ensure pandas_ta inputs are float (pandas_ta often works with floats internally)
        df_float = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # Calculate ATR (pandas_ta)
        df['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=vt_atr_period).apply(_safe_decimal_conversion, default=Decimal(0), allow_none=True)

        # Calculate Volume EMA and Ratio (pandas_ta)
        df['vol_ema'] = ta.ema(df_float['volume'], length=vt_vol_ema_length).apply(_safe_decimal_conversion, default=Decimal(0), allow_none=True)

        # Calculate Volume Ratio (Last Volume / Volume EMA) - Avoid division by zero
        # Use .replace(0, np.nan) on the Decimal column to handle zero EMA, then convert to float for division
        df['vol_ratio'] = (df['volume'].astype(float) / df['vol_ema'].replace(Decimal(0), np.nan).astype(float)).fillna(1.0).apply(_safe_decimal_conversion, default=Decimal(1.0), allow_none=True)

        # Normalize Volume Ratio (e.g., 0-100+) - Example logic, adjust as needed
        # Convert ratio to integer percentage, capping at a high value like 200
        df['vol_norm_int'] = df['vol_ratio'].apply(lambda x: min(int(x * 100), 200) if x is not None and x.is_finite() else 0)


        # Calculate Trend Line (EMA or other)
        df['trend_line'] = ta.ema(df_float['close'], length=vt_length).apply(_safe_decimal_conversion, default=Decimal(0), allow_none=True)


        # Calculate ATR Bands (using Decimal multiplication)
        # Ensure ATR is valid before multiplying
        df['upper_band'] = df.apply(lambda row: row['trend_line'] + (row['atr'] * vt_atr_multiplier) if row['trend_line'] is not None and row['atr'] is not None and row['atr'].is_finite() and row['atr'] > 0 else None, axis=1)
        df['lower_band'] = df.apply(lambda row: row['trend_line'] - (row['atr'] * vt_atr_multiplier) if row['trend_line'] is not None and row['atr'] is not None and row['atr'].is_finite() and row['atr'] > 0 else None, axis=1)


        # Check for NaNs in crucial output columns (last row)
        output_cols = ['trend_line', 'atr', 'upper_band', 'lower_band', 'vol_ratio', 'vol_norm_int']
        if df[output_cols].iloc[-1].isnull().any():
            nan_cols = df[output_cols].iloc[-1].isnull()
            nan_details = ', '.join([col for col in output_cols if nan_cols[col]])
            lg.warning(f"{NEON_YELLOW}VT Calc: Calculation resulted in NaN(s) for last candle in columns: {nan_details}. Signal generation may be affected.{RESET}")

        lg.debug(f"VT Calc: Completed calculations. Last ATR: {df['atr'].iloc[-1].normalize() if not df.empty and df['atr'].iloc[-1] is not None else 'N/A'}, Last Vol Norm: {df['vol_norm_int'].iloc[-1] if not df.empty and df['vol_norm_int'].iloc[-1] is not None else 'N/A'}")

        return df # Return DataFrame with new columns

    except Exception as e:
        lg.error(f"{NEON_RED}VT Calc: Unexpected error during calculation: {e}{RESET}", exc_info=True)
        # Add NA columns on error to ensure downstream functions don't crash on missing columns
        for col in ['trend_line', 'atr', 'upper_band', 'lower_band', 'vol_ema', 'vol_ratio', 'vol_norm_int']:
             df[col] = pd.NA
        return df # Return DataFrame with potential NA columns

def identify_order_blocks(df: pd.DataFrame, strategy_params: Dict[str, Any], logger: logging.Logger) -> List[OrderBlock]:
    """
    Identifies Order Blocks (OBs) based on Pivot Highs/Lows and configured source/params.
    Simplified detection using pivot points directly.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns (Decimal).
        strategy_params: Dictionary with OB configuration parameters.
        logger: Logger instance.

    Returns:
        List[OrderBlock]: A list of *all* identified Order Blocks (active or not).
                          Further logic is needed to filter for *active* OBs.
    """
    lg = logger
    ob_source = strategy_params.get('ob_source', DEFAULT_OB_SOURCE)
    ph_left = strategy_params.get('ph_left', DEFAULT_PH_LEFT)
    ph_right = strategy_params.get('ph_right', DEFAULT_PH_RIGHT)
    pl_left = strategy_params.get('pl_left', DEFAULT_PL_LEFT)
    pl_right = strategy_params.get('pl_right', DEFAULT_PL_RIGHT)
    ob_max_boxes = strategy_params.get('ob_max_boxes', DEFAULT_OB_MAX_BOXES)

    # Ensure necessary columns exist and are Decimal
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or \
       not all(pd.api.types.is_extension_array_dtype(df[col].dtype) for col in required_cols) or \
       len(df) < max(ph_left + ph_right, pl_left + pl_right) + 1: # Need enough data for pivot calculation window
        lg.warning(f"{NEON_YELLOW}OB Calc: Insufficient or invalid data ({len(df)} rows, missing/non-decimal columns). Cannot identify OBs.{RESET}")
        return [] # Return empty list on insufficient data

    df = df.copy() # Work on a copy

    # Define high/low source for pivots based on config ('Wicks' vs 'Body')
    # For pivots, 'Wicks' uses actual high/low. 'Body' uses max(open, close) / min(open, close) ?
    # Let's use 'high' and 'low' columns for pivots regardless of 'ob_source', but use 'ob_source'
    # when defining the OB box boundaries themselves later if needed.
    # Simplified Pivot Calc: Find index where high[center] is highest in window, low[center] is lowest.
    # Use float for rolling/comparison, then convert result back.
    df_float_high = df['high'].astype(float)
    df_float_low = df['low'].astype(float)

    ph_window = ph_left + ph_right + 1
    pl_window = pl_left + pl_right + 1

    # Identify potential pivot high indices
    pivot_high_indices = []
    if len(df) >= ph_window:
        # Find indices where the central value in the window is the maximum
        rolling_max = df_float_high.rolling(window=ph_window, center=True, min_periods=ph_window).max()
        # Compare the central element with the rolling max. Use .iloc for index-based access
        # Note: This comparison can return NaN for windows at the start/end due to min_periods
        pivot_high_mask = (df_float_high.iloc[ph_left:len(df)-ph_right] == rolling_max.iloc[ph_left:len(df)-ph_right])
        pivot_high_indices = df.index[ph_left:len(df)-ph_right][pivot_high_mask].tolist()

    # Identify potential pivot low indices
    pivot_low_indices = []
    if len(df) >= pl_window:
        # Find indices where the central value in the window is the minimum
        rolling_min = df_float_low.rolling(window=pl_window, center=True, min_periods=pl_window).min()
         # Compare the central element with the rolling min
        pivot_low_mask = (df_float_low.iloc[pl_left:len(df)-pl_right] == rolling_min.iloc[pl_left:len(df)-pl_right])
        pivot_low_indices = df.index[pl_left:len(df)-pl_right][pivot_low_mask].tolist()

    all_order_blocks: List[OrderBlock] = []

    # Create OrderBlock objects from pivot indices
    for ts in pivot_high_indices:
        candle = df.loc[ts]
        # Bearish OB: Often defined by the candle *before* the pivot high, or the pivot candle itself.
        # Simplified: Use the pivot candle's high and low for the box boundaries.
        ob_id = f"BEAR_{int(ts.timestamp() * 1000)}"
        top = candle['high']
        bottom = candle['low']
        all_order_blocks.append(OrderBlock(
            id=ob_id, type="BEAR", timestamp=ts, top=top, bottom=bottom,
            active=True, violated=False, violation_ts=None
        ))

    for ts in pivot_low_indices:
        candle = df.loc[ts]
        # Bullish OB: Often defined by the candle *before* the pivot low, or the pivot candle itself.
        # Simplified: Use the pivot candle's high and low for the box boundaries.
        ob_id = f"BULL_{int(ts.timestamp() * 1000)}"
        top = candle['high']
        bottom = candle['low']
        all_order_blocks.append(OrderBlock(
            id=ob_id, type="BULL", timestamp=ts, top=top, bottom=bottom,
            active=True, violated=False, violation_ts=None
        ))

    # Sort OBs by timestamp (most recent first) and trim to max_boxes limit
    # We need to track activity/violation across time, so keeping more history than just active might be needed.
    # But for generating signals based on *current* active OBs, sorting and trimming by recency is logical.
    all_order_blocks.sort(key=lambda x: x['timestamp'], reverse=True)
    all_order_blocks = all_order_blocks[:ob_max_boxes * 2] # Keep more than max_boxes initially to filter active ones

    # --- Filter for Active Order Blocks ---
    # An OB is typically considered active until violated by price action.
    # Simple violation: Price closes above a BEAR OB's top or below a BULL OB's bottom.
    # More complex: Any part of a future candle wick/body crosses the boundary.
    # Let's use a simple close violation for now.

    active_bull_boxes: List[OrderBlock] = []
    active_bear_boxes: List[OrderBlock] = []
    last_candle_ts = df.index[-1]
    # Get prices for violation check (using the latest candle)
    last_close_dec = df['close'].iloc[-1] if not df.empty else Decimal('0')
    last_high_dec = df['high'].iloc[-1] if not df.empty else Decimal('0')
    last_low_dec = df['low'].iloc[-1] if not df.empty else Decimal('0')

    # Iterate through candles to check for violations relative to *their* position in history
    # This is more accurate than checking against only the last candle.
    # This requires iterating from older candles to newer ones.
    all_order_blocks_chrono = sorted(all_order_blocks, key=lambda x: x['timestamp']) # Sort chronologically

    temp_active_bull = [] # Use temporary lists to build active OBs
    temp_active_bear = []

    for idx, candle in df.iterrows():
        current_ts = candle.name # Timestamp is the index
        current_open = candle['open']
        current_high = candle['high']
        current_low = candle['low']
        current_close = candle['close']

        # Check existing active OBs for violation by the current candle
        # Iterate through a copy to allow modification during iteration if needed (though we modify 'active' flag, not the list)
        # Let's check against temp lists first.
        for ob in list(temp_active_bull): # Check bullish OBs
            if ob['active']: # Check only if not already violated by a previous candle
                # Bull OB (Bottom: demand zone) violated if price closes BELOW its bottom
                if current_close < ob['bottom']:
                    ob['violated'] = True
                    ob['violation_ts'] = current_ts
                    ob['active'] = False
                    lg.debug(f"OB Calc: Bull OB violated at {current_ts.strftime('%Y-%m-%d %H:%M')}: {ob['bottom'].normalize()} by close {current_close.normalize()}")
                # Could add more complex violation checks here (wick crossing, body crossing, closing within a percentage, etc.)

        for ob in list(temp_active_bear): # Check bearish OBs
            if ob['active']: # Check only if not already violated by a previous candle
                # Bear OB (Top: supply zone) violated if price closes ABOVE its top
                if current_close > ob['top']:
                    ob['violated'] = True
                    ob['violation_ts'] = current_ts
                    ob['active'] = False
                    lg.debug(f"OB Calc: Bear OB violated at {current_ts.strftime('%Y-%m-%d %H:%M')}: {ob['top'].normalize()} by close {current_close.normalize()}")
                 # Could add more complex violation checks here

        # Add new OBs formed by the *current* candle (if any) to the temp lists
        # Find OBs from all_order_blocks_chrono that match the current candle's timestamp
        new_obs_this_candle = [ob for ob in all_order_blocks_chrono if ob['timestamp'] == current_ts]
        for new_ob in new_obs_this_candle:
             if new_ob['type'] == "BULL":
                  # Check if this new BULL OB is immediately violated by the current candle itself
                  if current_close < new_ob['bottom']:
                       new_ob['violated'] = True
                       new_ob['violation_ts'] = current_ts
                       new_ob['active'] = False
                       lg.debug(f"OB Calc: New Bull OB immediately violated by its own close.")
                  temp_active_bull.append(new_ob)
             elif new_ob['type'] == "BEAR":
                  # Check if this new BEAR OB is immediately violated by the current candle itself
                  if current_close > new_ob['top']:
                       new_ob['violated'] = True
                       new_ob['violation_ts'] = current_ts
                       new_ob['active'] = False
                       lg.debug(f"OB Calc: New Bear OB immediately violated by its own close.")
                  temp_active_bear.append(new_ob)

        # After processing the current candle, keep only the *still active* OBs that were added up to this candle
        # This is complex state management. A simpler approach for a bot might be to just keep
        # the *most recent* N active OBs that haven't been violated *by any candle so far*.
        # Let's refine: Iterate through the sorted all_order_blocks_chrono. Add to active lists.
        # For each subsequent candle, check if it violates any OBs currently in the active lists.

    # Re-calculate active OBs based on the *entire* DataFrame's price history and initial OBs
    active_bull_boxes = []
    active_bear_boxes = []
    # Iterate through the chronologically sorted list of all potential OBs
    for ob in all_order_blocks_chrono:
         # Check if this OB has been violated by any candle *after* its formation time
         # Filter the DataFrame to candles strictly after the OB's timestamp
         df_after_ob = df.loc[df.index > ob['timestamp']]

         is_violated = False
         violation_ts = None

         if not df_after_ob.empty:
              if ob['type'] == "BULL": # Bull OB violated if any candle closes below its bottom
                  violating_candles = df_after_ob[df_after_ob['close'] < ob['bottom']]
                  if not violating_candles.empty:
                      is_violated = True
                      violation_ts = violating_candles.index[0] # Get timestamp of the first violating candle
              elif ob['type'] == "BEAR": # Bear OB violated if any candle closes above its top
                   violating_candles = df_after_ob[df_after_ob['close'] > ob['top']]
                   if not violating_candles.empty:
                       is_violated = True
                       violation_ts = violating_candles.index[0] # Get timestamp of the first violating candle

         # Update OB state
         ob['violated'] = is_violated
         ob['violation_ts'] = violation_ts
         ob['active'] = not is_violated # Active if not violated

         # Add to active lists if still active, up to max_boxes
         if ob['active']:
              if ob['type'] == "BULL" and len(active_bull_boxes) < ob_max_boxes:
                   active_bull_boxes.append(ob)
              elif ob['type'] == "BEAR" and len(active_bear_boxes) < ob_max_boxes:
                   active_bear_boxes.append(ob)

    # Sort active boxes by recency for signal logic (most recent first)
    active_bull_boxes.sort(key=lambda x: x['timestamp'], reverse=True)
    active_bear_boxes.sort(key=lambda x: x['timestamp'], reverse=True)

    lg.debug(f"OB Calc: Identified {len(active_bull_boxes)} active Bull OBs and {len(active_bear_boxes)} active Bear OBs.")

    return active_bull_boxes + active_bear_boxes # Return a combined list of active boxes

def analyze_market_strategy(df: pd.DataFrame, strategy_params: Dict[str, Any], symbol: str, logger: logging.Logger) -> StrategyAnalysisResults:
    """
    Analyzes market data using Volumatic Trend and Order Blocks based on config parameters.

    Args:
        df: Cleaned DataFrame with 'open', 'high', 'low', 'close', 'volume' (Decimal).
        strategy_params: Dictionary with strategy configuration parameters.
        symbol: The trading symbol.
        logger: Logger instance.

    Returns:
        StrategyAnalysisResults: Dictionary containing analysis results.
    """
    lg = logger
    lg.debug(f"Running strategy analysis for {symbol} (Rows: {len(df)})...")

    # Define required data length for strategy calculation BEFORE attempting calcs
    # Need enough data for VT lengths (length, ATR period, Vol EMA period) AND OB pivot lookbacks
    required_len = max(
        strategy_params.get('vt_length', DEFAULT_VT_LENGTH),
        strategy_params.get('vt_atr_period', DEFAULT_VT_ATR_PERIOD),
        strategy_params.get('vt_vol_ema_length', DEFAULT_VT_VOL_EMA_LENGTH),
        strategy_params.get('ph_left', DEFAULT_PH_LEFT) + strategy_params.get('ph_right', DEFAULT_PH_RIGHT) + 1, # Pivot High window size
        strategy_params.get('pl_left', DEFAULT_PL_LEFT) + strategy_params.get('pl_right', DEFAULT_PL_RIGHT) + 1  # Pivot Low window size
    ) + 10 # Add a buffer for calculation stability

    if len(df) < required_len:
        lg.warning(f"{NEON_YELLOW}Strategy Analysis: Insufficient data for full strategy calculation ({len(df)} rows). Need at least ~{required_len}. Returning default results.{RESET}")
        # Return default/empty results gracefully if data is insufficient
        last_close_dec = df['close'].iloc[-1] if not df.empty and 'close' in df.columns and df['close'].iloc[-1] is not None else Decimal('0')
        return StrategyAnalysisResults(
            dataframe=df, last_close=last_close_dec, current_trend_up=None, trend_just_changed=False,
            active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

    df = df.copy() # Ensure we work on a copy

    # --- Calculate Volumatic Trend Indicators ---
    vt_length = strategy_params.get('vt_length', DEFAULT_VT_LENGTH)
    vt_atr_period = strategy_params.get('vt_atr_period', DEFAULT_VT_ATR_PERIOD)
    vt_vol_ema_length = strategy_params.get('vt_vol_ema_length', DEFAULT_VT_VOL_EMA_LENGTH)
    vt_atr_multiplier = _safe_decimal_conversion(strategy_params.get('vt_atr_multiplier', DEFAULT_VT_ATR_MULTIPLIER))

    # Call the dedicated VT calculation function
    df = calculate_volumatic_trend(df, vt_length, vt_atr_period, vt_vol_ema_length, vt_atr_multiplier, lg)

    # Get last calculated indicator values
    last_candle = df.iloc[-1] if not df.empty else None
    last_close_dec = last_candle['close'] if last_candle is not None and last_candle['close'] is not None else Decimal('0')
    last_trend_line = last_candle['trend_line'] if last_candle is not None else None
    last_vol_ratio = last_candle['vol_ratio'] if last_candle is not None else None
    last_atr = last_candle['atr'] if last_candle is not None else None
    last_upper_band = last_candle['upper_band'] if last_candle is not None else None
    last_lower_band = last_candle['lower_band'] if last_candle is not None else None
    last_vol_norm_int = last_candle['vol_norm_int'] if last_candle is not None else None


    # Determine Volumatic Trend Direction and Change
    current_trend_up: Optional[bool] = None # True if UP, False if DOWN, None if Sideways/Undetermined
    trend_just_changed: bool = False

    # Check for valid last candle data before determining trend
    if last_candle is not None and last_trend_line is not None and last_vol_ratio is not None and \
       last_close_dec is not None and last_trend_line.is_finite() and last_vol_ratio.is_finite():

        # Volumatic Trend UP: Close > Trend Line AND Volume Ratio > 1.0 (or some threshold)
        is_current_up = last_close_dec > last_trend_line and last_vol_ratio > Decimal('1.0')
        # Volumatic Trend DOWN: Close < Trend Line AND Volume Ratio > 1.0
        is_current_down = last_close_dec < last_trend_line and last_vol_ratio > Decimal('1.0')

        if is_current_up:
            current_trend_up = True
        elif is_current_down:
            current_trend_up = False
        else:
            current_trend_up = None # Sideways or weak trend


        # Check if trend just changed (compare with previous candle if available)
        if len(df) >= 2:
             prev_candle = df.iloc[-2]
             prev_trend_line = prev_candle['trend_line'] if 'trend_line' in prev_candle and prev_candle['trend_line'] is not None else None
             prev_vol_ratio = prev_candle['vol_ratio'] if 'vol_ratio' in prev_candle and prev_candle['vol_ratio'] is not None else None
             prev_close_dec = prev_candle['close'] if 'close' in prev_candle and prev_candle['close'] is not None else Decimal('0')

             if prev_trend_line is not None and prev_vol_ratio is not None and prev_close_dec is not None and \
                prev_trend_line.is_finite() and prev_vol_ratio.is_finite():
                  is_prev_up = prev_close_dec > prev_trend_line and prev_vol_ratio > Decimal('1.0')
                  is_prev_down = prev_close_dec < prev_trend_line and prev_vol_ratio > Decimal('1.0')

                  prev_trend_up: Optional[bool] = None
                  if is_prev_up: prev_trend_up = True
                  elif is_prev_down: prev_trend_up = False
                  else: prev_trend_up = None

                  trend_just_changed = (current_trend_up is not None and prev_trend_up is None) or \
                                       (current_trend_up is None and prev_trend_up is not None) or \
                                       (current_trend_up is not None and prev_trend_up is not None and current_trend_up != prev_trend_up)
        else:
            # Not enough data to check for trend change, assume no change for now
            trend_just_changed = False

    else:
        lg.warning("Strategy Analysis: Last candle data incomplete or invalid for trend calculation.")
        current_trend_up = None
        trend_just_changed = False # Cannot determine change if current trend is unknown


    # --- Identify Active Order Blocks ---
    # Call the dedicated OB identification function
    all_active_obs = identify_order_blocks(df, strategy_params, lg)
    active_bull_boxes = [ob for ob in all_active_obs if ob['type'] == 'BULL']
    active_bear_boxes = [ob for ob in all_active_obs if ob['type'] == 'BEAR']

    # Return the comprehensive analysis results
    results: StrategyAnalysisResults = {
        'dataframe': df,
        'last_close': last_close_dec,
        'current_trend_up': current_trend_up,
        'trend_just_changed': trend_just_changed,
        'active_bull_boxes': active_bull_boxes,
        'active_bear_boxes': active_bear_boxes,
        'vol_norm_int': last_vol_norm_int,
        'atr': last_atr,
        'upper_band': last_upper_band,
        'lower_band': last_lower_band,
    }

    return results

def generate_trading_signal(analysis_results: StrategyAnalysisResults, current_position: PositionInfo, strategy_params: Dict[str, Any], logger: logging.Logger) -> SignalResult:
    """
    Generates a trading signal based on StrategyAnalysisResults and current position.

    Args:
        analysis_results: Results from analyze_market_strategy.
        current_position: Enhanced PositionInfo for the symbol.
        strategy_params: Dictionary with strategy configuration parameters.
        logger: Logger instance.

    Returns:
        SignalResult: Dictionary containing the signal, reason, and initial stop/take profit prices.
    """
    lg = logger
    symbol = current_position['symbol']
    pos_side = current_position['side']
    last_close = analysis_results['last_close']
    current_trend_up = analysis_results['current_trend_up']
    trend_just_changed = analysis_results['trend_just_changed']
    active_bull_boxes = analysis_results['active_bull_boxes']
    active_bear_boxes = analysis_results['active_bear_boxes']
    last_atr = analysis_results['atr'] # Use ATR from strategy analysis

    # Ensure essential data is available for signal generation
    if last_close <= POSITION_QTY_EPSILON or last_atr is None or not last_atr.is_finite() or last_atr <= 0:
        lg.warning(f"{NEON_YELLOW}Signal Gen: Critical data missing (LastClose={last_close.normalize()}, ATR={last_atr}). Cannot generate signal. Holding.{RESET}")
        return SignalResult(signal="HOLD", reason="Missing critical data", initial_sl_price=None, initial_tp_price=None)

    # Get strategy and protection parameters
    ob_entry_proximity_factor = _safe_decimal_conversion(strategy_params.get('ob_entry_proximity_factor', DEFAULT_OB_ENTRY_PROXIMITY_FACTOR))
    ob_exit_proximity_factor = _safe_decimal_conversion(strategy_params.get('ob_exit_proximity_factor', DEFAULT_OB_EXIT_PROXIMITY_FACTOR))
    initial_sl_atr_multiple = _safe_decimal_conversion(CONFIG.get('protection', {}).get('initial_stop_loss_atr_multiple', DEFAULT_INITIAL_STOP_LOSS_ATR_MULTIPLE))
    initial_tp_atr_multiple = _safe_decimal_conversion(CONFIG.get('protection', {}).get('initial_take_profit_atr_multiple', DEFAULT_INITIAL_TAKE_PROFIT_ATR_MULTIPLE))

    # Check if factors were parsed correctly
    if ob_entry_proximity_factor is None or ob_entry_proximity_factor < Decimal('1.0'): ob_entry_proximity_factor = Decimal('1.0')
    if ob_exit_proximity_factor is None or ob_exit_proximity_factor < Decimal('1.0'): ob_exit_proximity_factor = Decimal('1.0')
    if initial_sl_atr_multiple is None or initial_sl_atr_multiple <= 0: initial_sl_atr_multiple = Decimal('1.8') # Fallback if invalid
    if initial_tp_atr_multiple is None or initial_tp_atr_multiple < 0: initial_tp_atr_multiple = Decimal('0') # Fallback if invalid


    # --- Calculate Initial SL/TP Prices (Based on last close and ATR) ---
    initial_sl_price: Optional[Decimal] = None
    initial_tp_price: Optional[Decimal] = None

    # This is calculated regardless of signal, but only returned if signal is BUY/SELL
    # Long: SL below, TP above
    calculated_sl_long = last_close - (last_atr * initial_sl_atr_multiple)
    calculated_tp_long = last_close + (last_atr * initial_tp_atr_multiple) if initial_tp_atr_multiple > 0 else None

    # Short: SL above, TP below
    calculated_sl_short = last_close + (last_atr * initial_sl_atr_multiple)
    calculated_tp_short = last_close - (last_atr * initial_tp_atr_multiple) if initial_tp_atr_multiple > 0 else None

    # Ensure calculated SL/TP are valid prices (> 0)
    initial_sl_price_long = calculated_sl_long if calculated_sl_long > POSITION_QTY_EPSILON else None
    initial_tp_price_long = calculated_tp_long if calculated_tp_long is not None and calculated_tp_long > POSITION_QTY_EPSILON else None
    initial_sl_price_short = calculated_sl_short if calculated_sl_short > POSITION_QTY_EPSILON else None
    initial_tp_price_short = calculated_tp_short if calculated_tp_short is not None and calculated_tp_short > POSITION_QTY_EPSILON else None

    lg.debug(f"Signal Gen: Calculated Potential SL(L): {initial_sl_price_long}, TP(L): {initial_tp_price_long}, SL(S): {initial_sl_price_short}, TP(S): {initial_tp_price_short}")


    # --- Exit Conditions (Check BEFORE Entry) ---
    # Strategy Exit: Price action violates an active OB of the *opposite* type
    # This complements native SL/TP/TSL, acting as a potential early exit.

    # Check Bear OBs for Long position exit signal
    if pos_side == 'long' and active_bear_boxes:
        # Find the nearest Bear OB (most recent)
        nearest_bear_ob = active_bear_boxes[0]
        # Exit Long if price is near or above the nearest Bear OB's top
        exit_trigger_price = nearest_bear_ob['top'] / ob_exit_proximity_factor # Calculate price below top based on proximity
        if last_close >= exit_trigger_price:
             lg.info(f"{NEON_YELLOW}Signal Gen: Long Exit Triggered! Price ({last_close.normalize()}) near/above nearest Bear OB top ({nearest_bear_ob['top'].normalize()} * {ob_exit_proximity_factor}).{RESET}")
             return SignalResult(signal="EXIT_LONG", reason="Price near/above Bear OB", initial_sl_price=None, initial_tp_price=None)

    # Check Bull OBs for Short position exit signal
    if pos_side == 'short' and active_bull_boxes:
        # Find the nearest Bull OB (most recent)
        nearest_bull_ob = active_bull_boxes[0]
        # Exit Short if price is near or below the nearest Bull OB's bottom
        exit_trigger_price = nearest_bull_ob['bottom'] * ob_exit_proximity_factor # Calculate price above bottom based on proximity
        if last_close <= exit_trigger_price:
             lg.info(f"{NEON_YELLOW}Signal Gen: Short Exit Triggered! Price ({last_close.normalize()}) near/below nearest Bull OB bottom ({nearest_bull_ob['bottom'].normalize()} * {ob_exit_proximity_factor}).{RESET}")
             return SignalResult(signal="EXIT_SHORT", reason="Price near/below Bull OB", initial_sl_price=None, initial_tp_price=None)


    # --- Entry Conditions (Check ONLY if Flat) ---
    if pos_side == 'none':
        # Entry Signal: Volumatic Trend is aligned AND price is near an active OB of the *aligned* type
        # (e.g., VT UP AND price near a BULL OB)

        # Long Entry Conditions:
        # 1. Volumatic Trend is clearly UP (current_trend_up == True)
        # 2. There is at least one active BULL OB
        # 3. The last close price is near (below or within) the nearest active BULL OB's bottom
        if current_trend_up is True and active_bull_boxes:
             nearest_bull_ob = active_bull_boxes[0] # Most recent Bull OB
             entry_trigger_price = nearest_bull_ob['bottom'] * ob_entry_proximity_factor # Price above bottom based on proximity
             # Check if last close is below the entry trigger price (i.e., near or below the OB)
             if last_close <= entry_trigger_price:
                  lg.info(f"{NEON_GREEN}Long Entry Signal! VT UP, Price ({last_close.normalize()}) near/below nearest Bull OB bottom ({nearest_bull_ob['bottom'].normalize()} * {ob_entry_proximity_factor}).{RESET}")
                  return SignalResult(signal="BUY", reason="VT UP & Price near Bull OB", initial_sl_price=initial_sl_price_long, initial_tp_price=initial_tp_price_long)

        # Short Entry Conditions:
        # 1. Volumatic Trend is clearly DOWN (current_trend_up == False)
        # 2. There is at least one active BEAR OB
        # 3. The last close price is near (above or within) the nearest active BEAR OB's top
        elif current_trend_up is False and active_bear_boxes:
             nearest_bear_ob = active_bear_boxes[0] # Most recent Bear OB
             entry_trigger_price = nearest_bear_ob['top'] / ob_entry_proximity_factor # Price below top based on proximity
             # Check if last close is above the entry trigger price (i.e., near or above the OB)
             if last_close >= entry_trigger_price:
                  lg.info(f"{NEON_RED}Short Entry Signal! VT DOWN, Price ({last_close.normalize()}) near/above nearest Bear OB top ({nearest_bear_ob['top'].normalize()} / {ob_entry_proximity_factor}).{RESET}")
                  return SignalResult(signal="SELL", reason="VT DOWN & Price near Bear OB", initial_sl_price=initial_sl_price_short, initial_tp_price=initial_tp_price_short)

        # If flat and no entry signal
        lg.debug("Signal Gen: Holding. Flat, no entry signal generated by strategy conditions.")
        return SignalResult(signal="HOLD", reason="Strategy conditions not met for entry", initial_sl_price=None, initial_tp_price=None)

    # If in a position and no exit signal was generated above
    lg.debug(f"Signal Gen: Holding. In {pos_side} position, no exit signal generated by strategy conditions.")
    return SignalResult(signal="HOLD", reason="In position, no strategy exit triggered", initial_sl_price=None, initial_tp_price=None)


# --- Trading Execution ---
def calculate_order_quantity(exchange: ccxt.Exchange, market_info: MarketInfo, account_balance_quote: Decimal, current_price: Decimal, stop_loss_price: Decimal, side: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Calculates the order quantity based on risk percentage, account equity,
    estimated stop loss distance, current price, leverage, and market limits.

    Args:
        exchange (ccxt.Exchange): The CCXT exchange instance.
        market_info (MarketInfo): The enhanced market info for the symbol.
        account_balance_quote (Decimal): The total available equity in the quote currency (USDT).
        current_price (Decimal): The current market price for the symbol.
        stop_loss_price (Decimal): The calculated price where the stop loss would trigger.
        side (str): The trade side ('buy' or 'sell').
        logger (logging.Logger): The logger instance.

    Returns:
        The calculated order quantity (Decimal) formatted to market precision,
        or None if calculation is not possible or results in zero/negative quantity.
    """
    lg = logger
    symbol = market_info['symbol']

    # Get config parameters
    risk_percentage = _safe_decimal_conversion(CONFIG.get('risk_per_trade', DEFAULT_CONFIG['risk_per_trade']))
    leverage_cfg = _safe_decimal_conversion(CONFIG.get('leverage', DEFAULT_CONFIG['leverage']))
    min_amount_dec = market_info.get('min_amount_decimal')
    amount_step_dec = market_info.get('amount_precision_step_decimal')
    contract_size_dec = market_info.get('contract_size_decimal', Decimal('1.0')) # Default 1.0 if missing/invalid

    # Ensure essential inputs are valid
    if account_balance_quote is None or account_balance_quote <= POSITION_QTY_EPSILON or \
       current_price <= POSITION_QTY_EPSILON or \
       stop_loss_price is None or stop_loss_price <= POSITION_QTY_EPSILON or \
       risk_percentage is None or risk_percentage <= 0 or risk_percentage > 1 or \
       leverage_cfg is None or leverage_cfg <= 0 or \
       amount_step_dec is None or amount_step_dec <= POSITION_QTY_EPSILON:
        lg.warning(f"{NEON_YELLOW}Qty Calc: Insufficient funds ({account_balance_quote}), invalid prices (Current: {current_price.normalize()}, SL: {stop_loss_price.normalize()}), invalid risk ({risk_percentage}), invalid leverage ({leverage_cfg}), or missing market info step size ({amount_step_dec}). Cannot calculate quantity.{RESET}")
        return None

    try:
        # Risk Amount = Total Equity * Risk Percentage per trade
        risk_amount_quote = account_balance_quote * risk_percentage
        lg.debug(f"Qty Calc: Account Equity: {account_balance_quote.normalize()} {QUOTE_CURRENCY}, Risk %: {risk_percentage}, Risk Amount: {risk_amount_quote.normalize()} {QUOTE_CURRENCY}")

        # Price difference between estimated entry and stop loss
        price_diff = (current_price - stop_loss_price).abs()

        if price_diff <= POSITION_QTY_EPSILON:
            lg.warning(f"{NEON_YELLOW}Qty Calc: Stop Loss price ({stop_loss_price.normalize()}) is too close or equal to current price ({current_price.normalize()}). Risk calculation requires a price difference. Cannot calculate quantity.{RESET}")
            return None

        # Calculate position size in Quote Currency based on risk and price difference
        # Position Value (Quote) = Risk Amount / (% Loss at SL / Leverage)
        # This simplifies to: Position Value (Quote) = Risk Amount * Leverage / % Loss at SL
        # % Loss at SL = (Price Difference / Entry Price)
        # So, Position Value (Quote) = Risk Amount * Leverage / (Price Difference / Entry Price)
        # Position Value (Quote) = Risk Amount * Leverage * Entry Price / Price Difference
        # Quantity = Position Value (Quote) / Entry Price / Contract Size
        # Quantity = (Risk Amount * Leverage * Entry Price / Price Difference) / Entry Price / Contract Size
        # Quantity = Risk Amount * Leverage / Price Difference / Contract Size
        # For linear contracts (contract size 1.0), Quantity = Risk Amount * Leverage / Price Difference
        # For inverse contracts (contract size ~USD value), Quantity = Risk Amount * Leverage / Price Difference / Contract Size (where contract size is in USD)
        # Let's use the more general form: Position Value (Base Currency) = Risk Amount / Price Difference (in base currency terms)
        # For Linear (USDT margined), 1 contract = 1 Base coin. Risk \$10, Price Diff \$1 -> Qty 10 coins. Position Value (USDT) = Qty * Price.
        # For Inverse (BTC margined), 1 contract = $ value of BTC. Risk \$10, Price Diff \$1 -> This needs careful calculation based on how price diff translates to loss per contract.
        # Let's assume Linear contracts for the primary calculation unless specified otherwise.
        # For Linear: Quantity (Base/Contracts) = Risk Amount (Quote) / Price Difference (Quote)

        # For linear contracts where quantity is in base coin: Quantity = Risk Amount / abs(SL_Price - Entry_Price)
        # This quantity represents the base currency amount.
        # Note: This simple formula implies that 1 unit of base currency is valued at the price. This holds for spot and linear contracts.
        try:
            # Calculate quantity in base currency (or contracts for linear where 1 contract = 1 base)
            quantity_base = risk_amount_quote / price_diff
            lg.debug(f"Qty Calc: Price Diff (Entry vs SL): {price_diff.normalize()}, Qty based on Risk: {quantity_base.normalize()}")
        except (DivisionByZero, InvalidOperation) as e:
             lg.error(f"{NEON_RED}Qty Calc: Error calculating quantity from risk: {e}. Price difference likely zero or invalid.{RESET}", exc_info=True)
             return None


        # This calculation method inherently incorporates the risk % and SL distance.
        # We don't need to multiply by leverage here for the *quantity based on risk*.
        # Leverage is used by the exchange to determine margin requirements, not the quantity itself based on risk %.

        # --- Apply market minimum and step size ---
        final_quantity_dec = quantity_base

        # Check against market minimum amount
        if min_amount_dec is not None and final_quantity_dec < min_amount_dec:
             lg.warning(f"{NEON_YELLOW}Qty Calc: Calculated quantity {final_quantity_dec.normalize()} is below market minimum {min_amount_dec.normalize()}. Adjusting up to minimum.{RESET}")
             final_quantity_dec = min_amount_dec

        # Adjust quantity to be a multiple of the step size (rounding down)
        if amount_step_dec > POSITION_QTY_EPSILON and final_quantity_dec > POSITION_QTY_EPSILON:
            # Quantize down to the nearest multiple of amount_step
            final_quantity_dec = (final_quantity_dec / amount_step_dec).quantize(Decimal("1"), rounding=ROUND_DOWN) * amount_step_dec
            lg.debug(f"Qty Calc: Adjusted quantity to market step {amount_step_dec.normalize()}: {final_quantity_dec.normalize()}")

        # Final check on calculated quantity - must be positive after adjustments
        if final_quantity_dec <= POSITION_QTY_EPSILON:
            lg.warning(f"{NEON_YELLOW}Qty Calc: Final calculated quantity {final_quantity_dec.normalize()} is zero or negligible after adjustments. Cannot place order.{RESET}")
            return None

        # --- Estimate Order Value and Check Against Minimum Order Value ---
        # For linear contracts (assuming contract size 1), order value = quantity * price
        # For inverse contracts, order value = quantity * contract size (where contract size is in quote currency)
        # Use market_info.contract_size_decimal for the calculation
        estimated_order_value_quote = final_quantity_dec * current_price * market_info.get('contract_size_decimal', Decimal('1.0'))
        lg.debug(f"Qty Calc: Estimated order value: {estimated_order_value_quote.normalize()} {QUOTE_CURRENCY}")

        # Check against exchange minimum order value (usually specified in quote currency)
        # CCXT market limits might have a 'cost' minimum
        min_cost_dec = market_info.get('min_cost_decimal')
        min_order_value_check = min_cost_dec if min_cost_dec is not None and min_cost_dec > POSITION_QTY_EPSILON else MIN_ORDER_VALUE_USDT # Use config constant as fallback

        if estimated_order_value_quote < min_order_value_check:
             lg.warning(f"{NEON_YELLOW}Qty Calc: Calculated order value {estimated_order_value_quote.normalize()} {QUOTE_CURRENCY} is below market minimum value {min_order_value_check.normalize()} {QUOTE_CURRENCY}. Cannot place order.{RESET}")
             return None

        # --- Estimate Initial Margin Requirement and Check Against Free Margin ---
        # This is a crucial step for Bybit V5, which uses initial margin rate per asset/position.
        # Estimated Initial Margin = Quantity * Entry Price * Initial Margin Rate (for asset) / Contract Size (if inverse)
        # A reliable way to get Initial Margin Rate is from exchange info or account configuration.
        # Simplified Approach: Approximate Initial Margin Required = Order Value (Quote) / Leverage
        # Use the configured leverage, clamped by market max leverage
        max_market_leverage = _safe_market_decimal(market_info.get('limits', {}).get('leverage', {}).get('max'), f"{symbol}.limits.leverage.max", default=Decimal('100'))
        effective_leverage = min(leverage_cfg, max_market_leverage)

        if effective_leverage <= POSITION_QTY_EPSILON:
             lg.error(f"{NEON_RED}Qty Calc: Effective leverage is zero or invalid ({effective_leverage}). Cannot estimate margin required.{RESET}")
             return None

        # Estimated Initial Margin Required for this order: Order Value / Effective Leverage
        estimated_margin_required = estimated_order_value_quote / effective_leverage
        lg.debug(f"Qty Calc: Estimated Initial Margin Required ({effective_leverage.normalize()}x leverage): {estimated_margin_required.normalize()} {QUOTE_CURRENCY}")

        # Fetch *free* balance specifically (usable margin balance) for the quote currency
        # Use the dedicated fetch_balance helper
        balance_data = fetch_balance(exchange, lg)
        free_balance_quote: Optional[Decimal] = None
        if balance_data and QUOTE_CURRENCY in balance_data.get('free', {}):
             free_balance_quote = _safe_decimal_conversion(balance_data['free'][QUOTE_CURRENCY])

        if free_balance_quote is None:
             lg.error(f"{NEON_RED}Qty Calc: Failed to fetch free balance for {QUOTE_CURRENCY}. Cannot perform margin check.{RESET}")
             # Decide whether to stop or proceed without margin check - stopping is safer.
             return None

        # Check if free balance is sufficient with the configured buffer
        # Use required_margin_buffer from config, default to 1.05 (5% buffer)
        required_margin_buffer = _safe_decimal_conversion(CONFIG.get('risk_sizing', {}).get('required_margin_buffer', Decimal('1.05')))
        if required_margin_buffer is None or required_margin_buffer < 1.0: required_margin_buffer = Decimal('1.05') # Fallback

        required_free_margin = estimated_margin_required * required_margin_buffer
        if free_balance_quote < required_free_margin:
            lg.warning(f"{NEON_YELLOW}Qty Calc: Insufficient Free Margin. Need {required_free_margin.normalize()} {QUOTE_CURRENCY} "
                           f"(includes {required_margin_buffer}x buffer) but have {free_balance_quote.normalize()} {QUOTE_CURRENCY}. "
                           f"Cannot place order of size {final_quantity_dec.normalize()}.{RESET}")
            # Send notification about insufficient margin
            if CONFIG.get("notifications", {}).get("enable_notifications", False):
                 notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
                 send_notification(f"Insufficient Margin ({symbol})",
                                   f"Bot needs {required_free_margin.normalize():.2f} {QUOTE_CURRENCY} free margin for order size {final_quantity_dec.normalize()}, but only has {free_balance_quote.normalize():.2f}. Cannot place order.",
                                   lg,
                                   notify_type)
            return None
        else:
             lg.debug(f"Qty Calc: Free margin ({free_balance_quote.normalize()}) sufficient ({required_free_margin.normalize()} needed).")

        lg.info(f"{NEON_GREEN}Qty Calc: Final Calculated Quantity: {final_quantity_dec.normalize()} {market_info.get('base')}. "
                    f"Estimated Order Value: {estimated_order_value_quote.normalize()} {QUOTE_CURRENCY}.{RESET}")

        return final_quantity_dec.normalize() # Return normalized Decimal quantity

    except Exception as e:
        lg.error(f"{NEON_RED}Qty Calc: Unexpected error during quantity calculation: {e}{RESET}", exc_info=True)
        return None # Return None on unexpected errors

def create_order(exchange: ccxt.Exchange, market_info: MarketInfo, type: str, side: str, amount: Decimal, price: Optional[Decimal] = None, stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None, trailing_stop_percentage: Optional[Decimal] = None, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Places an order with native Stop Loss, Take Profit, and Trailing Stop Loss via Bybit V5 API.

    Args:
        exchange (ccxt.Exchange): The CCXT exchange instance.
        market_info (MarketInfo): Enhanced market info for the symbol.
        type (str): Order type ('market', 'limit', etc.). This bot primarily uses 'market'.
        side (str): Order side ('buy' or 'sell').
        amount (Decimal): The quantity to trade.
        price (Optional[Decimal]): The price for limit orders (Optional).
        stop_loss_price (Optional[Decimal]): Native Stop Loss trigger price (Optional, Decimal).
        take_profit_price (Optional[Decimal]): Native Take Profit trigger price (Optional, Decimal).
        trailing_stop_percentage (Optional[Decimal]): Native Trailing Stop percentage (Optional, Decimal, e.g., 0.005 for 0.5%).
        logger (logging.Logger): The logger instance.

    Returns:
        Optional[Dict[str, Any]]: The CCXT order response dictionary if successful, None otherwise.
    """
    lg = logger
    symbol = market_info['symbol']

    if amount <= POSITION_QTY_EPSILON:
        lg.warning(f"{NEON_YELLOW}Create Order: Cannot place order with zero or negligible amount ({amount.normalize()}).{RESET}")
        return None

    # Format amount and prices to exchange precision
    formatted_amount = _format_amount(exchange, market_info, amount)
    if formatted_amount is None:
        lg.error(f"{NEON_RED}Create Order: Failed to format amount {amount.normalize()} for {symbol}. Cannot place order.{RESET}")
        return None

    formatted_price = _format_price(exchange, market_info, price) if price is not None else None
    # Only format stops if they are provided and positive
    formatted_stop_loss = _format_price(exchange, market_info, stop_loss_price) if stop_loss_price is not None and stop_loss_price > POSITION_QTY_EPSILON else None
    formatted_take_profit = _format_price(exchange, market_info, take_profit_price) if take_profit_price is not None and take_profit_price > POSITION_QTY_EPSILON else None

    # Bybit V5 requires 'category': 'linear' or 'inverse' for perpetuals/futures
    # Get category from market_info, default to linear
    params: Dict[str, Any] = {'category': market_info.get('info', {}).get('category', 'linear')}

    # Add Trailing Stop Loss percentage to params for Bybit V5 (linear contracts)
    # Bybit V5 requires 'trailingStop' parameter in create-order for linear perpetuals, value is percentage * 100.
    # CCXT handles the mapping of the standard 'trailingStop' parameter in `createOrder` to the correct exchange param.
    # For Bybit V5 linear, CCXT expects the percentage as a float for its `trailingStop` parameter in the main call or `params['trailingStop']`.
    # Let's add it to params as percentage * 100 as per Bybit V5 documentation.
    # Confirm the exact parameter name CCXT uses for Bybit V5 linear TSL percentage.
    # CCXT source suggests 'trailingStop' in `params` for Bybit V5 linear contracts: https://github.com/ccxt/ccxt/blob/master/python/ccxt/bybit.py#L2782
    # The value should be the percentage * 100 as a float.

    tsl_percentage = trailing_stop_percentage
    if tsl_percentage is not None and tsl_percentage > POSITION_QTY_EPSILON:
        if market_info['is_linear']:
             # Bybit V5 linear uses percentage * 100
             params['trailingStop'] = float(tsl_percentage * 100)
             lg.debug(f"Create Order: Adding native TSL percentage ({tsl_percentage.normalize()}) as params['trailingStop']: {params['trailingStop']}")
             # Add TSL activation price if configured and possible (requires entry price estimate)
             tsl_activation_offset_percent = _safe_decimal_conversion(CONFIG.get('protection', {}).get('trailing_stop_activation_percentage', DEFAULT_TRAILING_STOP_ACTIVATION_PERCENTAGE))
             if tsl_activation_offset_percent is not None and tsl_activation_offset_percent > POSITION_QTY_EPSILON and type == 'market':
                  # For Market orders, the entry price is unknown until filled.
                  # We can't set activationPrice accurately at order creation *unless*
                  # the exchange allows setting it relative to entry or uses a trigger price.
                  # Bybit V5 'activePrice' is usually a specific price. It's best set AFTER entry.
                  # However, CCXT's createOrder might allow it in params. Let's calculate based on current price as estimate.
                  # Activation Price = Current Price * (1 + offset_percent) for buy
                  # Activation Price = Current Price * (1 - offset_percent) for sell
                  # This is an estimate, native TSL activation might use the actual entry price.
                  # Let's hold off on setting activePrice in createOrder for market orders to avoid issues.
                  # It might be possible to add/modify stops AFTER entry.

                  # For now, only log that activation offset is configured but not set at order creation.
                  lg.debug(f"Create Order: TSL activation offset {tsl_activation_offset_percent.normalize()} configured, but not setting 'activePrice' on market order creation for Bybit V5.")
             elif tsl_activation_offset_percent is not None and tsl_activation_offset_percent > POSITION_QTY_EPSILON and type != 'market' and price is not None:
                   # If it's a limit order and we have a price, we can set an activation price
                   offset_factor = Decimal("1") + tsl_activation_offset_percent
                   if side == 'sell':
                       offset_factor = Decimal("1") - tsl_activation_offset_percent
                   activation_price = price * offset_factor
                   formatted_activation_price = _format_price(exchange, market_info, activation_price)
                   if formatted_activation_price:
                        params['activePrice'] = formatted_activation_price # Bybit V5 uses 'activePrice'
                        lg.debug(f"Create Order: Calculated TSL activation price ({tsl_activation_offset_percent.normalize()} offset): {formatted_activation_price}")
                   else:
                        lg.warning(f"{NEON_YELLOW}Create Order: Failed to format TSL activation price {activation_price.normalize()}. Not setting 'activePrice'.{RESET}")

        # Note: Trailing Stop for Inverse contracts on Bybit V5 uses an absolute distance in USD.
        # We would need logic here to convert the percentage or ATR distance to USD.
        # For this bot focused on linear, we primarily support percentage TSL.
        elif market_info['is_inverse']:
             lg.warning(f"{NEON_YELLOW}Create Order: Trailing Stop percentage ({tsl_percentage.normalize()}) configured, but symbol is inverse ({symbol}). Bybit V5 inverse TSL requires absolute USD distance. Not setting native TSL.{RESET}")
             # Do not add 'trailingStop' param if inverse, or add as absolute USD value if conversion logic exists.
             params.pop('trailingStop', None) # Ensure it's not accidentally set

    logger.info(f"{NEON_YELLOW}Conjuring Order | Symbol: {symbol}, Type: {type}, Side: {side}, Amount: {formatted_amount}...{RESET}")
    if formatted_price is not None:
        logger.info(f"  Price: {formatted_price}")
    if formatted_stop_loss is not None:
        logger.info(f"  Native Stop Loss: {formatted_stop_loss}")
    if formatted_take_profit is not None:
        logger.info(f"  Native Take Profit: {formatted_take_profit}")
    if 'trailingStop' in params:
         tsl_value_log = f"{params['trailingStop'] / 100:.2%}" if market_info['is_linear'] else f"{params['trailingStop']} USD"
         lg.info(f"  Native Trailing Stop: {tsl_value_log} (Activation Price: {params.get('activePrice', 'Immediate or TBD')})")

    # Get order fill timeout from config
    order_fill_timeout = CONFIG.get('api_timing', {}).get('order_fill_timeout_seconds', 15)
    stop_confirm_attempts = CONFIG.get('api_timing', {}).get('stop_attach_confirm_attempts', 3)
    stop_confirm_delay = CONFIG.get('api_timing', {}).get('stop_attach_confirm_delay_seconds', 1)


    try:
        # Place the order using CCXT's createOrder, including native SL/TP/TSL parameters
        # CCXT maps stopLoss and takeProfit directly for native Bybit V5 orders
        # Pass stopLoss and takeProfit prices directly to create_order args as strings
        order = exchange.create_order(
            symbol=symbol,
            type=type,
            side=side,
            amount=float(amount), # CCXT expects float for amount/price in the main call
            price=float(price) if price is not None else None, # CCXT expects float
            stopLoss=float(stop_loss_price) if formatted_stop_loss is not None else None, # Pass as float/None, CCXT handles formatting internally for these params
            takeProfit=float(take_profit_price) if formatted_take_profit is not None else None, # Pass as float/None
            params=params # Include the category and TSL params
        )

        order_id = order.get('id')
        order_status = order.get('status')

        lg.info(f"{NEON_GREEN}Order Conjured! | ID: {order_id}, Status: {order_status}{RESET}")
        # Send notification about order placement
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
             send_notification(f"Order Placed ({symbol})",
                               f"Placed {side.upper()} {type.upper()} order for {amount.normalize()} {market_info.get('base')}. Status: {order_status}. ID: {order_id}",
                               lg,
                               notify_type)

        # --- Wait for Market Order Fill (if applicable) ---
        # Market orders should fill quickly, but waiting and fetching details is crucial
        # to get the *actual* average entry price and confirm the order is closed.
        if type == 'market':
            lg.debug(f"Waiting up to {order_fill_timeout}s for market order {order_id} fill...")
            filled_order = None
            try:
                # Poll order status until closed or timeout
                wait_start_time = time.time()
                while time.time() - wait_start_time < order_fill_timeout:
                    time.sleep(1) # Poll every second
                    # Fetch order status - requires category for Bybit V5
                    fetched_order = exchange.fetch_order(order_id, symbol, params={'category': params['category']})
                    if fetched_order and fetched_order.get('status') == 'closed':
                        filled_order = fetched_order
                        lg.debug(f"Market order {order_id} detected as 'closed'.")
                        break
                    lg.debug(f"Order {order_id} status: {fetched_order.get('status')}. Filled: {fetched_order.get('filled',0)}/{fetched_order.get('amount',0)}")


                if filled_order:
                    filled_qty = _safe_decimal_conversion(filled_order.get('filled', '0'))
                    avg_price = _safe_decimal_conversion(filled_order.get('average', '0'))
                    if filled_qty is not None and avg_price is not None and filled_qty >= amount * (Decimal("1") - POSITION_QTY_EPSILON) and avg_price > POSITION_QTY_EPSILON: # Check if filled significantly close to requested amount
                        lg.success(f"{NEON_GREEN}Market order {order_id} filled! Filled Qty: {filled_qty.normalize()}, Avg Price: {avg_price.normalize()}{RESET}")
                        # Update the returned order object with potentially more accurate filled info
                        order['filled'] = filled_qty
                        order['average'] = avg_price
                        order['status'] = 'closed' # Ensure status is marked closed
                        # After successful market order fill, wait briefly then verify position/stops
                        time.sleep(CONFIG.get('position_confirm_delay_seconds', POSITION_CONFIRM_DELAY_SECONDS))
                        # No need to explicitly call confirm_stops_attached here, the main loop's
                        # get_current_position will fetch the latest state including stops
                        # attached to the position, and the state will be logged.
                        return order # Return the updated order dictionary
                    else:
                        lg.warning(f"{NEON_YELLOW}Market order {order_id} status is 'closed' but filled quantity ({filled_qty.normalize() if filled_qty is not None else 'N/A'}) is less than requested ({amount.normalize()}) or avg price zero. Potential partial fill or data issue.{RESET}")
                        # It's closed, even if partially filled or price is weird. Return the fetched order.
                        # Still wait briefly and fetch position to see what happened.
                        time.sleep(CONFIG.get('position_confirm_delay_seconds', POSITION_CONFIRM_DELAY_SECONDS))
                        return filled_order
                else:
                    lg.error(f"{NEON_RED}Market order {order_id} did not report 'closed' status after {order_fill_timeout}s. Potential issue or partial fill. Manual check required!{RESET}")
                    # Return the last fetched state if available, else the initial order response
                    # Wait briefly and fetch position to see what state the exchange is in.
                    time.sleep(CONFIG.get('position_confirm_delay_seconds', POSITION_CONFIRM_DELAY_SECONDS))
                    return fetched_order if 'fetched_order' in locals() and fetched_order else order

            except Exception as fill_check_err:
                 lg.error(f"{NEON_RED}Error while waiting/checking market order fill for {order_id}: {fill_check_err}{RESET}", exc_info=True)
                 # Return the initial order response as fallback if fill check fails
                 # Wait briefly and fetch position to see what happened.
                 time.sleep(CONFIG.get('position_confirm_delay_seconds', POSITION_CONFIRM_DELAY_SECONDS))
                 return order

        else: # For non-market orders (limit etc.), just return the initial response
            # For limit orders with native stops, we'd need logic to confirm the order is active/open
            # and that the stops are attached to the order itself (not position).
            # For this bot, we assume MARKET entry, so this else block is less critical.
            lg.debug(f"Non-market order type '{type}'. Skipping fill confirmation wait.")
            return order

    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Order Failed ({symbol}): Insufficient funds - {e}{RESET}")
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
             send_notification(f"Order FAILED ({symbol})", f"Failed to place {side.upper()} order: Insufficient Funds.", lg, notify_type)
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Order Failed ({symbol}): Invalid order request - {e}. Check quantity precision, price, stop/TP params, min/max limits, contract status.{RESET}")
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
             send_notification(f"Order FAILED ({symbol})", f"Failed to place {side.upper()} order: Invalid Request ({e}).", lg, notify_type)
    except ccxt.DDoSProtection as e:
        lg.warning(f"{NEON_YELLOW}Order Failed ({symbol}): Rate limit hit - {e}. Backing off.{RESET}")
        time.sleep(exchange.rateLimit / 1000 + 1) # Wait a bit longer than rate limit
    except ccxt.RequestTimeout as e:
        lg.warning(f"{NEON_YELLOW}Order Failed ({symbol}): Request timed out - {e}. Network issue or high load.{RESET}")
    except ccxt.NetworkError as e:
        lg.warning(f"{NEON_YELLOW}Order Failed ({symbol}): Network error - {e}. Check connection.{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Order Failed ({symbol}): Exchange error - {e}. Check account status, symbol status, Bybit system status, or API permissions for stops/TSL.{RESET}")
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
             send_notification(f"Order FAILED ({symbol})", f"Failed to place {side.upper()} order: Exchange Error ({e}).", lg, notify_type)
    except Exception as e:
        lg.error(f"{NEON_RED}Order Failed ({symbol}): Unexpected error - {e}{RESET}", exc_info=True)
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
             send_notification(f"Order FAILED ({symbol})", f"Failed to place {side.upper()} order: Unexpected Error: {type(e).__name__}.", lg, notify_type)

    return None # Return None if order placement or fill confirmation failed


def close_position(exchange: ccxt.Exchange, market_info: MarketInfo, current_position: PositionInfo, logger: logging.Logger) -> bool:
    """
    Closes the current active position using a market order.

    Args:
        exchange (ccxt.Exchange): The CCXT exchange instance.
        market_info (MarketInfo): Enhanced market info for the symbol.
        current_position (PositionInfo): Enhanced PositionInfo of the current position.
        logger (logging.Logger): The logger instance.

    Returns:
        bool: True if the close order is successfully placed (and assumed filled as market)
              or if the position is confirmed closed after the attempt, False otherwise.
    """
    lg = logger
    symbol = market_info['symbol']
    pos_side = current_position['side']
    pos_qty = current_position['size_decimal']

    if pos_side == 'none' or pos_qty <= POSITION_QTY_EPSILON:
        lg.info("Close Position: No active position to close.")
        return True # Already flat

    # Determine the side for the closing order (opposite of position side)
    close_side = 'sell' if pos_side == 'long' else 'buy'

    lg.warning(f"{NEON_YELLOW}Initiating Position Closure: Closing {pos_side} position for {symbol} (Qty: {pos_qty.normalize()}) with market order ({close_side})...{RESET}")

    # Get order fill timeout from config
    order_fill_timeout = CONFIG.get('api_timing', {}).get('order_fill_timeout_seconds', 15)
    post_close_delay = CONFIG.get('api_timing', {}).get('post_close_delay_seconds', 3) # Assuming this config exists or define it.

    try:
        # Bybit V5 closing a position uses the *opposite* side Market order.
        # Quantity should be the full position quantity.
        # CCXT's createOrder with 'reduceOnly': True is the standard way.
        # Bybit V5 also supports 'positionIdx' in params to specify which position to close in Hedge mode,
        # but for One-Way mode (positionIdx=0), it's usually not strictly necessary if using reduceOnly.
        # Let's add 'positionIdx': 0 to be explicit for One-Way.
        # Get category from market_info, default to linear
        params = {'category': market_info.get('info', {}).get('category', 'linear'), 'reduceOnly': True, 'positionIdx': 0} # Explicit for One-Way

        # Use the exact position quantity (absolute value)
        formatted_qty = _format_amount(exchange, market_info, pos_qty.abs())
        if formatted_qty is None:
            lg.error(f"{NEON_RED}Close Position: Failed to format position quantity {pos_qty.abs().normalize()} for {symbol}. Cannot place close order.{RESET}")
            return False


        lg.debug(f"Placing market order to close position. Symbol: {symbol}, Side: {close_side}, Quantity: {formatted_qty}, Params: {params}")

        close_order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=close_side,
            amount=float(pos_qty.abs()), # CCXT expects float for amount
            params=params
        )

        order_id = close_order.get('id')
        order_status = close_order.get('status')

        lg.info(f"{NEON_GREEN}Position Close Order Conjured! | ID: {order_id}, Status: {order_status}{RESET}")
        # Send notification about close order placement
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
             send_notification(f"Position Close Order ({symbol})",
                               f"Placed {close_side.upper()} MARKET order to close {pos_side} position. Qty: {pos_qty.abs().normalize()}. ID: {order_id}. Status: {order_status}.",
                               lg,
                               notify_type)


        # For Market Close orders, wait for confirmation of fill
        lg.debug(f"Waiting up to {order_fill_timeout}s for close market order {order_id} fill...")
        filled_close_order = None
        try:
            wait_start_time = time.time()
            while time.time() - wait_start_time < order_fill_timeout:
                time.sleep(1) # Poll every second
                # Use fetchOrder with category and positionIdx if needed for this specific order on Bybit V5
                fetched_order = exchange.fetch_order(order_id, symbol, params={'category': params['category']})
                if fetched_order and fetched_order.get('status') == 'closed':
                    filled_close_order = fetched_order
                    lg.debug(f"Close market order {order_id} detected as 'closed'.")
                    break
                lg.debug(f"Close Order {order_id} status: {fetched_order.get('status')}. Filled: {fetched_order.get('filled',0)}/{fetched_order.get('amount',0)}")


            if filled_close_order:
                filled_qty = _safe_decimal_conversion(filled_close_order.get('filled', '0'))
                avg_price = _safe_decimal_conversion(filled_close_order.get('average', '0'))
                # Check if filled quantity is close enough to the position quantity
                if filled_qty is not None and filled_qty >= pos_qty.abs() * (Decimal("1") - POSITION_QTY_EPSILON):
                    lg.success(f"{NEON_GREEN}Position close order {order_id} filled! Filled Qty: {filled_qty.normalize()}.{RESET}")
                    # Give the exchange a moment to update the position state after fill
                    time.sleep(post_close_delay)
                    # Re-fetch position to confirm it's flat
                    final_position_state = get_current_position(exchange, market_info, lg)
                    if final_position_state['side'] == 'none':
                         lg.success(f"{NEON_GREEN}Position for {symbol} confirmed closed after fill.{RESET}")
                         return True # Successfully placed and filled close order, and position is confirmed flat
                    else:
                         lg.warning(f"{NEON_YELLOW}Position close order {order_id} filled, but position check still shows {final_position_state['side']} with Qty {final_position_state['size_decimal'].normalize()}. Manual check recommended.{RESET}")
                         if CONFIG.get("notifications", {}).get("enable_notifications", False):
                             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
                             send_notification(f"Position Close WARNING ({symbol})",
                                               f"Close order {order_id} filled, but position {final_position_state['side']} Qty {final_position_state['size_decimal'].normalize()} still reported. Manual check required.",
                                               lg,
                                               notify_type)
                         return True # Assume success but warn

                else:
                     lg.warning(f"{NEON_YELLOW}Position close order {order_id} status is 'closed' but filled quantity ({filled_qty.normalize() if filled_qty is not None else 'N/A'}) is significantly less than position size ({pos_qty.abs().normalize()}). Position may not be fully closed. Manual check required!{RESET}")
                     # It's closed according to the exchange, but potentially partial fill.
                     # Still wait briefly and re-check position state.
                     time.sleep(post_close_delay)
                     final_position_state = get_current_position(exchange, market_info, lg)
                     if final_position_state['side'] == 'none':
                         lg.success(f"{NEON_GREEN}Position for {symbol} confirmed closed after partial fill warning.{RESET}")
                         return True # Position is indeed closed
                     else:
                          lg.error(f"{NEON_RED}Position close order {order_id} potentially partially filled, and position still active with Qty {final_position_state['size_decimal'].normalize()}. Manual intervention required!{RESET}")
                          if CONFIG.get("notifications", {}).get("enable_notifications", False):
                               notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
                               send_notification(f"Position Close FAILED ({symbol})",
                                                 f"Close order {order_id} potentially partial, pos still active Qty {final_position_state['size_decimal'].normalize()}. Manual intervention required.",
                                                 lg,
                                                 notify_type)
                          return False # Position still active, indicate failure
            else:
                lg.error(f"{NEON_RED}Position close market order {order_id} did not report 'closed' status after {order_fill_timeout}s. Position may not be closed. Manual check required!{RESET}")
                # Decide how to handle - returning False might trigger another close attempt if position still exists.
                # Let's re-check position state immediately after timeout
                time.sleep(post_close_delay) # Small delay then check
                post_close_pos = get_current_position(exchange, market_info, lg)
                if post_close_pos['side'] == 'none':
                     lg.success(f"{NEON_GREEN}Position for {symbol} confirmed closed after fill check timeout.{RESET}")
                     return True # Position is indeed closed
                else:
                     lg.error(f"{NEON_RED}Position for {symbol} still active after fill check timeout. Quantity remaining: {post_close_pos['size_decimal'].normalize()}. Manual intervention required!{RESET}")
                     if CONFIG.get("notifications", {}).get("enable_notifications", False):
                           notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
                           send_notification(f"Position Close FAILED ({symbol})",
                                             f"Close order {order_id} timed out, pos still active Qty {post_close_pos['size_decimal'].normalize()}. Manual intervention required.",
                                             lg,
                                             notify_type)
                     return False # Position still active, indicate failure
        except Exception as fill_check_err:
             lg.error(f"{NEON_RED}Error while waiting/checking close market order fill for {order_id}: {fill_check_err}{RESET}", exc_info=True)
             # Assume failure if check fails, let the main loop potentially retry closing
             # Still wait briefly and re-check position state.
             time.sleep(post_close_delay)
             final_position_state = get_current_position(exchange, market_info, lg)
             if final_position_state['side'] == 'none':
                  lg.success(f"{NEON_GREEN}Position for {symbol} confirmed closed after fill check error.{RESET}")
                  return True # Position is indeed closed
             else:
                 lg.error(f"{NEON_RED}Position for {symbol} still active after fill check error. Quantity remaining: {final_position_state['size_decimal'].normalize()}. Manual intervention required!{RESET}")
                 if CONFIG.get("notifications", {}).get("enable_notifications", False):
                      notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
                      send_notification(f"Position Close FAILED ({symbol})",
                                        f"Error checking close order {order_id}, pos still active Qty {final_position_state['size_decimal'].normalize()}. Manual intervention required.",
                                        lg,
                                        notify_type)
                 return False


    except ccxt.InsufficientFunds as e:
        # This can happen if trying to close with wrong side/params, or margin call issues
        lg.error(f"{NEON_RED}Close Order Failed ({symbol}): Insufficient funds (during close?) - {e}. Check position, margin, order params.{RESET}")
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
             send_notification(f"Position Close FAILED ({symbol})", f"Failed to place close order: Insufficient Funds ({e}).", lg, notify_type)
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Close Order Failed ({symbol}): Invalid order request - {e}. Check quantity, side, reduceOnly param, positionIdx.{RESET}")
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
             send_notification(f"Position Close FAILED ({symbol})", f"Failed to place close order: Invalid Request ({e}).", lg, notify_type)
    except ccxt.DDoSProtection as e:
        lg.warning(f"{NEON_YELLOW}Close Order Failed ({symbol}): Rate limit hit - {e}. Backing off.{RESET}")
        time.sleep(exchange.rateLimit / 1000 + 1)
    except ccxt.RequestTimeout as e:
        lg.warning(f"{NEON_YELLOW}Close Order Failed ({symbol}): Request timed out - {e}. Network issue.{RESET}")
    except ccxt.NetworkError as e:
        lg.warning(f"{NEON_YELLOW}Close Order Failed ({symbol}): Network error - {e}. Check connection.{RESET}")
    except ccxt.ExchangeError as e:
        # Bybit error like "Order quantity below minimum" or "Position size is 0" can happen if already closed
        lg.error(f"{NEON_RED}Close Order Failed ({symbol}): Exchange error - {e}. Position might already be closed or another issue.{RESET}")
        # Re-check position immediately if exchange error might indicate already closed state
        time.sleep(post_close_delay)
        post_close_pos = get_current_position(exchange, market_info, lg)
        if post_close_pos['side'] == 'none':
             lg.success(f"{NEON_GREEN}Position for {symbol} confirmed closed after ExchangeError during close attempt.{RESET}")
             return True # Position is indeed closed
        else:
             if CONFIG.get("notifications", {}).get("enable_notifications", False):
                  notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
                  send_notification(f"Position Close FAILED ({symbol})",
                                    f"Failed to place close order: Exchange Error ({e}). Pos still active Qty {post_close_pos['size_decimal'].normalize()}. Manual intervention required.",
                                    lg,
                                    notify_type)
             return False # Position still active, indicate failure
    except Exception as e:
        lg.error(f"{NEON_RED}Close Order Failed ({symbol}): Unexpected error - {e}{RESET}", exc_info=True)
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
             send_notification(f"Position Close FAILED ({symbol})", f"Failed to place close order: Unexpected Error: {type(e).__name__}.", lg, notify_type)

    return False # Return False if order placement or confirmation failed

def cancel_all_orders_for_symbol(exchange: ccxt.Exchange, market_info: MarketInfo, reason: str, logger: logging.Logger) -> int:
    """
    Attempts to cancel all open orders for a specific symbol.

    Args:
        exchange (ccxt.Exchange): The CCXT exchange instance.
        market_info (MarketInfo): Enhanced market info for the symbol.
        reason (str): A string indicating why cancellation is being attempted (for logging).
        logger (logging.Logger): The logger instance.

    Returns:
        int: The number of cancellation attempts made (not necessarily successful cancellations).
    """
    lg = logger
    symbol = market_info['symbol']
    lg.info(f"{NEON_BLUE}Order Cleanup Ritual: Initiating for {symbol} (Reason: {reason})...{RESET}")
    attempts = 0
    try:
        # Bybit V5 cancelAllOrders requires 'category' parameter for futures
        # It also *strongly* prefers the exchange-specific market ID.
        # Use category from market_info, default to linear
        category = market_info.get('info', {}).get('category', 'linear')
        market_id = market_info.get('id', symbol) # Use exchange ID if available

        # CCXT's cancel_all_orders method
        lg.warning(f"{NEON_YELLOW}Order Cleanup: Attempting to cancel ALL open orders for {symbol} (Category: {category})...{RESET}")
        attempts += 1
        # Bybit V5 cancelAllOrders takes symbol and category in params
        # CCXT map: https://github.com/ccxt/ccxt/blob/master/python/ccxt/bybit.py#L1860
        # It seems to pass symbol and category directly in params.
        response = exchange.cancel_all_orders(symbol=symbol, params={'category': category})

        # Bybit V5 cancelAllOrders response structure can vary, often empty or confirms actions.
        # Success is generally indicated by no exception and a non-error response structure.
        lg.info(f"{NEON_GREEN}Order Cleanup: cancel_all_orders request sent for {symbol}. Response: {response}{RESET}")
        lg.info(f"{NEON_GREEN}Order Cleanup Ritual Finished for {symbol}. Attempt successful (reported {attempts} actions/attempts).{RESET}")
        return attempts # Report attempts made

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
        lg.warning(f"{NEON_YELLOW}Order Cleanup Error for {symbol}: {type(e).__name__} - {e}. Could not cancel orders.{RESET}")
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
             send_notification(f"Order Cancel FAILED ({symbol})", f"Failed to cancel all orders: {type(e).__name__}.", lg, notify_type)
    except ccxt.NotSupported:
        lg.error(f"{NEON_RED}Order Cleanup Error: Exchange '{exchange.id}' does not support cancelAllOrders method. Cannot perform cleanup.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Order Cleanup Unexpected Error for {symbol}: {type(e).__name__} - {e}{RESET}", exc_info=True)
        if CONFIG.get("notifications", {}).get("enable_notifications", False):
             notify_type = CONFIG.get('notifications', {}).get('notification_type', 'email')
             send_notification(f"Order Cancel FAILED ({symbol})", f"Failed to cancel all orders: Unexpected Error: {type(e).__name__}.", lg, notify_type)

    return attempts # Return attempts even on failure

# --- Position Management Logic (Break-Even, Trailing Stop Update) ---
# Note: Native TSL/TP/SL are placed with the entry order.
# Break-Even logic needs to *update* the existing native Stop Loss order.
# Trailing Stop Loss *activation* can be client-side, and then we rely on the native TSL.

def manage_position_protection(exchange: ccxt.Exchange, market_info: MarketInfo, current_position: PositionInfo, strategy_analysis: StrategyAnalysisResults, logger: logging.Logger) -> PositionInfo:
    """
    Manages position protection orders (Break-Even, Trailing Stop activation/check).
    Checks if BE or TSL conditions are met and attempts to update the Stop Loss order natively.

    Args:
        exchange (ccxt.Exchange): The CCXT exchange instance.
        market_info (MarketInfo): Enhanced market info for the symbol.
        current_position (PositionInfo): Enhanced PositionInfo of the current position (will be modified in place).
        strategy_analysis (StrategyAnalysisResults): Latest strategy analysis results.
        logger (logging.Logger): The logger instance.

    Returns:
        PositionInfo: The updated PositionInfo object (potentially with BE/TSL state flags changed).
    """
    lg = logger
    symbol = market_info['symbol']

    # Do nothing if not in an active position
    if current_position['side'] == 'none' or abs(current_position['size_decimal']) <= POSITION_QTY_EPSILON:
        return current_position

    pos_side = current_position['side']
    entry_price = current_position.get('entryPrice_decimal')
    mark_price = current_position.get('markPrice_decimal') # Use mark price for profit evaluation
    current_sl_price = current_position.get('stopLossPrice_dec') # Get current native SL price (parsed)
    current_tp_price = current_position.get('takeProfitPrice_dec') # Get current native TP price (parsed)
    # Bybit V5 'trailingStop' in position info is the TRIGGER price, not the distance.
    # If this is non-zero, native TSL is likely already active or has been set.
    # We don't explicitly manage TSL *moves* client-side if native TSL is set;
    # we only manage the *activation* price check (if needed client-side before native activation)
    # and rely on the exchange for the actual trailing.
    # Check if native TSL is set (trigger price > 0)
    native_tsl_set = current_position.get('trailingStopPrice_dec') is not None and current_position['trailingStopPrice_dec'] > POSITION_QTY_EPSILON

    last_atr = strategy_analysis.get('atr') # Use latest ATR from analysis

    # Ensure essential prices are available
    if entry_price is None or mark_price is None or last_atr is None or not last_atr.is_finite() or last_atr <= 0:
        lg.warning(f"{NEON_YELLOW}Protection: Cannot manage position protection for {symbol}. Missing Entry Price ({entry_price}), Mark Price ({mark_price}), or valid ATR ({last_atr}).{RESET}")
        return current_position # Cannot manage protection without these

    # Get protection parameters from config
    enable_break_even = CONFIG.get('protection', {}).get('enable_break_even', DEFAULT_ENABLE_BREAK_EVEN)
    be_trigger_atr_multiple = _safe_decimal_conversion(CONFIG.get('protection', {}).get('break_even_trigger_atr_multiple', DEFAULT_BREAK_EVEN_TRIGGER_ATR_MULTIPLE))
    be_offset_ticks = CONFIG.get('protection', {}).get('break_even_offset_ticks', DEFAULT_BREAK_EVEN_OFFSET_TICKS)

    enable_trailing_stop = CONFIG.get('protection', {}).get('enable_trailing_stop', DEFAULT_ENABLE_TRAILING_STOP)
    tsl_activation_percentage = _safe_decimal_conversion(CONFIG.get('protection', {}).get('trailing_stop_activation_percentage', DEFAULT_TRAILING_STOP_ACTIVATION_PERCENTAGE))
    tsl_callback_rate = _safe_decimal_conversion(CONFIG.get('protection', {}).get('trailing_stop_callback_rate', DEFAULT_TRAILING_STOP_CALLBACK_RATE)) # Only needed if we set TSL initially, not for BE/TSL updates client-side

    # Check BE/TSL parameters
    if be_trigger_atr_multiple is None or be_trigger_atr_multiple <= 0: enable_break_even = False # Disable BE if trigger invalid
    if be_offset_ticks is None or be_offset_ticks < 0: be_offset_ticks = 0 # Default offset to 0 if invalid
    if tsl_activation_percentage is None or tsl_activation_percentage < 0: enable_trailing_stop = False # Disable TSL if activation invalid


    # --- Break-Even Logic ---
    # Move SL to break-even if profit target is reached AND BE is enabled AND BE hasn't been activated for this position
    if enable_break_even and not current_position['be_activated']:
        # Calculate profit needed to trigger BE (in Quote currency per Base unit * Contract Size)
        # Profit Target = Entry Price + (ATR * BE Trigger Multiple) for Long
        # Profit Target = Entry Price - (ATR * BE Trigger Multiple) for Short
        profit_target_price: Optional[Decimal] = None
        if pos_side == 'long':
            profit_target_price = entry_price + (last_atr * be_trigger_atr_multiple)
            # Check if current Mark Price has reached or exceeded the target
            if mark_price >= profit_target_price:
                lg.info(f"{NEON_YELLOW}Protection ({symbol}): Break-Even Triggered for Long! Mark Price ({mark_price.normalize()}) >= Target ({profit_target_price.normalize()}).{RESET}")
                # Calculate new SL price (Entry Price + Offset in ticks)
                # Need price tick size from market info
                price_tick_size = market_info.get('price_precision_step_decimal')
                if price_tick_size is not None and price_tick_size > POSITION_QTY_EPSILON:
                    be_sl_price = entry_price + (price_tick_size * Decimal(be_offset_ticks))
                    # Ensure new SL price is above current SL price (never move SL backwards)
                    if current_sl_price is None or be_sl_price > current_sl_price:
                        lg.info(f"{NEON_YELLOW}Protection ({symbol}): Attempting to move SL to Break-Even ({be_sl_price.normalize()})...{RESET}")
                        # Attempt to update the stop loss on the exchange
                        if update_native_stop_loss(exchange, market_info, be_sl_price, current_tp_price, native_tsl_set, lg):
                             lg.success(f"{NEON_GREEN}Protection ({symbol}): Stop Loss successfully updated to Break-Even: {be_sl_price.normalize()}{RESET}")
                             current_position['be_activated'] = True # Mark BE as activated for this position
                        else:
                            lg.error(f"{NEON_RED}Protection ({symbol}): Failed to update Stop Loss to Break-Even.{RESET}")
                    else:
                        lg.debug(f"Protection ({symbol}): Calculated BE SL ({be_sl_price.normalize()}) is not higher than current SL ({current_sl_price.normalize()}). Skipping update.")
                else:
                    lg.error(f"{NEON_RED}Protection ({symbol}): Cannot calculate Break-Even SL offset. Price tick size missing/invalid.{RESET}")

        elif pos_side == 'short':
            profit_target_price = entry_price - (last_atr * be_trigger_atr_multiple)
            # Check if current Mark Price has reached or fallen below the target
            if mark_price <= profit_target_price:
                lg.info(f"{NEON_YELLOW}Protection ({symbol}): Break-Even Triggered for Short! Mark Price ({mark_price.normalize()}) <= Target ({profit_target_price.normalize()}).{RESET}")
                 # Calculate new SL price (Entry Price - Offset in ticks)
                price_tick_size = market_info.get('price_precision_step_decimal')
                if price_tick_size is not None and price_tick_size > POSITION_QTY_EPSILON:
                    be_sl_price = entry_price - (price_tick_size * Decimal(be_offset_ticks))
                     # Ensure new SL price is below current SL price (never move SL backwards)
                    if current_sl_price is None or be_sl_price < current_sl_price:
                         lg.info(f"{NEON_YELLOW}Protection ({symbol}): Attempting to move SL to Break-Even ({be_sl_price.normalize()})...{RESET}")
                         # Attempt to update the stop loss on the exchange
                         if update_native_stop_loss(exchange, market_info, be_sl_price, current_tp_price, native_tsl_set, lg):
                              lg.success(f"{NEON_GREEN}Protection ({symbol}): Stop Loss successfully updated to Break-Even: {be_sl_price.normalize()}{RESET}")
                              current_position['be_activated'] = True # Mark BE as activated for this position
                         else:
                            lg.error(f"{NEON_RED}Protection ({symbol}): Failed to update Stop Loss to Break-Even.{RESET}")
                    else:
                        lg.debug(f"Protection ({symbol}): Calculated BE SL ({be_sl_price.normalize()}) is not lower than current SL ({current_sl_price.normalize()}). Skipping update.")
                else:
                    lg.error(f"{NEON_RED}Protection ({symbol}): Cannot calculate Break-Even SL offset. Price tick size missing/invalid.{RESET}")

    # --- Trailing Stop Loss Activation (Client-Side Check) ---
    # This is less critical if native TSL is used and set initially, as the exchange handles activation.
    # However, if the exchange's TSL needs a
