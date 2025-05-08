# File: utils.py
"""
Utility functions and constants for the application.

This module includes:
- Configuration constants (file paths, API settings, default values).
- Timezone handling with a fallback for older Python versions.
- Custom logging formatter for redacting sensitive information.
- Helper functions for determining price precision and minimum tick size from market data.
- Color constants for terminal output using Colorama.
- Formatting utilities for application-specific data types (e.g., signals).
"""

import logging
import os
import datetime as dt  # Use dt alias for datetime module
from decimal import Decimal, getcontext, InvalidOperation
from typing import Any, Dict, Optional, Type

from colorama import Fore, Style, init

# --- Module-level logger ---
_module_logger = logging.getLogger(__name__)

# --- Timezone Handling ---
_ActualZoneInfo: Type[
    dt.tzinfo
]  # This will hold the class to use for timezones (zoneinfo.ZoneInfo or FallbackZoneInfo)


class FallbackZoneInfo(dt.tzinfo):
    """
    A basic, UTC-only fallback implementation for timezone handling,
    mimicking zoneinfo.ZoneInfo for the 'UTC' timezone.
    It is used if the 'zoneinfo' module (Python 3.9+) is not available.
    This class only truly supports 'UTC'; other timezone keys will default to UTC.
    """

    _offset: dt.timedelta
    _name: str

    def __init__(self, key: str):
        super().__init__()  # Call tzinfo's __init__

        # For this fallback, we only truly support "UTC".
        # key is expected to be a string as per type hint.
        if key.upper() == "UTC":
            self._offset = dt.timedelta(0)
            self._name = "UTC"
        else:
            _module_logger.warning(
                f"FallbackZoneInfo initialized with key '{key}' which is not 'UTC'. "
                f"This fallback only supports UTC. Effective timezone will be UTC."
            )
            self._offset = dt.timedelta(0)
            self._name = "UTC"  # Effective name is UTC

    def fromutc(self, dt_obj: dt.datetime) -> dt.datetime:
        """
        Called by datetime.astimezone() after adjusting dt_obj to UTC.
        Since this class represents UTC, no further adjustment is needed.
        dt_obj.tzinfo is self when this method is called.
        """
        if not isinstance(dt_obj, dt.datetime):
            raise TypeError("fromutc() requires a datetime argument")
        # dt_obj is already in UTC wall time, with tzinfo=self.
        return dt_obj.replace(tzinfo=self)  # Ensure tzinfo is correctly self

    def utcoffset(self, dt_obj: Optional[dt.datetime]) -> dt.timedelta:
        """Returns the UTC offset for this timezone."""
        # dt_obj can be None as per tzinfo.utcoffset signature
        return self._offset

    def dst(self, dt_obj: Optional[dt.datetime]) -> Optional[dt.timedelta]:
        """Returns the Daylight Saving Time offset (always 0 for UTC)."""
        # dt_obj can be None as per tzinfo.dst signature
        return dt.timedelta(0)

    def tzname(self, dt_obj: Optional[dt.datetime]) -> Optional[str]:
        """Returns the name of the timezone."""
        # dt_obj can be None as per tzinfo.tzname signature
        return self._name

    def __repr__(self) -> str:
        return f"<FallbackZoneInfo name='{self._name}'>"

    def __str__(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FallbackZoneInfo):
            return NotImplemented
        # All FallbackZoneInfo instances are effectively UTC and thus equal
        # if they have the same _name and _offset (which they will, by construction).
        return self._name == other._name and self._offset == other._offset

    def __hash__(self) -> int:
        # All FallbackZoneInfo instances hash the same if _name is always "UTC".
        return hash((self._name, self._offset))


try:
    from zoneinfo import ZoneInfo as _ZoneInfo_FromModule  # Import with a temporary name (Python 3.9+)

    _ActualZoneInfo = _ZoneInfo_FromModule  # Assign to the variable with the correct type hint
except ImportError:
    _module_logger.warning(
        "Module 'zoneinfo' not found (requires Python 3.9+ and possibly 'tzdata' package). "
        "Using a basic UTC-only fallback for timezone handling. "
        "For full timezone support on older Python, consider installing 'pytz'."
    )
    _ActualZoneInfo = FallbackZoneInfo  # FallbackZoneInfo is Type[dt.tzinfo]

# Global timezone object, lazily initialized
TIMEZONE: Optional[dt.tzinfo] = None

# --- Decimal Context ---
getcontext().prec = 38  # Set precision for Decimal calculations

# --- Colorama Initialization ---
init(autoreset=True)  # Initialize Colorama for cross-platform colored output

# --- Configuration Constants ---
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
DEFAULT_TIMEZONE = "America/Chicago"  # Default if TIMEZONE env var is not set

# --- API and Bot Behavior Constants ---
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
RETRY_ERROR_CODES = (429, 500, 502, 503, 504)  # Tuple for immutability
RETRY_HTTP_STATUS_CODES = (408, 429, 500, 502, 503, 504)  # For HTTP specific retries
LOOP_DELAY_SECONDS = 10
POSITION_CONFIRM_DELAY_SECONDS = 8

# --- Trading Constants ---
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "D": "1d",
    "W": "1w",
    "M": "1M",
}
FIB_LEVELS = [
    Decimal("0.0"),
    Decimal("0.236"),
    Decimal("0.382"),
    Decimal("0.5"),
    Decimal("0.618"),
    Decimal("0.786"),
    Decimal("1.0"),
]

# --- Indicator Default Periods ---
DEFAULT_INDICATOR_PERIODS = {
    "atr_period": 14,
    "cci_window": 20,
    "williams_r_window": 14,
    "mfi_window": 14,
    "stoch_rsi_window": 14,
    "stoch_rsi_rsi_window": 12,
    "stoch_rsi_k": 3,
    "stoch_rsi_d": 3,
    "rsi_period": 14,
    "bollinger_bands_period": 20,
    "bollinger_bands_std_dev": 2.0,
    "sma_10_window": 10,
    "ema_short_period": 9,
    "ema_long_period": 21,
    "momentum_period": 7,
    "volume_ma_period": 15,
    "fibonacci_window": 50,
    "psar_af": 0.02,
    "psar_max_af": 0.2,
}

# --- Colorama Color Constants ---
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.LIGHTBLUE_EX
NEON_PURPLE = Fore.LIGHTMAGENTA_EX
NEON_YELLOW = Fore.LIGHTYELLOW_EX
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.LIGHTCYAN_EX
RESET_ALL_STYLE = Style.RESET_ALL


def set_timezone(tz_str: str) -> None:
    """
    Sets the global timezone for the application.

    If the specified timezone string is invalid or 'zoneinfo' is unavailable
    for non-UTC timezones, it defaults to UTC.

    Args:
        tz_str: The timezone string (e.g., "America/Chicago", "UTC").
    """
    global TIMEZONE
    try:
        TIMEZONE = _ActualZoneInfo(tz_str)
        _module_logger.info(f"Timezone successfully set to: {str(TIMEZONE)}")
        # Warn if fallback is active (i.e., _ActualZoneInfo is FallbackZoneInfo)
        # and a non-UTC zone was requested, as FallbackZoneInfo will default to UTC.
        if _ActualZoneInfo is FallbackZoneInfo and tz_str.upper() != "UTC":
            _module_logger.warning(
                f"'zoneinfo' module is not available and timezone '{tz_str}' was requested. "
                f"Effective timezone is UTC due to FallbackZoneInfo limitations."
            )
    except Exception as tz_err:
        # This handles errors from _ActualZoneInfo(tz_str) constructor,
        # e.g., zoneinfo.ZoneInfoNotFoundError if tz_str is invalid.
        _module_logger.warning(
            f"Could not load timezone '{tz_str}'. Defaulting to UTC. Error: {tz_err}",
            exc_info=True,  # Log full traceback for the timezone loading error
        )
        # Ensure TIMEZONE is always set, defaulting to UTC
        if _ActualZoneInfo is not FallbackZoneInfo:  # _ActualZoneInfo is zoneinfo.ZoneInfo
            try:
                TIMEZONE = _ActualZoneInfo("UTC")  # Try real zoneinfo.ZoneInfo("UTC")
            except Exception as utc_load_err:  # Should be extremely rare
                _module_logger.error(
                    f"Critical: Could not load 'UTC' with available 'zoneinfo' module: {utc_load_err}. "
                    "Forcing FallbackZoneInfo for UTC.",
                    exc_info=True,
                )
                TIMEZONE = FallbackZoneInfo("UTC")  # Ultimate fallback
        else:  # zoneinfo not available, _ActualZoneInfo is FallbackZoneInfo
            # FallbackZoneInfo("UTC") is robust and should not fail.
            TIMEZONE = _ActualZoneInfo("UTC")  # This is FallbackZoneInfo("UTC")


def get_timezone() -> dt.tzinfo:
    """
    Retrieves the global timezone object.

    Initializes it from the TIMEZONE environment variable or `DEFAULT_TIMEZONE`
    if not already set.

    Returns:
        A datetime.tzinfo object representing the configured timezone.
    """
    global TIMEZONE
    if TIMEZONE is None:
        env_tz = os.getenv("TIMEZONE")
        chosen_tz_str: str

        if env_tz:
            _module_logger.info(f"TIMEZONE environment variable found: '{env_tz}'. Attempting to use it.")
            chosen_tz_str = env_tz
        else:
            _module_logger.info(f"TIMEZONE environment variable not set. Using default timezone: '{DEFAULT_TIMEZONE}'.")
            chosen_tz_str = DEFAULT_TIMEZONE

        set_timezone(chosen_tz_str)  # This will initialize TIMEZONE

    # At this point, TIMEZONE must be set by set_timezone.
    # An assert helps catch unexpected None states during development.
    assert TIMEZONE is not None, "TIMEZONE should have been initialized by set_timezone."
    return TIMEZONE


class SensitiveFormatter(logging.Formatter):
    """
    Custom logging formatter that redacts sensitive information (API key and secret)
    from log messages.
    """

    # Store sensitive data as class attributes.
    # This assumes one set of credentials for the application instance using this formatter.
    _api_key: Optional[str] = None
    _api_secret: Optional[str] = None

    @classmethod
    def set_sensitive_data(cls, api_key: Optional[str], api_secret: Optional[str]) -> None:
        """Sets the API key and secret to be redacted."""
        cls._api_key = api_key
        cls._api_secret = api_secret

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting sensitive data."""
        # Get the formatted message from the base class.
        msg = super().format(record)

        # Redact API key if it's set and present in the message.
        # Ensure _api_key is not None and not an empty string before replacing.
        if SensitiveFormatter._api_key:
            msg = msg.replace(SensitiveFormatter._api_key, "***API_KEY***")

        # Redact API secret if it's set and present in the message.
        # Ensure _api_secret is not None and not an empty string.
        if SensitiveFormatter._api_secret:
            msg = msg.replace(SensitiveFormatter._api_secret, "***API_SECRET***")
        return msg


def get_price_precision(market_info: Dict[str, Any], logger: logging.Logger) -> int:
    """
    Determines the number of decimal places for price formatting for a given market.

    It tries to extract this information from `market_info['precision']['price']`
    (interpreting int as places, float/str as tick size) or
    `market_info['limits']['price']['min']` as a fallback heuristic.

    Args:
        market_info: A dictionary containing market data from CCXT.
        logger: Logger instance for logging warnings or debug information.

    Returns:
        The determined price precision (number of decimal places). Defaults to 4.
    """
    symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")
    default_precision = 4  # Define default early for clarity

    # 1. Try 'precision'.'price'
    precision_data = market_info.get("precision", {})
    price_precision_value = precision_data.get("price")

    if price_precision_value is not None:
        if isinstance(price_precision_value, int):
            if price_precision_value >= 0:
                logger.debug(f"Using integer price precision for {symbol}: {price_precision_value}")
                return price_precision_value
            else:
                logger.warning(
                    f"Invalid negative integer price precision for {symbol}: {price_precision_value}. "
                    "Proceeding to next check."
                )

        elif isinstance(price_precision_value, (str, float)):
            try:
                # Assume this value represents the tick size
                tick_size = Decimal(str(price_precision_value))
                if tick_size > Decimal("0"):
                    # Precision is the number of decimal places in the tick size
                    # .normalize() removes trailing zeros, .as_tuple().exponent gives num decimal places (negative)
                    precision = abs(tick_size.normalize().as_tuple().exponent)
                    logger.debug(f"Derived price precision for {symbol} from tick size {tick_size}: {precision}")
                    return precision
                else:  # tick_size is zero or negative
                    logger.warning(
                        f"Invalid non-positive tick size '{price_precision_value}' from market_info.precision.price "
                        f"for {symbol}. Proceeding to next check."
                    )
            except (ValueError, TypeError, InvalidOperation) as e:
                logger.warning(
                    f"Could not parse market_info.precision.price '{price_precision_value}' "
                    f"as Decimal for {symbol}: {e}. Proceeding to next check."
                )
        else:
            logger.warning(
                f"Unsupported type for market_info.precision.price for {symbol}: {type(price_precision_value)}. "
                "Proceeding to next check."
            )

    # 2. Try 'limits'.'price'.'min' as a heuristic for tick size to derive precision
    price_limits = market_info.get("limits", {}).get("price", {})
    min_price_str = price_limits.get("min")

    if min_price_str is not None:
        try:
            min_price_tick = Decimal(str(min_price_str))
            if min_price_tick > Decimal("0"):
                # If min_price is positive, assume it can represent the tick size or a value
                # from which precision can be derived.
                precision = abs(min_price_tick.normalize().as_tuple().exponent)
                logger.debug(
                    f"Derived price precision for {symbol} from min_price_limit {min_price_tick} "
                    f"(parsed from '{min_price_str}'): {precision}"
                )
                return precision
            # else: min_price_tick is zero or negative, not useful for precision.
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.warning(
                f"Could not parse market_info.limits.price.min '{min_price_str}' "
                f"as Decimal for {symbol}: {e}. Falling back to default."
            )

    # 3. Default precision
    logger.warning(
        f"Could not determine price precision for {symbol} from market_info. Using default: {default_precision}."
    )
    return default_precision


def get_min_tick_size(market_info: Dict[str, Any], logger: logging.Logger) -> Decimal:
    """
    Determines the minimum tick size (price increment) for a given market.

    It tries to extract this from `market_info['precision']['price']`
    (interpreting float/str as tick size, int as precision for 10^-n) or
    `market_info['limits']['price']['min']`. Falls back to a value derived
    from `get_price_precision`.

    Args:
        market_info: A dictionary containing market data from CCXT.
        logger: Logger instance for logging warnings or debug information.

    Returns:
        The minimum tick size as a Decimal.
    """
    symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")

    # 1. Try 'precision'.'price'
    precision_data = market_info.get("precision", {})
    price_precision_value = precision_data.get("price")

    if price_precision_value is not None:
        if isinstance(price_precision_value, (str, float)):  # Typically, this is the tick size itself
            try:
                tick_size = Decimal(str(price_precision_value))
                if tick_size > Decimal("0"):
                    logger.debug(f"Using tick size from precision.price for {symbol}: {tick_size}")
                    return tick_size
                else:  # tick_size is zero or negative
                    logger.warning(
                        f"Invalid non-positive value '{price_precision_value}' from market_info.precision.price "
                        f"for {symbol}. Proceeding to next check."
                    )
            except (ValueError, TypeError, InvalidOperation) as e:
                logger.warning(
                    f"Could not parse market_info.precision.price '{price_precision_value}' "
                    f"as Decimal for {symbol}: {e}. Proceeding to next check."
                )

        elif isinstance(price_precision_value, int):  # Often number of decimal places
            if price_precision_value >= 0:
                # If 'precision'.'price' is an int, it's often number of decimal places for price
                tick_size = Decimal("1e-" + str(price_precision_value))
                logger.debug(f"Calculated tick size from integer precision for {symbol}: {tick_size}")
                return tick_size
            else:
                logger.warning(
                    f"Invalid negative integer for precision.price for {symbol}: {price_precision_value}. "
                    "Proceeding to next check."
                )
        else:
            logger.warning(
                f"Unsupported type for market_info.precision.price for {symbol}: {type(price_precision_value)}. "
                "Proceeding to next check."
            )

    # 2. Try 'limits'.'price'.'min'
    # This is sometimes the tick size, or at least related.
    price_limits = market_info.get("limits", {}).get("price", {})
    min_price_str = price_limits.get("min")

    if min_price_str is not None:
        try:
            min_tick_from_limit = Decimal(str(min_price_str))
            # If min_price from limits is positive, it might be the tick size.
            if min_tick_from_limit > Decimal("0"):
                logger.debug(
                    f"Using min_price from limits ('{min_price_str}') as tick size for {symbol}: {min_tick_from_limit}"
                )
                return min_tick_from_limit
            # else: min_tick_from_limit is zero or negative, not a valid tick size.
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.warning(
                f"Could not parse market_info.limits.price.min '{min_price_str}' "
                f"as Decimal for {symbol}: {e}. Falling back."
            )

    # 3. Fallback: derive tick size from calculated price precision
    # This re-uses get_price_precision, which has its own logging for defaults.
    price_prec_places = get_price_precision(market_info, logger)  # This call might log a warning if it defaults
    fallback_tick_size = Decimal("1e-" + str(price_prec_places))
    logger.warning(
        f"Could not determine specific min_tick_size for {symbol} from market_info. "
        f"Using fallback based on derived price precision ({price_prec_places} places): {fallback_tick_size}"
    )
    return fallback_tick_size


def _format_signal(signal_payload: Any, *, success: bool = True, detail: Optional[str] = None) -> str:
    """
    Formats a trading signal or related message for display, potentially with color.
    This is a placeholder to resolve the import error and can be customized
    based on the actual structure and requirements for signal formatting.

    Args:
        signal_payload: The content of the signal (e.g., "BUY", "SELL", a dictionary).
        success: If True, formats as a success/neutral message. If False, as an error/warning.
        detail: Optional additional detail string.

    Returns:
        A formatted string representation of the signal.
    """
    base_color = NEON_GREEN if success else NEON_RED
    prefix = "Signal" if success else "Signal Alert"

    signal_str = str(signal_payload)
    # Basic truncation for very long signal payloads to keep logs readable
    if len(signal_str) > 100:
        signal_str = signal_str[:97] + "..."

    message = f"{base_color}{prefix}: {signal_str}{RESET_ALL_STYLE}"
    if detail:
        message += f" ({base_color}{detail}{RESET_ALL_STYLE})"  # Color detail similarly

    return message


# --- End of utils.py ---
RESET = "[0m"

RESET = "\033[0m"
# file: main.py
"""
Main execution script for the XR Scalper Trading Bot.

This script initializes the bot, loads configuration, sets up logging,
connects to the exchange, and runs the main trading loop which iterates
through specified symbols, analyzes market data, and potentially places trades
based on the defined strategy.
"""

import logging
import sys
import time  # Retained for time.monotonic(), though asyncio.get_event_loop().time() is an alternative for async
import signal  # Import signal for graceful shutdown
import traceback  # Import traceback for detailed error reporting
import asyncio  # Import asyncio for asynchronous operations
from datetime import datetime
from typing import Dict, Any, Optional  # Added Optional for type hinting

import ccxt
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

# --- Import Custom Modules ---
try:
    import config_loader
    import exchange_api  # Assumed to contain async initialize_exchange
    import logger_setup  # logger_setup should provide configure_logging(config)
    import trading_strategy  # Assumed to contain async analyze_and_trade_symbol
    import utils  # utils should provide constants, timezone handling, sensitive data masking
except ImportError as e:
    print(
        "CRITICAL ERROR: Failed to import one or more custom modules. Ensure all required modules "
        "(config_loader.py, exchange_api.py, logger_setup.py, trading_strategy.py, utils.py) "
        "are in the correct directory and do not have syntax errors.",
        file=sys.stderr,
    )
    print(f"ImportError details: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: An unexpected error occurred during module import: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


# --- Constants ---
DEFAULT_LOOP_INTERVAL_SECONDS = 60  # Default loop interval if not in config

# --- Global State for Shutdown ---
shutdown_requested = False


def signal_handler(signum: int, frame: Any) -> None:
    """
    Signal handler to set the shutdown flag on receiving signals like SIGINT or SIGTERM.
    """
    global shutdown_requested
    shutdown_requested = True
    print(f"\nSignal {signum} received ({signal.Signals(signum).name}). Requesting bot shutdown...", file=sys.stderr)


# --- Main Execution Function ---
async def main() -> None:  # Changed to async def
    """
    Main function to initialize the bot, set up resources, and run the analysis/trading loop.
    Handles setup, configuration loading, exchange connection, and the main processing loop.
    """
    global API_KEY, API_SECRET  # Access globals if not passed as arguments

    CONFIG: Dict[str, Any] = {}
    try:
        CONFIG = config_loader.load_config()
        print(
            f"Configuration loaded successfully from '{utils.CONFIG_FILE}'.", file=sys.stderr
        )  # Basic print before logger
    except FileNotFoundError:
        print(f"CRITICAL: Configuration file '{utils.CONFIG_FILE}' not found. Exiting.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL: Error loading configuration: {e}. Exiting.", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    try:
        logger_setup.configure_logging(CONFIG)
    except Exception as e:
        print(f"CRITICAL: Error configuring logging: {e}. Exiting.", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    init_logger = logging.getLogger("xrscalper_bot_init")
    init_logger.info("--- Initializing XR Scalper Bot (Async Version) ---")

    configured_timezone = os.getenv("TIMEZONE", CONFIG.get("timezone", utils.DEFAULT_TIMEZONE))
    try:
        utils.set_timezone(configured_timezone)
        init_logger.info(f"Using Timezone: {utils.get_timezone()}")
    except Exception as e:
        init_logger.warning(
            f"Failed to set timezone to '{configured_timezone}': {e}. Using system default.", exc_info=True
        )

    try:
        if hasattr(utils, "SensitiveFormatter") and hasattr(utils.SensitiveFormatter, "set_sensitive_data"):
            if API_KEY and API_SECRET:
                utils.SensitiveFormatter.set_sensitive_data(API_KEY, API_SECRET)
                init_logger.debug("Sensitive data masking configured for logging.")
            else:
                init_logger.warning(
                    ".env API keys not loaded/available globally, sensitive data masking may not function correctly."
                )
        else:
            init_logger.warning("Sensitive data masking features not found in utils module.")
    except Exception as e:
        init_logger.warning(f"Error configuring sensitive data masking: {e}", exc_info=True)

    try:
        current_time_str = datetime.now(utils.get_timezone()).strftime("%Y-%m-%d %H:%M:%S %Z")
        init_logger.info(f"Startup Time: {current_time_str}")
    except Exception as e:
        init_logger.warning(f"Could not format startup time with timezone: {e}. Falling back.", exc_info=True)
        init_logger.info(f"Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    quote_currency = CONFIG.get("quote_currency")
    if quote_currency:
        init_logger.info(f"Quote Currency: {quote_currency}")
    else:
        init_logger.warning("Quote currency not specified in configuration.")

    try:
        python_version = sys.version.split()[0]
        ccxt_version = getattr(ccxt, "__version__", "N/A")
        pandas_version = getattr(pd, "__version__", "N/A")
        pandas_ta_version = getattr(ta, "__version__", "N/A")
        if hasattr(ta, "version") and callable(ta.version) and pandas_ta_version == "N/A":  # More specific check
            try:
                pandas_ta_version = ta.version()
            except Exception:
                pandas_ta_version = "N/A (ta.version() error)"
    except Exception as e:
        init_logger.warning(f"Error getting dependency versions: {e}", exc_info=True)
        python_version = sys.version.split()[0]
        ccxt_version = pandas_version = pandas_ta_version = "N/A"

    init_logger.info(
        f"Versions: Python={python_version}, CCXT={ccxt_version}, Pandas={pandas_version}, PandasTA={pandas_ta_version}"
    )

    enable_trading = CONFIG.get("enable_trading", False)
    use_sandbox = CONFIG.get("use_sandbox", True)

    if enable_trading:
        init_logger.warning("!!! LIVE TRADING IS ENABLED !!!")
        if use_sandbox:
            init_logger.warning("--> Operating in SANDBOX (Testnet) Environment.")
        else:
            init_logger.warning("!!! CAUTION: OPERATING WITH REAL MONEY !!!")

        risk_per_trade_config = CONFIG.get("risk_per_trade", 0.0)
        risk_pct = risk_per_trade_config * 100
        leverage = CONFIG.get("leverage", 1)
        symbols_to_trade = CONFIG.get("symbols_to_trade", [])
        tsl_enabled = CONFIG.get("enable_trailing_stop", False)
        be_enabled = CONFIG.get("enable_break_even", False)
        entry_order_type = CONFIG.get("entry_order_type", "market").lower()
        interval = CONFIG.get("interval", "N/A")  # Ensure interval is a string for CCXT_INTERVAL_MAP

        init_logger.info("--- Critical Trading Settings ---")
        init_logger.info(f"  Symbols: {symbols_to_trade}")
        init_logger.info(f"  Interval: {str(interval)}")  # Ensure interval is logged as string
        init_logger.info(f"  Entry Order Type: {entry_order_type}")
        init_logger.info(f"  Risk per Trade: {risk_pct:.2f}% ({risk_per_trade_config})")
        init_logger.info(f"  Leverage: {leverage}x")
        init_logger.info(f"  Trailing Stop Loss Enabled: {tsl_enabled}")
        init_logger.info(f"  Break Even Enabled: {be_enabled}")
        init_logger.info("---------------------------------")

        if not symbols_to_trade:
            init_logger.error("Trading enabled, but 'symbols_to_trade' list is empty. Exiting.")
            return
        if risk_per_trade_config <= 0:
            init_logger.warning(
                f"Risk per trade is set to {risk_pct:.2f}%. Positions might not open unless strategy ignores risk."
            )
        if leverage <= 0:
            init_logger.error(f"Leverage must be greater than 0. Found {leverage}. Exiting.")
            return
        if entry_order_type not in ["market", "limit"]:
            init_logger.error(
                f"Invalid 'entry_order_type': '{entry_order_type}'. Must be 'market' or 'limit'. Exiting."
            )
            return
        # Ensure interval is a string before checking in CCXT_INTERVAL_MAP keys
        if not interval or str(interval) not in utils.CCXT_INTERVAL_MAP:
            init_logger.error(f"Invalid or missing 'interval': '{interval}'. Cannot map to CCXT timeframe. Exiting.")
            return

        init_logger.info("Review settings. Starting trading loop in 5 seconds...")
        await asyncio.sleep(5)  # Use asyncio.sleep
    else:
        init_logger.info("Live trading is DISABLED. Running in analysis-only mode.")
        symbols_to_process_analysis = CONFIG.get("symbols_to_trade", [])
        if not symbols_to_process_analysis:
            init_logger.error("'symbols_to_trade' is empty. Nothing to process in analysis mode. Exiting.")
            return
        init_logger.info(f"Analysis symbols: {symbols_to_process_analysis}")

    init_logger.info("Initializing exchange connection...")
    # Ensure API_KEY and API_SECRET are available (loaded globally)
    exchange: Optional[ccxt.Exchange] = None  # Type hint for clarity
    try:
        # Assuming initialize_exchange is an async function
        exchange = await exchange_api.initialize_exchange(API_KEY, API_SECRET, CONFIG, init_logger)
    except Exception as e:
        init_logger.error(f"Unhandled exception during exchange_api.initialize_exchange: {e}", exc_info=True)
        init_logger.error("Bot cannot continue. Exiting.")
        return

    if not exchange:
        init_logger.error(
            "Failed to initialize exchange connection (exchange object is None). Bot cannot continue. Exiting."
        )
        return

    symbols_to_process = CONFIG.get("symbols_to_trade", [])

    try:
        init_logger.info(f"Loading exchange markets for {exchange.id}...")  # This should now work
        await exchange.load_markets()  # Use await for async ccxt method
        init_logger.info(f"Exchange '{exchange.id}' initialized and markets loaded successfully.")

        if enable_trading:
            if use_sandbox:
                init_logger.info("Connected to exchange in SANDBOX mode.")
            else:
                init_logger.info("Connected to exchange in LIVE (Real Money) mode.")
        else:
            init_logger.info(f"Connected to exchange (mode: {'sandbox' if use_sandbox else 'live'}) for analysis.")

        available_symbols = exchange.symbols if hasattr(exchange, "symbols") and exchange.symbols else []
        if not available_symbols:
            init_logger.error("Could not retrieve available symbols from exchange. Cannot validate config. Exiting.")
            return

        invalid_symbols = [s for s in symbols_to_process if s not in available_symbols]
        if invalid_symbols:
            init_logger.error(f"Invalid symbols in config (not on exchange): {invalid_symbols}. Exiting.")
            return
        init_logger.info("All configured symbols validated against exchange markets.")

    except ccxt.NetworkError as ne:
        init_logger.error(
            f"Network Error during exchange setup: {ne}. Check connection/status. Exiting.", exc_info=True
        )
        return
    except ccxt.ExchangeError as ee:
        init_logger.error(
            f"Exchange Error during exchange setup: {ee}. Check API keys/permissions. Exiting.", exc_info=True
        )
        return
    except AttributeError as ae:  # Specifically catch if exchange object is not as expected after await
        init_logger.error(
            f"AttributeError during exchange setup (likely API issue or response format): {ae}. Exiting.", exc_info=True
        )
        return
    except Exception as e:
        init_logger.error(
            f"An unexpected error occurred during exchange initialization/market loading: {e}. Exiting.", exc_info=True
        )
        return

    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        init_logger.info("Signal handlers registered for graceful shutdown.")
    except Exception as e:  # e.g. on Windows if SIGTERM is not available
        init_logger.warning(
            f"Failed to register signal handlers: {e}. Graceful shutdown via signals might be affected.", exc_info=True
        )

    loop_interval_seconds_cfg = CONFIG.get("loop_interval_seconds", DEFAULT_LOOP_INTERVAL_SECONDS)
    if not isinstance(loop_interval_seconds_cfg, (int, float)) or loop_interval_seconds_cfg <= 0:
        init_logger.error(f"Invalid 'loop_interval_seconds': {loop_interval_seconds_cfg}. Must be positive. Exiting.")
        return
    loop_interval_seconds = float(loop_interval_seconds_cfg)

    init_logger.info(f"Starting main analysis/trading loop for symbols: {symbols_to_process}")
    init_logger.info(f"Loop Interval: {loop_interval_seconds} seconds")

    try:
        while not shutdown_requested:
            loop_start_time = time.monotonic()  # time.monotonic() is fine for measuring intervals
            current_cycle_time = datetime.now(utils.get_timezone())
            init_logger.info(f"--- Starting Loop Cycle @ {current_cycle_time.strftime('%Y-%m-%d %H:%M:%S %Z')} ---")

            for symbol in symbols_to_process:
                if shutdown_requested:
                    init_logger.info("Shutdown requested, stopping symbol processing in current cycle.")
                    break

                safe_symbol_name = symbol.replace("/", "_").replace(":", "-")
                symbol_logger = logging.getLogger(f"xrscalper_bot_{safe_symbol_name}")

                try:
                    symbol_logger.info(f"Processing symbol: {symbol}")
                    # Assuming analyze_and_trade_symbol is an async function
                    await trading_strategy.analyze_and_trade_symbol(
                        exchange, symbol, CONFIG, symbol_logger, enable_trading
                    )
                except ccxt.NetworkError as ne:
                    symbol_logger.error(f"Network Error for {symbol}: {ne}. Retrying next cycle.", exc_info=True)
                except ccxt.ExchangeError as ee:
                    symbol_logger.error(f"Exchange Error for {symbol}: {ee}. Retrying next cycle.", exc_info=True)
                except Exception as symbol_err:
                    symbol_logger.error(
                        f"!!! Unhandled Exception during analysis for {symbol}: {symbol_err} !!!", exc_info=True
                    )
                    symbol_logger.warning("Attempting to continue to next symbol/cycle.")
                finally:
                    # Optional: await asyncio.sleep(CONFIG.get("delay_between_symbols", 0.5))
                    symbol_logger.info(f"Finished processing symbol: {symbol}\n")

            if shutdown_requested:
                init_logger.info("Shutdown requested, exiting main loop after current cycle.")
                break

            loop_end_time = time.monotonic()
            elapsed_time = loop_end_time - loop_start_time
            sleep_duration = max(0.1, loop_interval_seconds - elapsed_time)

            init_logger.debug(
                f"Loop cycle finished. Elapsed: {elapsed_time:.2f}s. Sleeping for: {sleep_duration:.2f}s."
            )
            if sleep_duration > 0:  # Always sleep if > 0, asyncio.sleep(0) yields control
                await asyncio.sleep(sleep_duration)  # Use asyncio.sleep
            if elapsed_time > loop_interval_seconds:
                init_logger.warning(
                    f"Loop cycle duration ({elapsed_time:.2f}s) exceeded target interval ({loop_interval_seconds}s)."
                )

    except Exception as loop_err:
        init_logger.critical(f"!!! CRITICAL UNHANDLED EXCEPTION IN MAIN ASYNC LOOP: {loop_err} !!!", exc_info=True)
        init_logger.critical("The bot encountered a critical error and will exit.")
    finally:
        init_logger.info("--- XR Scalper Bot Shutting Down ---")
        # Add any async cleanup specific to the exchange if needed, e.g., await exchange.close()
        if exchange and hasattr(exchange, "close") and callable(exchange.close):
            try:
                init_logger.info(f"Closing exchange connection for {exchange.id}...")
                await exchange.close()  # Important for async ccxt exchanges
                init_logger.info("Exchange connection closed.")
            except Exception as ex_close_err:
                init_logger.error(f"Error closing exchange connection: {ex_close_err}", exc_info=True)

        logging.shutdown()


# --- Script Entry Point ---
# These globals are set before main() is called.
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        print("CRITICAL ERROR: BYBIT_API_KEY and BYBIT_API_SECRET must be set in the .env file.", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This handles Ctrl+C if it bypasses the signal handler or occurs during asyncio.run setup/teardown
        print("\nBot execution interrupted by user (KeyboardInterrupt). Finalizing shutdown.", file=sys.stderr)
        # Ensure shutdown_requested is set if not already, for any lingering tasks (though asyncio.run will exit)
        shutdown_requested = True
    except Exception as e:
        print(f"CRITICAL UNHANDLED ERROR at script execution entry point: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)  # Ensure exit on critical error
    finally:
        # This print will occur after main() has completed or been interrupted.
        # logging.shutdown() should have been called within main's finally block.
        print("XR Scalper Bot script execution has concluded.", file=sys.stderr)
# File: exchange_api.py
"""Module for interacting with cryptocurrency exchanges via CCXT with enhanced error handling, retries, and Bybit V5 API support."""

import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Union
import asyncio  # For asyncio.sleep
import importlib.metadata  # For getting package version

# Use async support version of ccxt
import ccxt.async_support as ccxt_async  # Renamed to avoid conflict with standard ccxt if used elsewhere

# Import constants and utility functions
from utils import (
    MAX_API_RETRIES,
    NEON_GREEN,
    NEON_RED,
    NEON_YELLOW,
    RESET,  # For log coloring
    RETRY_DELAY_SECONDS,
    get_min_tick_size,
    get_price_precision,
    # DEFAULT_INDICATOR_PERIODS, # Removed as it's not used in this file
)

module_logger = logging.getLogger(__name__)
_market_info_cache: Dict[str, Dict[str, Any]] = {}


def _exponential_backoff(attempt: int, base_delay: float = RETRY_DELAY_SECONDS, max_cap: float = 60.0) -> float:
    """Calculate delay for exponential backoff with a cap."""
    delay = base_delay * (2**attempt)
    return min(delay, max_cap)


async def _handle_fetch_exception(
    e: Exception, logger: logging.Logger, attempt: int, total_attempts: int, item_desc: str, context_info: str
) -> bool:
    """Helper to log and determine if a fetch exception is retryable for async functions."""
    is_retryable = False
    current_retry_delay = RETRY_DELAY_SECONDS  # Default delay
    error_detail = str(e)

    if isinstance(e, (ccxt_async.NetworkError, ccxt_async.RequestTimeout, asyncio.TimeoutError)):
        log_level_method = logger.warning
        is_retryable = True
        msg = f"Network/Timeout error fetching {item_desc} for {context_info}"
    elif isinstance(e, (ccxt_async.RateLimitExceeded, ccxt_async.DDoSProtection)):
        log_level_method = logger.warning
        is_retryable = True
        msg = f"Rate limit/DDoS triggered fetching {item_desc} for {context_info}"
        # Use longer, exponential backoff for rate limits
        current_retry_delay = _exponential_backoff(attempt, base_delay=RETRY_DELAY_SECONDS * 3, max_cap=180.0)
    elif isinstance(e, ccxt_async.ExchangeError):
        log_level_method = logger.error  # Default to error for ExchangeError
        is_retryable = False
        err_str_lower = error_detail.lower()
        # Phrases that usually indicate a non-retryable client-side or setup error for fetch operations
        non_retryable_phrases = [
            "symbol",
            "market",
            "not found",
            "invalid",
            "parameter",
            "argument",
            "orderid",
            "insufficient",
            "balance",
            "margin account not exist",
        ]
        # Bybit specific error codes that are often non-retryable for fetch operations
        non_retryable_codes = [
            10001,  # params error
            110025,  # position not found / not exist
            110009,  # margin account not exist
            110045,  # unified account not exist
        ]
        if (
            any(phrase in err_str_lower for phrase in non_retryable_phrases)
            or getattr(e, "code", None) in non_retryable_codes
        ):
            msg = f"Exchange error (likely non-retryable) fetching {item_desc} for {context_info}"
        else:
            # Some exchange errors might be temporary (e.g., temporary trading ban, internal server error)
            msg = f"Potentially temporary Exchange error fetching {item_desc} for {context_info}"
            log_level_method = logger.warning  # Downgrade to warning for retry
            is_retryable = True
    else:
        log_level_method = logger.error
        is_retryable = False
        msg = f"Unexpected error fetching {item_desc} for {context_info}"
        # Log with exc_info for unexpected errors to get traceback
        log_level_method(f"{NEON_RED}{msg}: {error_detail}{RESET}", exc_info=True)
        return False  # No retry for truly unexpected errors handled here

    log_level_method(
        f"{NEON_YELLOW if is_retryable else NEON_RED}{msg}: {error_detail} (Attempt {attempt + 1}/{total_attempts}){RESET}"
    )

    if is_retryable and attempt < total_attempts - 1:
        logger.warning(f"Waiting {current_retry_delay:.2f}s before retrying {item_desc} fetch for {context_info}...")
        await asyncio.sleep(current_retry_delay)
    return is_retryable


async def initialize_exchange(
    api_key: str, api_secret: str, config: Dict[str, Any], logger: logging.Logger
) -> Optional[ccxt_async.Exchange]:
    exchange: Optional[ccxt_async.Exchange] = None
    try:
        try:
            ccxt_version = importlib.metadata.version("ccxt")
            logger.info(f"Using CCXT version: {ccxt_version}")
        except importlib.metadata.PackageNotFoundError:
            logger.warning("Could not determine CCXT version. Ensure 'ccxt' is installed.")

        exchange_id = config.get("exchange_id", "bybit").lower()
        if not hasattr(ccxt_async, exchange_id):
            logger.error(f"{NEON_RED}Exchange ID '{exchange_id}' not found in CCXT async library.{RESET}")
            return None

        exchange_class = getattr(ccxt_async, exchange_id)
        exchange_options = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,  # CCXT's built-in rate limiter
            "options": {
                "defaultType": config.get(
                    "default_market_type", "linear"
                ),  # e.g., 'linear', 'inverse', 'spot', 'unified'
                "adjustForTimeDifference": True,  # Auto-sync time with server
                # Timeouts for various operations (in milliseconds)
                "fetchTickerTimeout": config.get("ccxt_fetch_ticker_timeout_ms", 15000),
                "fetchBalanceTimeout": config.get("ccxt_fetch_balance_timeout_ms", 20000),
                "createOrderTimeout": config.get("ccxt_create_order_timeout_ms", 25000),
                "cancelOrderTimeout": config.get("ccxt_cancel_order_timeout_ms", 20000),
                "fetchPositionsTimeout": config.get("ccxt_fetch_positions_timeout_ms", 20000),
                "fetchOHLCVTimeout": config.get("ccxt_fetch_ohlcv_timeout_ms", 20000),
                "loadMarketsTimeout": config.get("ccxt_load_markets_timeout_ms", 30000),
            },
        }
        if exchange_id == "bybit":
            # Bybit: Market orders do not require a price parameter
            exchange_options["options"]["createOrderRequiresPrice"] = False

        exchange = exchange_class(exchange_options)

        if config.get("use_sandbox"):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            if hasattr(exchange, "set_sandbox_mode") and callable(exchange.set_sandbox_mode):
                try:
                    # Some exchanges have a method to switch to sandbox
                    await exchange.set_sandbox_mode(True)  # CCXT standard way
                    logger.info(f"Sandbox mode enabled for {exchange.id} via set_sandbox_mode(True).")
                except Exception as sandbox_err:
                    logger.warning(
                        f"Error calling set_sandbox_mode(True) for {exchange.id}: {sandbox_err}. "
                        f"Attempting manual URL override if known."
                    )
                    # Fallback for Bybit if set_sandbox_mode is problematic or not available
                    if exchange.id == "bybit":
                        testnet_url = exchange.urls.get("test", "https://api-testnet.bybit.com")
                        exchange.urls["api"] = testnet_url
                        logger.info(f"Manual Bybit testnet URL set: {testnet_url}")
            elif exchange.id == "bybit":  # Direct manual override for Bybit
                testnet_url = exchange.urls.get("test", "https://api-testnet.bybit.com")
                exchange.urls["api"] = testnet_url
                logger.info(f"Manual Bybit testnet URL override applied: {testnet_url}")
            else:
                logger.warning(
                    f"{NEON_YELLOW}{exchange.id} doesn't support set_sandbox_mode or known manual override. "
                    f"Ensure API keys are Testnet keys if using sandbox.{RESET}"
                )

        logger.info(f"Loading markets for {exchange.id}...")
        await exchange.load_markets(reload=True)  # reload=True ensures fresh market data
        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {config.get('use_sandbox', False)}")

        quote_currency = config.get("quote_currency", "USDT")
        default_market_type = exchange.options.get("defaultType", "N/A")
        logger.info(f"Attempting initial balance fetch for {quote_currency} (context: {default_market_type})...")
        balance_decimal = await fetch_balance(exchange, quote_currency, logger)
        if balance_decimal is not None:
            logger.info(
                f"{NEON_GREEN}Initial balance fetch successful for {quote_currency}: {balance_decimal:.4f}{RESET}"
            )
        else:
            logger.error(
                f"{NEON_RED}Initial balance fetch FAILED for {quote_currency}. "
                f"Bot might not function correctly if balance is critical.{RESET}"
            )
            # Consider closing exchange and returning None if initial balance is absolutely critical
            # await exchange.close()
            # return None
        return exchange
    except Exception as e:
        logger.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        if exchange:
            await exchange.close()
    return None


async def fetch_current_price_ccxt(
    exchange: ccxt_async.Exchange, symbol: str, logger: logging.Logger
) -> Optional[Decimal]:
    attempts = 0
    total_attempts = MAX_API_RETRIES + 1
    while attempts < total_attempts:
        try:
            logger.debug(f"Fetching ticker for {symbol}... (Attempt {attempts + 1}/{total_attempts})")
            ticker = await exchange.fetch_ticker(symbol)

            # Prioritize different price fields from the ticker
            price_sources = []
            if ticker.get("bid") is not None and ticker.get("ask") is not None:
                try:
                    bid = Decimal(str(ticker["bid"]))
                    ask = Decimal(str(ticker["ask"]))
                    if bid > 0 and ask > 0 and ask >= bid:
                        price_sources.append((bid + ask) / Decimal("2"))  # Mid-price
                except (InvalidOperation, TypeError):
                    logger.debug(f"Could not parse bid/ask for mid-price for {symbol}.", exc_info=True)

            # Order of preference: last, close, ask, bid (after mid-price)
            for key in ["last", "close", "ask", "bid"]:
                if ticker.get(key) is not None:
                    price_sources.append(ticker[key])

            for price_val in price_sources:
                if price_val is not None:
                    try:
                        price_dec = Decimal(str(price_val))
                        if price_dec > 0:
                            logger.debug(f"Price for {symbol} obtained: {price_dec}")
                            return price_dec
                    except (InvalidOperation, TypeError):
                        continue  # Try next price source

            logger.warning(
                f"No valid price (last, close, bid, ask, mid) found in ticker for {symbol} on attempt {attempts + 1}. Ticker: {ticker}"
            )
            # Raise an error to trigger retry mechanism if ticker was fetched but no price found
            raise ccxt_async.ExchangeError("No valid price found in ticker data.")
        except Exception as e:
            if not await _handle_fetch_exception(e, logger, attempts, total_attempts, f"price for {symbol}", symbol):
                return None  # Non-retryable error or max retries exceeded
        attempts += 1
    logger.error(f"Failed to fetch price for {symbol} after {total_attempts} attempts.")
    return None


async def fetch_klines_ccxt(
    exchange: ccxt_async.Exchange,
    symbol: str,
    timeframe: str,
    limit: int = 250,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    current_logger = logger or module_logger
    if not exchange.has["fetchOHLCV"]:
        current_logger.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return pd.DataFrame()

    total_attempts = MAX_API_RETRIES + 1
    for attempt in range(total_attempts):
        try:
            current_logger.debug(
                f"Fetching klines for {symbol} (Timeframe: {timeframe}, Limit: {limit}) (Attempt {attempt + 1}/{total_attempts})"
            )
            ohlcv_data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            if (
                ohlcv_data
                and isinstance(ohlcv_data, list)
                and len(ohlcv_data) > 0
                and all(isinstance(row, list) and len(row) >= 6 for row in ohlcv_data)
            ):  # Check structure
                df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

                # Convert timestamp to datetime, normalize to UTC, then remove tz for consistency
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=True)
                df.dropna(subset=["timestamp"], inplace=True)  # Drop rows where timestamp conversion failed
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
                df.set_index("timestamp", inplace=True)

                # Convert OHLCV columns to Decimal, handling potential errors
                for col in ["open", "high", "low", "close", "volume"]:
                    try:
                        # Ensure data is string before Decimal conversion; handle empty strings
                        df[col] = (
                            df[col]
                            .astype(object)
                            .apply(lambda x: Decimal(str(x)) if pd.notna(x) and str(x).strip() != "" else None)
                        )
                    except (InvalidOperation, TypeError) as conv_err:
                        current_logger.warning(
                            f"Could not convert column '{col}' to Decimal for {symbol} due to: {conv_err}. "
                            f"Falling back to pd.to_numeric, data might lose precision or be invalid."
                        )
                        df[col] = pd.to_numeric(df[col], errors="coerce")  # Coerce errors to NaN

                # Drop rows with NaN in critical OHLC columns after conversion attempts
                df.dropna(subset=["open", "high", "low", "close"], how="any", inplace=True)

                # Filter out candles with non-positive close price or negative volume
                df = df[
                    df["close"].apply(lambda x: isinstance(x, Decimal) and x > Decimal(0) or (pd.notna(x) and x > 0))
                ].copy()
                df = df[
                    df["volume"].apply(lambda x: isinstance(x, Decimal) and x >= Decimal(0) or (pd.notna(x) and x >= 0))
                ].copy()

                if df.empty:
                    current_logger.warning(
                        f"Klines data for {symbol} {timeframe} is empty after cleaning and validation."
                    )
                    # Consider this a fetch failure to allow retry if appropriate
                    raise ccxt_async.ExchangeError("Cleaned kline data is empty.")

                df.sort_index(inplace=True)  # Ensure chronological order
                current_logger.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}.")
                return df
            else:
                current_logger.warning(
                    f"Received empty or invalid kline data structure for {symbol} {timeframe}. "
                    f"Data: {ohlcv_data[:2] if ohlcv_data else 'None'}... (Attempt {attempt + 1})"
                )
                raise ccxt_async.ExchangeError("Empty or invalid kline data structure from exchange.")
        except Exception as e:
            if not await _handle_fetch_exception(
                e, current_logger, attempt, total_attempts, f"klines for {symbol} {timeframe}", symbol
            ):
                return pd.DataFrame()  # Non-retryable or max retries hit

    current_logger.error(f"Failed to fetch klines for {symbol} {timeframe} after {total_attempts} attempts.")
    return pd.DataFrame()


async def fetch_orderbook_ccxt(
    exchange: ccxt_async.Exchange, symbol: str, limit: int, logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    if not exchange.has["fetchOrderBook"]:
        logger.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
        return None

    attempts = 0
    total_attempts = MAX_API_RETRIES + 1
    while attempts < total_attempts:
        try:
            logger.debug(f"Fetching order book for {symbol} (Limit: {limit}) (Attempt {attempts + 1}/{total_attempts})")
            order_book = await exchange.fetch_order_book(symbol, limit=limit)

            if (
                order_book
                and isinstance(order_book, dict)
                and "bids" in order_book
                and isinstance(order_book["bids"], list)
                and "asks" in order_book
                and isinstance(order_book["asks"], list)
            ):
                if not order_book["bids"] and not order_book["asks"]:
                    logger.warning(f"Order book for {symbol} fetched but bids and asks arrays are empty.")
                # Basic structure is valid, return it
                return order_book
            else:
                logger.warning(
                    f"Invalid order book structure received for {symbol} on attempt {attempts + 1}. "
                    f"Data: {str(order_book)[:200]}..."  # Log snippet of problematic data
                )
                raise ccxt_async.ExchangeError("Invalid order book structure received.")
        except Exception as e:
            if not await _handle_fetch_exception(
                e, logger, attempts, total_attempts, f"orderbook for {symbol}", symbol
            ):
                return None
        attempts += 1
    logger.error(f"Failed to fetch order book for {symbol} after {total_attempts} attempts.")
    return None


async def fetch_balance(
    exchange: ccxt_async.Exchange, currency: str, logger: logging.Logger, params: Optional[Dict] = None
) -> Optional[Decimal]:
    request_params = params.copy() if params is not None else {}  # Use a copy to avoid modifying caller's dict

    # Specific handling for Bybit account types if not provided in params
    if exchange.id == "bybit" and "accountType" not in request_params:
        default_type = exchange.options.get("defaultType", "").upper()
        if default_type == "UNIFIED":
            request_params["accountType"] = "UNIFIED"
        elif default_type in ["LINEAR", "INVERSE", "CONTRACT"]:  # CONTRACT covers linear/inverse for older API versions
            request_params["accountType"] = "CONTRACT"
        elif default_type == "SPOT":
            request_params["accountType"] = "SPOT"
        # If defaultType is not set or recognized, Bybit might use a default or require it.

    attempts = 0
    total_attempts = MAX_API_RETRIES + 1
    while attempts < total_attempts:
        try:
            logger.debug(
                f"Fetching balance for {currency} (Attempt {attempts + 1}/{total_attempts}). "
                f"Params: {request_params if request_params else 'None'}"
            )
            balance_info = await exchange.fetch_balance(params=request_params)

            if balance_info:
                # Try to get currency-specific balance data first
                currency_data = balance_info.get(currency.upper())  # Ensure currency code is uppercase
                available_balance_str = None

                if currency_data and currency_data.get("free") is not None:
                    available_balance_str = str(currency_data["free"])
                elif currency_data and currency_data.get("total") is not None:
                    # Use 'total' if 'free' is not available, but log a warning
                    available_balance_str = str(currency_data["total"])
                    logger.warning(
                        f"Using 'total' balance for {currency} as 'free' is unavailable. "
                        f"This might include locked funds."
                    )
                # Fallback for structures where 'free' is a top-level dict
                elif (
                    "free" in balance_info
                    and isinstance(balance_info["free"], dict)
                    and balance_info["free"].get(currency.upper()) is not None
                ):
                    available_balance_str = str(balance_info["free"][currency.upper()])

                if available_balance_str is not None:
                    try:
                        final_balance = Decimal(available_balance_str)
                        if final_balance >= Decimal(0):
                            logger.info(f"Available {currency} balance: {final_balance:.8f}")
                            return final_balance
                        else:
                            logger.error(
                                f"Parsed balance for {currency} is negative ({final_balance}). This is unusual."
                            )
                            # Depending on policy, might treat negative as error or 0. Here, it's an error state.
                    except InvalidOperation:
                        logger.error(
                            f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}."
                        )
                else:
                    logger.error(
                        f"Could not determine free balance for {currency}. "
                        f"Relevant balance keys: {list(balance_info.keys() if isinstance(balance_info, dict) else [])}. "
                        f"Currency data: {currency_data}"
                    )
            else:
                logger.error(f"Balance info response was None or empty on attempt {attempts + 1}.")

            # If balance parsing failed or info was empty, raise to retry
            raise ccxt_async.ExchangeError(f"Balance parsing or fetch failed for {currency}.")
        except Exception as e:
            if not await _handle_fetch_exception(
                e, logger, attempts, total_attempts, f"balance for {currency}", currency
            ):
                return None
        attempts += 1
    logger.error(f"Failed to fetch balance for {currency} after {total_attempts} attempts.")
    return None


async def get_market_info(
    exchange: ccxt_async.Exchange, symbol: str, logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    cache_key = f"{exchange.id}:{symbol}"
    if cache_key in _market_info_cache:
        logger.debug(f"Using cached market info for {symbol}.")
        return _market_info_cache[cache_key]

    try:
        # Ensure markets are loaded; reload if symbol not found initially
        if not exchange.markets or symbol not in exchange.markets:
            logger.info(f"Market info for {symbol} not found or markets not loaded. Reloading markets...")
            await exchange.load_markets(reload=True)

        if symbol not in exchange.markets:
            logger.error(f"Market {symbol} still not found after reloading markets.")
            return None

        market = exchange.market(symbol)
        if not market:  # Should not happen if symbol is in exchange.markets, but defensive check
            logger.error(f"exchange.market({symbol}) returned None despite symbol being in markets list.")
            return None

        # Ensure essential precision and limits keys exist with sane defaults if missing
        market.setdefault("precision", {})
        market["precision"].setdefault("price", "1e-8")  # Default price precision (e.g., 8 decimal places)
        market["precision"].setdefault("amount", "1e-8")  # Default amount precision

        market.setdefault("limits", {})
        market["limits"].setdefault("amount", {}).setdefault("min", "0")  # Default min amount
        market["limits"].setdefault("cost", {}).setdefault("min", "0")  # Default min cost

        # Determine if the market is a contract (future/swap)
        market["is_contract"] = market.get("contract", False) or market.get("type", "unknown").lower() in [
            "swap",
            "future",
            "option",
            "linear",
            "inverse",
        ]

        # Calculate 'amountPrecision' (number of decimal places for amount) if not present
        # This is often derived from market['precision']['amount'] (step size)
        if "amountPrecision" not in market or not isinstance(market.get("amountPrecision"), int):
            amount_step_val = market["precision"].get("amount")
            derived_precision = 8  # Default fallback
            if isinstance(amount_step_val, (int)) and amount_step_val >= 0:
                derived_precision = amount_step_val  # If it's already an integer precision
            elif isinstance(amount_step_val, (float, str, Decimal)):
                try:
                    step = Decimal(str(amount_step_val))
                    if step > 0:
                        # Calculate decimal places from step (e.g., 0.001 -> 3)
                        derived_precision = abs(step.normalize().as_tuple().exponent)
                except (InvalidOperation, TypeError):
                    logger.warning(
                        f"Could not derive amountPrecision from step '{amount_step_val}' for {symbol}. Using default."
                    )
            market["amountPrecision"] = derived_precision

        logger.debug(
            f"Market Info for {symbol}: Type={market.get('type')}, Contract={market['is_contract']}, "
            f"TickSize(PriceStep)={market['precision']['price']}, AmountStep={market['precision']['amount']}, "
            f"AmountPrecision(DecimalPlaces)={market['amountPrecision']}"
        )
        _market_info_cache[cache_key] = market
        return market
    except Exception as e:
        logger.error(f"Error getting or processing market info for {symbol}: {e}", exc_info=True)
        return None


async def get_open_position(
    exchange: ccxt_async.Exchange, symbol: str, market_info: Dict[str, Any], logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    if not exchange.has.get("fetchPositions"):  # .get is safer for dicts like 'has'
        logger.warning(f"Exchange {exchange.id} does not support fetchPositions.")
        return None

    market_id = market_info.get("id")  # Exchange-specific market ID
    if not market_id:
        logger.error(f"Market ID missing in market_info for {symbol}. Cannot reliably fetch position.")
        return None

    positions: List[Dict[str, Any]] = []
    try:
        logger.debug(f"Fetching position for {symbol} (Market ID: {market_id})")
        # Standard CCXT way: fetch positions for specific symbols if supported
        fetched_positions_raw = await exchange.fetch_positions([symbol])
        # Filter again, as some exchanges might return all positions even when one symbol is requested
        positions = [p for p in fetched_positions_raw if p.get("symbol") == symbol]

    except ccxt_async.ArgumentsRequired:  # If fetchPositions([symbols]) is not supported, try fetching all
        logger.debug(
            f"fetchPositions for {exchange.id} with symbol argument failed or is not supported. "
            f"Attempting to fetch all positions and filter for {symbol}."
        )
        try:
            all_positions = await exchange.fetch_positions()
            # Filter by symbol or market_id from the 'info' field for robustness
            positions = [
                p
                for p in all_positions
                if p.get("symbol") == symbol or (p.get("info") and p["info"].get("symbol") == market_id)
            ]
        except Exception as e_all:
            logger.error(f"Error fetching all positions while trying to find {symbol}: {e_all}", exc_info=True)
            return None
    except ccxt_async.ExchangeError as e:
        # Specific error messages/codes indicating no position exists
        no_pos_indicators = ["position not found", "no position", "position does not exist"]
        # Bybit: 110025 (position not found), 10001 (can sometimes mean no position for certain requests)
        if any(msg in str(e).lower() for msg in no_pos_indicators) or (
            exchange.id == "bybit" and getattr(e, "code", None) in [110025, 10001]
        ):  # Bybit specific
            logger.info(f"No open position found for {symbol} (Exchange reported: {e}).")
            return None
        logger.error(f"Exchange error fetching positions for {symbol}: {e}.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching positions for {symbol}: {e}", exc_info=True)
        return None

    if not positions:
        logger.info(f"No position data structures returned or matched for {symbol}.")
        return None

    active_position_data = None
    # Determine a small threshold based on amount precision to filter out dust positions
    raw_amount_step = market_info.get("precision", {}).get("amount", "1e-8")
    try:
        size_threshold = Decimal(str(raw_amount_step)) / Decimal("100")
        if size_threshold <= 0:
            size_threshold = Decimal("1e-9")  # Ensure positive threshold
    except InvalidOperation:
        size_threshold = Decimal("1e-9")  # Fallback if amount_step is invalid

    for pos_data in positions:
        # Try to get position size from common fields ('contracts', 'info.size', 'info.qty')
        size_str = (
            pos_data.get("contracts") or pos_data.get("info", {}).get("size") or pos_data.get("info", {}).get("qty")
        )
        if size_str is None:
            continue

        try:
            pos_size_dec = Decimal(str(size_str))

            # Bybit V5 specific: 'positionSide' can be 'None' for closed/zero positions in hedge mode
            bybit_v5_pos_side = pos_data.get("info", {}).get("positionSide", "").lower()
            if exchange.id == "bybit" and bybit_v5_pos_side == "none" and abs(pos_size_dec) <= size_threshold:
                continue  # Skip Bybit "None" side positions that are effectively zero

            if abs(pos_size_dec) > size_threshold:  # Position is considered active
                active_position_data = pos_data.copy()  # Work with a copy

                # Standardize 'contractsDecimal' to be positive absolute size
                active_position_data["contractsDecimal"] = abs(pos_size_dec)

                # Standardize 'side' ('long' or 'short')
                current_side = active_position_data.get("side", "").lower()
                if not current_side or current_side == "none":  # Infer if 'side' is missing or 'None'
                    if exchange.id == "bybit" and bybit_v5_pos_side in ["buy", "sell"]:
                        current_side = "long" if bybit_v5_pos_side == "buy" else "short"
                    elif pos_size_dec > size_threshold:  # Positive size implies long
                        current_side = "long"
                    elif pos_size_dec < -size_threshold:  # Negative size implies short (for some exchanges)
                        current_side = "short"
                    else:
                        continue  # Ambiguous or zero size, skip
                active_position_data["side"] = current_side

                # Entry Price
                ep_str = active_position_data.get("entryPrice") or active_position_data.get("info", {}).get("avgPrice")
                active_position_data["entryPriceDecimal"] = Decimal(str(ep_str)) if ep_str is not None else None

                # Map various potential field names to standardized Decimal keys
                field_map = {
                    "markPriceDecimal": ["markPrice"],
                    "liquidationPriceDecimal": ["liquidationPrice", "liqPrice"],
                    "unrealizedPnlDecimal": ["unrealizedPnl", "unrealisedPnl", "pnl", ("info", "unrealisedPnl")],
                    "stopLossPriceDecimal": ["stopLoss", "stopLossPrice", "slPrice", ("info", "stopLoss")],
                    "takeProfitPriceDecimal": ["takeProfit", "takeProfitPrice", "tpPrice", ("info", "takeProfit")],
                    "trailingStopLossValue": [
                        ("info", "trailingStop"),
                        ("info", "trailing_stop"),
                        ("info", "tpslTriggerPrice"),
                    ],  # Bybit: distance or price
                    "trailingStopActivationPrice": [
                        ("info", "activePrice"),
                        ("info", "triggerPrice"),
                        ("info", "trailing_trigger_price"),
                    ],
                }
                for dec_key, str_keys_list in field_map.items():
                    val_str = None
                    for sk_item in str_keys_list:
                        if isinstance(sk_item, tuple):  # e.g., ('info', 'someKey')
                            val_str = active_position_data.get(sk_item[0], {}).get(sk_item[1])
                        else:
                            val_str = active_position_data.get(sk_item)
                        if val_str is not None:
                            break  # Found a value

                    if val_str is not None and str(val_str).strip():  # Ensure not empty string
                        # For SL/TP, "0" often means not set; treat as None for consistency
                        if str(val_str) == "0" and dec_key in ["stopLossPriceDecimal", "takeProfitPriceDecimal"]:
                            active_position_data[dec_key] = None
                        else:
                            try:
                                active_position_data[dec_key] = Decimal(str(val_str))
                            except (InvalidOperation, TypeError):
                                active_position_data[dec_key] = None  # Failed conversion
                    else:
                        active_position_data[dec_key] = None  # Not found or empty

                # Timestamp (ms)
                ts_str = (
                    active_position_data.get("timestamp")
                    or active_position_data.get("info", {}).get("updatedTime")
                    or active_position_data.get("info", {}).get("updated_at")
                    or active_position_data.get("info", {}).get("createTime")
                )
                active_position_data["timestamp_ms"] = int(float(ts_str)) if ts_str else None
                break  # Found and processed an active position
        except (InvalidOperation, ValueError, TypeError) as e:
            logger.warning(f"Error parsing position data for {symbol}: {e}. Data: {pos_data}", exc_info=True)
            continue  # Try next position data if parsing fails

    if active_position_data:
        logger.info(
            f"Active {active_position_data.get('side', 'N/A').upper()} position found for {symbol}: "
            f"Size={active_position_data.get('contractsDecimal', 'N/A')}, "
            f"Entry={active_position_data.get('entryPriceDecimal', 'N/A')}"
        )
        return active_position_data

    logger.info(f"No active open position found for {symbol} after filtering (size > {size_threshold:.8f}).")
    return None


async def set_leverage_ccxt(
    exchange: ccxt_async.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger
) -> bool:
    if not market_info.get("is_contract", False):
        logger.info(f"Leverage setting skipped for {symbol} as it's not a contract market.")
        return True  # No action needed, considered success

    if not (isinstance(leverage, int) and leverage > 0):
        logger.warning(f"Invalid leverage value {leverage} for {symbol}. Must be a positive integer.")
        return False

    if not (hasattr(exchange, "set_leverage") and callable(exchange.set_leverage)):
        logger.error(f"Exchange {exchange.id} does not support set_leverage method via CCXT.")
        return False

    logger.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
    params = {}
    # Bybit V5 might require buyLeverage and sellLeverage for unified margin, or for hedge mode positions.
    # For one-way mode on derivatives, setting leverage for the symbol is usually enough.
    # CCXT's set_leverage should handle underlying requirements.
    # If specific params are needed for Bybit (e.g. positionIdx for hedge mode), they should be passed via a config or determined.
    # For now, this is a general approach. Bybit example:
    # if exchange.id == 'bybit': params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}

    try:
        response = await exchange.set_leverage(leverage, symbol, params=params)
        logger.debug(f"Set leverage response for {symbol}: {response}")

        # Specific handling for Bybit V5 response
        if exchange.id == "bybit" and isinstance(response, dict):
            ret_code = response.get("retCode")
            ret_msg = response.get("retMsg", "").lower()
            if ret_code == 0:
                logger.info(f"{NEON_GREEN}Leverage for {symbol} successfully set to {leverage}x (Bybit).{RESET}")
                return True
            # Bybit: 110043 means "Leverage not modified"
            elif ret_code == 110043 or "leverage not modified" in ret_msg or "same leverage" in ret_msg:
                logger.info(f"Leverage for {symbol} was already {leverage}x (Bybit: {ret_code} - {ret_msg}).")
                return True
            else:
                logger.error(f"Bybit error setting leverage for {symbol}: {ret_msg} (Code: {ret_code})")
                return False

        # Generic success: if no exception and not a specific Bybit failure
        logger.info(f"{NEON_GREEN}Leverage for {symbol} set/requested to {leverage}x (Generic CCXT response).{RESET}")
        return True
    except ccxt_async.ExchangeError as e:
        err_str, code = str(e).lower(), getattr(e, "code", None)
        # Check if error message indicates leverage was already set (common for some exchanges)
        if (
            "leverage not modified" in err_str or "no change" in err_str or (exchange.id == "bybit" and code == 110043)
        ):  # Bybit: 110043
            logger.info(f"Leverage for {symbol} already {leverage}x (Confirmed by error: {e}).")
            return True
        logger.error(f"Exchange error setting leverage for {symbol} to {leverage}x: {e} (Code: {code})")
    except Exception as e:
        logger.error(f"Unexpected error setting leverage for {symbol} to {leverage}x: {e}", exc_info=True)
    return False


async def place_trade(
    exchange: ccxt_async.Exchange,
    symbol: str,
    trade_signal: str,  # "BUY" or "SELL"
    position_size: Decimal,
    market_info: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    order_type: str = "market",
    limit_price: Optional[Decimal] = None,
    reduce_only: bool = False,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:  # Returns the CCXT order object if successful
    current_logger = logger or module_logger
    side = "buy" if trade_signal.upper() == "BUY" else "sell"
    action_description = "Reduce-Only" if reduce_only else "Open/Increase"

    try:
        if not (isinstance(position_size, Decimal) and position_size > 0):
            current_logger.error(
                f"Trade aborted for {symbol} ({side}): Invalid position_size ({position_size}). Must be a positive Decimal."
            )
            return None

        # Format amount to exchange's precision rules
        amount_str_for_api = exchange.amount_to_precision(symbol, float(position_size))
        amount_for_api = float(amount_str_for_api)  # CCXT generally expects float for amount

        if amount_for_api <= 0:
            current_logger.error(
                f"Trade aborted for {symbol} ({side}): Position size after formatting ({amount_for_api}) is not positive."
            )
            return None
    except Exception as e:
        current_logger.error(
            f"Trade aborted for {symbol} ({side}): Error formatting position_size {position_size}: {e}", exc_info=True
        )
        return None

    price_for_api: Optional[float] = None
    price_log_str: Optional[str] = None  # For logging purposes
    if order_type.lower() == "limit":
        if not (isinstance(limit_price, Decimal) and limit_price > 0):
            current_logger.error(
                f"Trade aborted for {symbol} ({side}): Limit order chosen, but invalid limit_price ({limit_price})."
            )
            return None
        try:
            price_log_str = exchange.price_to_precision(symbol, float(limit_price))
            price_for_api = float(price_log_str)  # CCXT generally expects float for price
            if price_for_api <= 0:
                raise ValueError("Formatted limit price is not positive.")
        except Exception as e:
            current_logger.error(
                f"Trade aborted for {symbol} ({side}): Error formatting limit_price {limit_price}: {e}", exc_info=True
            )
            return None
    elif order_type.lower() != "market":
        current_logger.error(f"Unsupported order type '{order_type}' for {symbol}. Only 'market' or 'limit' supported.")
        return None

    # Prepare parameters for the order
    final_params = {"reduceOnly": reduce_only}
    if exchange.id == "bybit":
        # For Bybit V5, positionIdx=0 for One-Way mode.
        # For Hedge Mode, 1 for Buy side, 2 for Sell side. This needs more context if hedge mode is used.
        # Assuming One-Way mode as default for simplicity here.
        final_params["positionIdx"] = 0

    if params:  # Merge any additional user-supplied params
        final_params.update(params)

    # For market reduce-only orders, IOC (Immediate Or Cancel) is often preferred/required
    if reduce_only and order_type.lower() == "market" and "timeInForce" not in final_params:
        final_params["timeInForce"] = "IOC"

    base_currency = market_info.get("base", "units")
    log_message = (
        f"Placing {action_description} {side.upper()} {order_type.upper()} order for {symbol}: "
        f"Size = {amount_for_api} {base_currency}"
    )
    if price_log_str:
        log_message += f", Price = {price_log_str}"
    log_message += f", Params = {final_params}"
    current_logger.info(log_message)

    try:
        order = await exchange.create_order(
            symbol, order_type.lower(), side, amount_for_api, price_for_api, final_params
        )
        if order:
            current_logger.info(
                f"{NEON_GREEN}{action_description} order for {symbol} PLACED successfully. "
                f"ID: {order.get('id')}, Status: {order.get('status', 'N/A')}{RESET}"
            )
            return order
        else:
            # This case (order is None without exception) should be rare with CCXT
            current_logger.error(f"Order placement for {symbol} returned None without raising an exception.")
            return None
    except ccxt_async.InsufficientFunds as e:
        current_logger.error(
            f"{NEON_RED}Insufficient funds to place {side} {order_type} order for {symbol}: {e}{RESET}"
        )
    except ccxt_async.InvalidOrder as e:
        current_logger.error(
            f"{NEON_RED}Invalid order parameters for {symbol} ({side}, {order_type}): {e}. "
            f"Details: Amount={amount_for_api}, Price={price_for_api}, Params={final_params}{RESET}",
            exc_info=True,
        )
    except ccxt_async.ExchangeError as e:  # Broader exchange errors
        current_logger.error(
            f"{NEON_RED}Exchange error placing {action_description} order for {symbol}: {e}{RESET}", exc_info=True
        )
    except Exception as e:  # Other unexpected errors
        current_logger.error(
            f"{NEON_RED}Unexpected error placing {action_description} order for {symbol}: {e}{RESET}", exc_info=True
        )
    return None


async def _set_position_protection(
    exchange: ccxt_async.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Union[Decimal, str]] = None,  # Can be Decimal distance or "0" to remove
    tsl_activation_price: Optional[Union[Decimal, str]] = None,  # Can be Decimal price or "0" for immediate
) -> bool:
    if "bybit" not in exchange.id.lower():  # This logic is highly specific to Bybit V5 API
        logger.error("Position protection logic (_set_position_protection) is currently Bybit V5 specific.")
        return False
    if not market_info.get("is_contract", False):
        logger.warning(f"Protection skipped for {symbol}: not a contract market.")
        return True  # No action needed if not a contract

    if not position_info or "side" not in position_info or not position_info["side"]:
        logger.error(f"Cannot set protection for {symbol}: invalid or missing position_info (especially 'side').")
        return False

    pos_side_str = position_info["side"].lower()  # 'long' or 'short'
    # Bybit V5 positionIdx: 0 for one-way, 1 for buy (long) in hedge, 2 for sell (short) in hedge
    # Assuming one-way mode (positionIdx=0) if not specified otherwise.
    # This might need to be configurable or detected if hedge mode is used.
    pos_idx_raw = position_info.get("info", {}).get("positionIdx", 0)
    try:
        position_idx = int(pos_idx_raw)
    except (ValueError, TypeError):
        logger.warning(f"Invalid positionIdx '{pos_idx_raw}' for {symbol}, defaulting to 0.")
        position_idx = 0

    # Determine category (linear/inverse) for Bybit V5 API
    market_type = market_info.get("type", "").lower()
    if market_info.get("linear", False) or market_type == "linear":
        category = "linear"
    elif market_info.get("inverse", False) or market_type == "inverse":
        category = "inverse"
    elif market_info.get("spot", False) or market_type == "spot":
        category = "spot"  # Protection on spot might not be supported or behave differently
        logger.warning(
            f"Attempting to set protection on SPOT symbol {symbol}, this may not be fully supported by Bybit's position protection endpoint."
        )
    else:  # Fallback, try 'linear' if type is ambiguous (e.g. 'swap')
        category = "linear"
        logger.warning(
            f"Market category for {symbol} is ambiguous (type: {market_type}). Defaulting to 'linear' for protection API call."
        )

    api_params: Dict[str, Any] = {
        "category": category,
        "symbol": market_info["id"],  # Exchange-specific symbol ID
        "positionIdx": position_idx,
    }
    log_parts = [
        f"Attempting to set/update protection for {symbol} ({pos_side_str.upper()}, PosIdx:{position_idx}, Cat:{category}):"
    ]
    protection_fields_to_send: Dict[str, str] = {}  # Fields like stopLoss, takeProfit, trailingStop, activePrice

    try:
        price_precision_places = get_price_precision(market_info, logger)  # Number of decimal places for price
        min_tick_size_dec = get_min_tick_size(market_info, logger)
        if not (min_tick_size_dec and min_tick_size_dec > 0):  # Fallback if min_tick_size is invalid
            min_tick_size_dec = Decimal(f"1e-{price_precision_places}")

        def format_price_for_api(price_decimal: Optional[Decimal]) -> Optional[str]:
            if not (price_decimal and isinstance(price_decimal, Decimal) and price_decimal > 0):
                return None  # Invalid input or zero price (unless "0" is explicitly allowed for removal)
            return exchange.price_to_precision(symbol, float(price_decimal))

        # Trailing Stop (TSL)
        # Bybit: 'trailingStop' is the distance value. 'activePrice' is the trigger price.
        # 'trailing_stop_distance' can be Decimal (for value) or "0" (string, to remove TSL).
        # 'tsl_activation_price' can be Decimal (for value) or "0" (string, for immediate activation).
        if isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0:
            # Calculate precision for distance (usually same as price precision for Bybit)
            distance_precision_places = abs(min_tick_size_dec.normalize().as_tuple().exponent)
            tsl_dist_str = exchange.decimal_to_precision(
                trailing_stop_distance,
                ccxt_async.ROUND,
                distance_precision_places,
                ccxt_async.DECIMAL_PLACES,
                ccxt_async.NO_PADDING,
            )
            # Ensure formatted distance is at least one tick
            if Decimal(tsl_dist_str) < min_tick_size_dec:
                tsl_dist_str = str(min_tick_size_dec.quantize(Decimal(f"1e-{distance_precision_places}")))

            tsl_act_price_str_final: Optional[str] = None
            if tsl_activation_price == "0":  # Special string "0" for immediate activation
                tsl_act_price_str_final = "0"
            elif isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0:
                tsl_act_price_str_final = format_price_for_api(tsl_activation_price)

            if tsl_dist_str and Decimal(tsl_dist_str) > 0 and tsl_act_price_str_final is not None:
                protection_fields_to_send.update({"trailingStop": tsl_dist_str, "activePrice": tsl_act_price_str_final})
                log_parts.append(
                    f"  - Trailing Stop: Distance={tsl_dist_str}, ActivationPrice={tsl_act_price_str_final}"
                )
                # Bybit: Setting TSL via this endpoint might override/clear fixed SL/TP.
                # Or, they might coexist. API docs are key. Assuming TSL takes precedence or is primary when set.
                # If TSL is set, fixed SL might be implicitly managed or ignored by this specific API call.
                # For safety, if TSL is set, don't send fixed SL unless API confirms they combine.
                # Here, we allow both to be sent if provided; Bybit will decide precedence.
                # stop_loss_price = None # Uncomment if TSL explicitly overrides fixed SL
            else:
                logger.error(
                    f"Failed to format TSL parameters for {symbol}. "
                    f"DistanceInput='{trailing_stop_distance}', FormattedDist='{tsl_dist_str}', "
                    f"ActivationInput='{tsl_activation_price}', FormattedAct='{tsl_act_price_str_final}'"
                )
        elif trailing_stop_distance == "0":  # Explicitly remove TSL
            protection_fields_to_send["trailingStop"] = "0"
            # When removing TSL, activePrice might also need to be "0" or omitted.
            # Bybit docs: "To cancel the TS, set trailingStop to '0'." activePrice seems not needed then.
            # If activePrice was in api_params from a previous logic path, ensure it's handled.
            # Here, protection_fields_to_send is built fresh, so no old activePrice.
            log_parts.append("  - Trailing Stop: Removing (distance set to '0')")

        # Fixed Stop Loss
        if stop_loss_price is not None:  # Allows Decimal(0) to remove SL
            sl_price_str = "0" if stop_loss_price == Decimal(0) else format_price_for_api(stop_loss_price)
            if sl_price_str is not None:  # format_price_for_api returns None for invalid input
                protection_fields_to_send["stopLoss"] = sl_price_str
                log_parts.append(f"  - Fixed Stop Loss: {sl_price_str}")

        # Fixed Take Profit
        if take_profit_price is not None:  # Allows Decimal(0) to remove TP
            tp_price_str = "0" if take_profit_price == Decimal(0) else format_price_for_api(take_profit_price)
            if tp_price_str is not None:
                protection_fields_to_send["takeProfit"] = tp_price_str
                log_parts.append(f"  - Fixed Take Profit: {tp_price_str}")

    except Exception as fmt_err:
        logger.error(f"Error formatting protection parameters for {symbol}: {fmt_err}", exc_info=True)
        return False

    if not protection_fields_to_send:
        logger.info(f"No valid protection parameters to set or update for {symbol}.")
        return True  # Nothing to do, considered success

    api_params.update(protection_fields_to_send)
    logger.info("\n".join(log_parts))
    logger.debug(f"  API Call to set trading stop/protection for {symbol}: params={api_params}")

    try:
        # Bybit V5 endpoint for SL/TP/TSL
        method_name_camel = "v5PrivatePostPositionSetTradingStop"
        method_name_snake = "v5_private_post_position_set_trading_stop"  # CCXT might use snake_case

        if hasattr(exchange, method_name_camel):
            set_protection_method = getattr(exchange, method_name_camel)
        elif hasattr(exchange, method_name_snake):
            set_protection_method = getattr(exchange, method_name_snake)
        else:
            logger.error(
                f"CCXT instance for {exchange.id} is missing the required method for setting position protection "
                f"(checked for '{method_name_camel}' and '{method_name_snake}'). "
                f"Ensure CCXT library is up-to-date and supports Bybit V5 position protection."
            )
            return False

        response = await set_protection_method(api_params)
        logger.debug(f"Set protection raw API response for {symbol}: {response}")

        if isinstance(response, dict) and response.get("retCode") == 0:
            logger.info(
                f"{NEON_GREEN}Protection for {symbol} successfully set/updated. "
                f"Message: {response.get('retMsg', 'OK')}{RESET}"
            )
            return True
        else:
            logger.error(
                f"{NEON_RED}Failed to set protection for {symbol}. "
                f"API Response Code: {response.get('retCode')}, Message: {response.get('retMsg')}{RESET}"
            )
            return False
    except Exception as e:
        logger.error(f"{NEON_RED}Error during API call to set protection for {symbol}: {e}{RESET}", exc_info=True)
        return False


async def set_trailing_stop_loss(
    exchange: ccxt_async.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    config: Dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None,  # Optionally set TP along with TSL
) -> bool:
    if not config.get("enable_trailing_stop", False):
        logger.info(f"Trailing Stop Loss is disabled in config for {symbol}.")
        # If TSL disabled but TP is provided, still try to set TP
        if take_profit_price and isinstance(take_profit_price, Decimal) and take_profit_price > 0:
            logger.info(f"TSL disabled, but attempting to set provided Take Profit for {symbol}.")
            return await _set_position_protection(
                exchange, symbol, market_info, position_info, logger, take_profit_price=take_profit_price
            )
        return True  # No TSL action needed, considered success in this context

    if not market_info.get("is_contract", False):
        logger.warning(f"Trailing Stop Loss is typically for contract markets. Skipped for {symbol} (not a contract).")
        # If TP is provided for a non-contract, still try to set it if API allows (e.g. spot TP orders)
        if take_profit_price and isinstance(take_profit_price, Decimal) and take_profit_price > 0:
            logger.info(f"Market is not a contract, but attempting to set provided Take Profit for {symbol}.")
            return await _set_position_protection(
                exchange, symbol, market_info, position_info, logger, take_profit_price=take_profit_price
            )
        return True

    try:
        # Load TSL parameters from config
        callback_rate_str = str(config.get("trailing_stop_callback_rate", "0.005"))  # e.g., 0.5%
        activation_percentage_str = str(config.get("trailing_stop_activation_percentage", "0.003"))  # e.g., 0.3%

        callback_rate = Decimal(callback_rate_str)
        activation_percentage = Decimal(activation_percentage_str)

        if callback_rate <= 0:
            raise ValueError("Trailing stop callback rate must be positive.")
        if activation_percentage < 0:  # Allow 0 for activation at entry or immediate
            raise ValueError("Trailing stop activation percentage must be non-negative.")
    except (InvalidOperation, ValueError, TypeError) as e:
        logger.error(f"Invalid TSL parameters in configuration for {symbol}: {e}. Please check config.")
        return False

    try:
        entry_price = position_info.get("entryPriceDecimal")
        position_side = position_info.get("side", "").lower()  # 'long' or 'short'
        if not (isinstance(entry_price, Decimal) and entry_price > 0):
            raise ValueError(f"Invalid or missing entry price in position_info: {entry_price}")
        if position_side not in ["long", "short"]:
            raise ValueError(f"Invalid or missing side in position_info: {position_side}")
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid position information for TSL setup ({symbol}): {e}. Position: {position_info}")
        return False

    try:
        price_precision_places = get_price_precision(market_info, logger)
        min_tick_size = get_min_tick_size(market_info, logger)
        # Fallback tick size if not properly defined in market_info
        quantize_fallback_tick = Decimal(f"1e-{price_precision_places}")
        effective_tick_size = min_tick_size if min_tick_size and min_tick_size > 0 else quantize_fallback_tick
        if not (effective_tick_size > 0):  # Should not happen with fallback
            logger.error(f"Could not determine a valid tick size for TSL calculations for {symbol}.")
            return False

        # Fetch current market price to aid activation logic, fallback to entry price
        current_market_price = await fetch_current_price_ccxt(exchange, symbol, logger)
        if not current_market_price:
            logger.warning(
                f"Could not fetch current market price for {symbol} for TSL logic. Using entry price as reference."
            )
            current_market_price = entry_price

        # Calculate theoretical activation price based on entry price and activation percentage
        price_change_for_activation = entry_price * activation_percentage
        raw_activation_price = entry_price + (
            price_change_for_activation if position_side == "long" else -price_change_for_activation
        )

        # Determine if position is already profitable enough for TSL to activate immediately (Bybit: activePrice="0")
        activate_immediately = False
        if config.get("tsl_activate_immediately_if_profitable", True):
            if position_side == "long" and current_market_price >= raw_activation_price:
                activate_immediately = True
            elif position_side == "short" and current_market_price <= raw_activation_price:
                activate_immediately = True

        final_activation_price_param: Union[Decimal, str]  # This will be passed to _set_position_protection
        calculated_activation_price_for_log: Optional[Decimal] = None  # For logging clarity

        if activate_immediately:
            final_activation_price_param = "0"  # Bybit: "0" for activePrice means activate immediately
            calculated_activation_price_for_log = current_market_price  # Log current price as reference
            logger.info(
                f"TSL for {symbol} ({position_side}): Position is already profitable beyond activation point. "
                f"Setting activePrice='0' for immediate trailing based on current market price ({current_market_price})."
            )
        else:
            # Calculate specific activation price, ensuring it's profitable and respects market tick size
            min_profit_activation_price = (
                entry_price + effective_tick_size if position_side == "long" else entry_price - effective_tick_size
            )

            if position_side == "long":
                if raw_activation_price < min_profit_activation_price:
                    raw_activation_price = min_profit_activation_price
                # Ensure activation price is ahead of current market price if not activating immediately
                if raw_activation_price < current_market_price:
                    raw_activation_price = current_market_price + effective_tick_size
                rounding_mode = ROUND_UP
            else:  # short
                if raw_activation_price > min_profit_activation_price:
                    raw_activation_price = min_profit_activation_price
                if raw_activation_price > current_market_price:
                    raw_activation_price = current_market_price - effective_tick_size
                rounding_mode = ROUND_DOWN

            # Quantize to tick size
            calculated_activation_price = (raw_activation_price / effective_tick_size).quantize(
                Decimal("1"), rounding=rounding_mode
            ) * effective_tick_size

            # Final validation of calculated activation price
            if calculated_activation_price <= 0:
                logger.error(
                    f"Calculated TSL Activation Price ({calculated_activation_price}) is not positive for {symbol}. Cannot set TSL."
                )
                return False
            if position_side == "long" and calculated_activation_price <= entry_price:
                logger.warning(
                    f"Calculated TSL Activation Price ({calculated_activation_price}) for LONG {symbol} is not profitable vs Entry ({entry_price}). "
                    f"Adjusting to one tick above entry."
                )
                calculated_activation_price = ((entry_price + effective_tick_size) / effective_tick_size).quantize(
                    Decimal("1"), rounding=ROUND_UP
                ) * effective_tick_size
            elif position_side == "short" and calculated_activation_price >= entry_price:
                logger.warning(
                    f"Calculated TSL Activation Price ({calculated_activation_price}) for SHORT {symbol} is not profitable vs Entry ({entry_price}). "
                    f"Adjusting to one tick below entry."
                )
                calculated_activation_price = ((entry_price - effective_tick_size) / effective_tick_size).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * effective_tick_size

            if calculated_activation_price <= 0:  # Re-check after adjustment
                logger.error(
                    f"Final TSL Activation Price ({calculated_activation_price}) non-positive for {symbol}. Cannot set TSL."
                )
                return False

            final_activation_price_param = calculated_activation_price
            calculated_activation_price_for_log = calculated_activation_price

        # Calculate trailing distance based on callback rate and entry price, quantize to tick size
        raw_trail_distance = entry_price * callback_rate
        trail_distance = (raw_trail_distance / effective_tick_size).quantize(
            Decimal("1"), rounding=ROUND_UP
        ) * effective_tick_size
        if trail_distance < effective_tick_size:  # Ensure distance is at least one tick
            trail_distance = effective_tick_size
        if trail_distance <= 0:
            logger.error(
                f"Calculated TSL trail distance ({trail_distance}) is not positive for {symbol}. Cannot set TSL."
            )
            return False

        log_act_price_str = (
            f"{calculated_activation_price_for_log:.{price_precision_places}f}"
            if calculated_activation_price_for_log
            else "N/A (Immediate)"
        )
        logger.info(
            f"Calculated TSL parameters for {symbol} ({position_side.upper()}):\n"
            f"  Entry Price: {entry_price:.{price_precision_places}f}\n"
            f"  Activation Price (for API): '{final_activation_price_param}' (Based on calculated: {log_act_price_str}, from {activation_percentage:.2%})\n"
            f"  Trail Distance: {trail_distance:.{price_precision_places}f} (From callback rate: {callback_rate:.2%})"
        )
        if take_profit_price and isinstance(take_profit_price, Decimal) and take_profit_price > 0:
            logger.info(f"  Also setting Take Profit at: {take_profit_price:.{price_precision_places}f}")

        return await _set_position_protection(
            exchange,
            symbol,
            market_info,
            position_info,
            logger,
            stop_loss_price=None,  # TSL typically replaces/manages the stop loss part
            take_profit_price=take_profit_price
            if isinstance(take_profit_price, Decimal) and take_profit_price > 0
            else None,
            trailing_stop_distance=trail_distance,
            tsl_activation_price=final_activation_price_param,  # This can be Decimal or string "0"
        )
    except Exception as e:
        logger.error(f"Unexpected error during TSL setup for {symbol}: {e}", exc_info=True)
        return False  # File: config_loader.py


import json
import os
from decimal import Decimal
from typing import Any, Dict, Optional, Union

# Import constants and color codes from utils
from utils import (
    CONFIG_FILE,
    DEFAULT_INDICATOR_PERIODS,
    POSITION_CONFIRM_DELAY_SECONDS,
    RETRY_DELAY_SECONDS,
    VALID_INTERVALS,
)


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
        # Optional: Check type consistency for non-dict items if needed
        # elif default_value is not None and not isinstance(updated_config.get(key), type(default_value)):
        #     # Allow None to be overridden, but check other types
        #     print(f"Warning: Config type mismatch for key '{key}'. Expected {type(default_value)}, got {type(updated_config.get(key))}. Using default.")
        #     updated_config[key] = default_value
    return updated_config


def load_config(filepath: str = CONFIG_FILE) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found,
    and ensuring all default keys are present with validation."""
    default_config = {
        "symbols_to_trade": ["FARTCOIN/USDT:USDT"],  # List of symbols (e.g., "FARTCOIN/USDT:USDT" for Bybit linear)
        "interval": "5",  # Default to '5' (map to 5m later)
        "retry_delay": RETRY_DELAY_SECONDS,
        "orderbook_limit": 25,  # Depth of orderbook to fetch
        "signal_score_threshold": 1.5,  # Score needed to trigger BUY/SELL signal
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8,  # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7,  # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5,  # How much higher volume needs to be than MA
        "scalping_signal_threshold": 2.5,  # Separate threshold for 'scalping' weight set
        "enable_trading": True,  # SAFETY FIRST: Default to True, enable consciously
        "use_sandbox": False,  # SAFETY FIRST: Default to False (testnet), disable consciously
        "risk_per_trade": 0.01,  # Risk 1% of account balance per trade (0 to 1)
        "leverage": 20,  # Set desired leverage (integer > 0)
        "max_concurrent_positions": 1,  # Limit open positions for this symbol
        "quote_currency": "USDT",  # Currency for balance check and sizing
        "entry_order_type": "market",  # "market" or "limit"
        "limit_order_offset_buy": 0.0005,  # Percentage offset from current price for BUY limit orders (e.g., 0.0005 = 0.05%)
        "limit_order_offset_sell": 0.0005,  # Percentage offset from current price for SELL limit orders (e.g., 0.0005 = 0.05%)
        # --- Trailing Stop Loss Config (Exchange-Native) ---
        "enable_trailing_stop": True,  # Default to enabling TSL (exchange TSL)
        "trailing_stop_callback_rate": 0.005,  # e.g., 0.5% trail distance (as decimal > 0) from high water mark
        "trailing_stop_activation_percentage": 0.003,  # e.g., Activate TSL when price moves 0.3% in favor from entry (>= 0)
        # --- Break-Even Stop Config ---
        "enable_break_even": True,  # Enable moving SL to break-even
        "break_even_trigger_atr_multiple": 1.0,  # Move SL when profit >= X * ATR (> 0)
        "break_even_offset_ticks": 2,  # Place BE SL X ticks beyond entry price (integer >= 0)
        # --- Position Management ---
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,  # Delay after order before checking position status (>= 0)
        "time_based_exit_minutes": None,  # Optional: Exit position after X minutes (> 0). Set to None or 0 to disable.
        # --- Indicator Control ---
        "indicators": {  # Control which indicators are calculated and contribute to score
            "ema_alignment": True,
            "momentum": True,
            "volume_confirmation": True,
            "stoch_rsi": True,
            "rsi": True,
            "bollinger_bands": True,
            "vwap": True,
            "cci": True,
            "wr": True,
            "psar": True,
            "sma_10": True,
            "mfi": True,
            "orderbook": True,  # Flag to enable fetching and scoring orderbook data
        },
        "weight_sets": {  # Define different weighting strategies
            "scalping": {  # Example weighting for a fast scalping strategy
                "ema_alignment": 0.2,
                "momentum": 0.3,
                "volume_confirmation": 0.2,
                "stoch_rsi": 0.6,
                "rsi": 0.2,
                "bollinger_bands": 0.3,
                "vwap": 0.4,
                "cci": 0.3,
                "wr": 0.3,
                "psar": 0.2,
                "sma_10": 0.1,
                "mfi": 0.2,
                "orderbook": 0.15,
            },
            "default": {  # A more balanced weighting strategy
                "ema_alignment": 0.3,
                "momentum": 0.2,
                "volume_confirmation": 0.1,
                "stoch_rsi": 0.4,
                "rsi": 0.3,
                "bollinger_bands": 0.2,
                "vwap": 0.3,
                "cci": 0.2,
                "wr": 0.2,
                "psar": 0.3,
                "sma_10": 0.1,
                "mfi": 0.2,
                "orderbook": 0.1,
            },
        },
        "active_weight_set": "default",  # Choose which weight set to use ("default" or "scalping")
    }
    # Add default indicator periods to the default config
    default_config.update(DEFAULT_INDICATOR_PERIODS)

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET}")
            return default_config  # Return default if creation failed

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
        # Ensure all keys from default are present, add missing ones
        updated_config = _ensure_config_keys(config_from_file, default_config)
        # If updates were made, write them back
        if updated_config != config_from_file:
            try:
                with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4)
                print(f"{NEON_YELLOW}Updated config file with missing default keys: {filepath}{RESET}")
            except IOError as e:
                print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")

        # --- Validate crucial values after loading/updating ---
        save_needed = False  # Flag to save config if corrections are made

        # Validate interval
        if updated_config.get("interval") not in VALID_INTERVALS:
            print(
                f"{NEON_RED}Invalid interval '{updated_config.get('interval')}' found in config. Using default '{default_config['interval']}'.{RESET}"
            )
            updated_config["interval"] = default_config["interval"]
            save_needed = True

        # Validate entry order type
        if updated_config.get("entry_order_type") not in ["market", "limit"]:
            print(
                f"{NEON_RED}Invalid entry_order_type '{updated_config.get('entry_order_type')}' in config. Using default 'market'.{RESET}"
            )
            updated_config["entry_order_type"] = "market"
            save_needed = True

        # Validate numeric ranges
        def validate_numeric(
            key: str,
            min_val: Optional[Union[int, float]] = None,
            max_val: Optional[Union[int, float]] = None,
            is_int: bool = False,
            allow_none: bool = False,
        ):
            nonlocal save_needed
            value = updated_config.get(key)
            default_value = default_config.get(key)
            valid = False

            if allow_none and value is None:
                valid = True
            # Explicitly check for bool type and exclude it from numeric validation
            elif isinstance(value, bool):
                print(f"{NEON_RED}Config value '{key}' ({value}) has invalid type bool. Expected numeric.")
            elif isinstance(value, (int, float)):
                if is_int and not isinstance(value, int):
                    print(f"{NEON_RED}Config value '{key}' ({value}) must be an integer.")
                else:
                    try:
                        val_decimal = Decimal(str(value))  # Use Decimal for comparison
                        min_decimal = Decimal(str(min_val)) if min_val is not None else None
                        max_decimal = Decimal(str(max_val)) if max_val is not None else None

                        if (min_decimal is None or val_decimal >= min_decimal) and (
                            max_decimal is None or val_decimal <= max_decimal
                        ):
                            valid = True
                        else:
                            range_str = ""
                            if min_val is not None:
                                range_str += f" >= {min_val}"
                            if max_val is not None:
                                range_str += f" <= {max_val}"
                            print(f"{NEON_RED}Config value '{key}' ({value}) out of range ({range_str.strip()}).")
                    except InvalidOperation:
                        print(
                            f"{NEON_RED}Config value '{key}' ({value}) could not be converted to Decimal for validation."
                        )
            else:
                print(f"{NEON_RED}Config value '{key}' ({value}) has invalid type {type(value)}. Expected numeric.")

            if not valid:
                print(f"{NEON_YELLOW}Using default value for '{key}': {default_value}{RESET}")
                updated_config[key] = default_value
                save_needed = True

        # Validate core settings
        validate_numeric("retry_delay", min_val=0)
        validate_numeric("risk_per_trade", min_val=0, max_val=1)
        validate_numeric("leverage", min_val=1, is_int=True)
        validate_numeric("max_concurrent_positions", min_val=1, is_int=True)
        validate_numeric("signal_score_threshold", min_val=0)
        validate_numeric("stop_loss_multiple", min_val=0)
        validate_numeric("take_profit_multiple", min_val=0)
        validate_numeric("trailing_stop_callback_rate", min_val=1e-9)  # Must be > 0
        validate_numeric("trailing_stop_activation_percentage", min_val=0)
        validate_numeric("break_even_trigger_atr_multiple", min_val=0)
        validate_numeric("break_even_offset_ticks", min_val=0, is_int=True)
        validate_numeric("position_confirm_delay_seconds", min_val=0)
        validate_numeric("time_based_exit_minutes", min_val=1, allow_none=True)  # Allow None, but min 1 if set
        validate_numeric("orderbook_limit", min_val=1, is_int=True)
        validate_numeric("limit_order_offset_buy", min_val=0)
        validate_numeric("limit_order_offset_sell", min_val=0)
        validate_numeric("bollinger_bands_std_dev", min_val=0)  # Ensure std dev is non-negative

        # Validate indicator periods (ensure positive integers/floats where applicable)
        for key, default_val in DEFAULT_INDICATOR_PERIODS.items():
            is_int_param = isinstance(default_val, int) or key in ["stoch_rsi_k", "stoch_rsi_d"]  # K/D should be int
            min_value = 1 if is_int_param else 1e-9  # Periods usually > 0, AF can be small float
            validate_numeric(key, min_val=min_value, is_int=is_int_param)

        # Validate symbols_to_trade is a non-empty list of strings
        symbols = updated_config.get("symbols_to_trade")
        if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) for s in symbols):
            print(f"{NEON_RED}Invalid 'symbols_to_trade' format in config. Must be a non-empty list of strings.{RESET}")
            updated_config["symbols_to_trade"] = default_config["symbols_to_trade"]
            print(
                f"{NEON_YELLOW}Using default value for 'symbols_to_trade': {updated_config['symbols_to_trade']}{RESET}"
            )
            save_needed = True

        # Save corrected config if needed
        if save_needed:
            try:
                with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4)
                print(f"{NEON_YELLOW}Corrected invalid values and saved updated config file: {filepath}{RESET}")
            except IOError as e:
                print(f"{NEON_RED}Error writing corrected config file {filepath}: {e}{RESET}")

        return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
        try:
            # Attempt to recreate default if loading failed badly
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        except IOError as e_create:
            print(f"{NEON_RED}Error creating default config file after load error: {e_create}{RESET}")
        return default_config  # Return default


# File: logger_setup.py
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import pytz  # Import pytz or zoneinfo for timezone objects

# Import utility functions and classes
# Assume utils.py provides LOG_DIRECTORY, SensitiveFormatter (needs modification),
# and get_timezone() which returns a timezone object (like from pytz)
from utils import LOG_DIRECTORY, SensitiveFormatter, get_timezone


# --- Custom Formatter to Handle Timezone ---
# We need a custom formatter or modify SensitiveFormatter to format time with TZ
# without changing the global converter.
class TimezoneAwareFormatter(SensitiveFormatter):
    """
    A logging formatter that formats time including timezone info
    using the provided timezone object, without modifying global state.
    Inherits sensitive data masking capabilities.
    """

    def __init__(self, fmt=None, datefmt=None, style="%", timezone=None):
        super().__init__(fmt, datefmt, style)
        self.timezone = timezone or get_timezone()  # Use provided TZ or default from utils

    # Override formatTime to use timezone-aware datetime
    def formatTime(self, record, datefmt=None):
        """
        Return the creation time of the specified LogRecord as text.

        This implementation uses the timezone object provided during
        formatter initialization.
        """
        # record.created is a Unix timestamp (seconds since epoch in UTC)
        # Convert the UTC timestamp to the desired timezone-aware datetime
        dt_utc = datetime.utcfromtimestamp(record.created)
        dt_tz_aware = pytz.utc.localize(dt_utc).astimezone(self.timezone)  # Use pytz to make UTC tz-aware then convert

        if datefmt:
            # If a date format is provided, use it. Ensure it can handle timezone (%Z or %z)
            s = dt_tz_aware.strftime(datefmt)
        else:
            # Default format including milliseconds and timezone abbreviation
            # Example: '2023-10-27 10:30:00,123 CDT'
            s = dt_tz_aware.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]  # Format with milliseconds
            # Append timezone name/abbreviation if available and not included in %Z/%z via strftime
            # Note: %Z behaviour varies by OS/Python version. Using astimezone handles it correctly.
            # The %Z in strftime should now work as expected with a timezone-aware dt object.
            # Let's rely on %Z in the datefmt if needed. The default format above doesn't use %Z.
            # A common default format is 'YYYY-MM-DD HH:MM:SS,ms'
            # Let's ensure the formatter format string includes %Z or %z if TZ is desired in the output.
            # The base formatter `format` method calls `formatTime`, so the format string from init applies.
            # We just needed to provide a tz-aware datetime object here.

        return s


# --- Global Logging Configuration Function ---
def configure_logging(config: dict):
    """
    Configures the root logger based on application configuration.
    Sets up console and file handlers with appropriate levels and formatting.
    """
    # Get desired log level from config, default to INFO
    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Get the configured timezone object
    tz = get_timezone()

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Root logger should capture everything

    # Prevent duplicate handlers if configure_logging is somehow called multiple times
    # Clear existing handlers from root logger, which might include default ones
    if root_logger.hasHandlers():
        print("Clearing existing root logger handlers...", file=sys.stderr)
        for handler in root_logger.handlers[:]:
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                print(f"Warning: Error removing/closing root handler: {e}", file=sys.stderr)

    # Ensure log directory exists
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

    # --- Console Handler ---
    # Logs INFO level and above by default, using configured log level
    console_handler = logging.StreamHandler(sys.stdout)
    # Use the TimezoneAwareFormatter
    # Format string includes timezone abbreviation (%Z)
    console_formatter = TimezoneAwareFormatter(
        "%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",  # Include Timezone Abbreviation
        timezone=tz,
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)  # Set console level from config
    root_logger.addHandler(console_handler)
    # Also add a handler for stderr for WARNING/ERROR/CRITICAL
    # This ensures errors are visible even if stdout is redirected
    error_console_handler = logging.StreamHandler(sys.stderr)
    error_console_handler.setFormatter(console_formatter)  # Use same formatter
    error_console_handler.setLevel(logging.WARNING)  # Only show warnings/errors on stderr
    # Check if a similar stderr handler already exists to avoid duplicates
    stderr_exists = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stderr for h in root_logger.handlers)
    if not stderr_exists:
        root_logger.addHandler(error_console_handler)
    else:
        # If stderr handler exists, just ensure its level is appropriate
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                handler.setLevel(min(handler.level, logging.WARNING))  # Ensure it captures at least WARNING

    # --- Main File Handler ---
    # Logs DEBUG level and above to a main log file
    main_log_filename = os.path.join(LOG_DIRECTORY, "xrscalper_bot.log")
    try:
        file_handler = RotatingFileHandler(
            main_log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        # Use the TimezoneAwareFormatter for the file log as well
        file_formatter = TimezoneAwareFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S,%f",  # Include milliseconds in file log
            timezone=tz,
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        root_logger.addHandler(file_handler)
    except Exception as e:
        # Log this error to console before main file logger is fully set up
        # Use a basic print or the console handler directly if available
        print(f"CRITICAL ERROR: Could not set up main file logger {main_log_filename}: {e}", file=sys.stderr)
        # The bot might still run with just console logging

    # --- Optional: Configure levels for chatty libraries ---
    # Example: Suppress DEBUG/INFO messages from ccxt if not needed
    # logging.getLogger('ccxt').setLevel(logging.WARNING)
    # logging.getLogger('urllib3').setLevel(logging.WARNING)
    # logging.getLogger('asyncio').setLevel(logging.WARNING)

    # Log that configuration is complete
    logging.getLogger("xrscalper_bot_init").info(
        f"Logging configured successfully with level: {logging.getLevelName(log_level)}"
    )


# Note: The original setup_logger function is removed.
# In main.py, after calling configure_logging(CONFIG), you should get loggers
# for specific components or symbols using:
# init_logger = logging.getLogger("xrscalper_bot_init")
# symbol_logger = logging.getLogger("xrscalper_bot_BTC_USDT")
# etc.
# Messages logged by these loggers will propagate up to the root logger
# and be handled by the console and file handlers configured here.
# If symbol-specific *file* logs are required, a dedicated function
# could be added to this module to add a RotatingFileHandler to
# a specific named logger if it doesn't have one, perhaps called
# setup_symbol_file_logger(symbol_name, config). But typically,
# logging to one main file with logger names indicating the source is sufficient.

# Example of a potential function if symbol-specific files ARE needed:
# def setup_symbol_file_logger(symbol_name: str, config: dict):
#     """Sets up a file handler specifically for a symbol logger."""
#     safe_symbol_name = symbol_name.replace('/', '_').replace(':', '-')
#     logger_name = f"xrscalper_bot_{safe_symbol_name}"
#     log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
#     symbol_logger = logging.getLogger(logger_name)
#
#     # Ensure symbol logger's level is low enough to capture desired messages
#     # It might inherit from root, or you might set it explicitly
#     # symbol_logger.setLevel(logging.DEBUG) # Or inherit
#
#     # Check if a file handler for this path already exists to avoid duplicates
#     if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == os.path.abspath(log_filename) for h in symbol_logger.handlers):
#         try:
#             os.makedirs(LOG_DIRECTORY, exist_ok=True)
#             file_handler = RotatingFileHandler(
#                 log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
#             )
#             tz = get_timezone()
#             file_formatter = TimezoneAwareFormatter(
#                 "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
#                 datefmt='%Y-%m-%d %H:%M:%S,%f', # Include milliseconds
#                 timezone=tz
#             )
#             file_handler.setFormatter(file_formatter)
#             file_handler.setLevel(logging.DEBUG) # Log everything to the symbol file
#             symbol_logger.addHandler(file_handler)
#             symbol_logger.info(f"Symbol-specific file logging enabled for {symbol_name}")
#         except Exception as e:
#             symbol_logger.error(f"Failed to set up symbol file logger for {symbol_name}: {e}", exc_info=True)
#
#     # Important: Ensure propagation is False if you *only* want messages in the symbol file
#     # and not the main log file. If you want them in both, propagation should be True
#     # and the symbol logger level should be high enough to pass messages up.
#     # Given the structure of main.py, keeping propagation True and relying on the
#     # root logger handlers is simpler and messages will appear in both console
#     # and main file, with the logger name identifying the source.
#     # symbol_logger.propagate = False # Decide based on whether you want logs in main file too
# File: risk_manager.py
import logging
from decimal import Decimal, getcontext
from typing import Dict, Optional, Union


# Import utility functions


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: Union[float, Decimal, str],
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange,
    logger: Optional[logging.Logger] = None,
) -> Optional[Decimal]:
    """
    Calculates the position size in base currency or contracts.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")
    quote_currency = market_info.get("quote", "QUOTE")
    base_currency = market_info.get("base", "BASE")
    is_contract = market_info.get("is_contract", False)
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation and Conversion ---
    if not (isinstance(balance, Decimal) and balance > 0):
        lg.error(f"Position sizing failed ({symbol}): Invalid or non-positive balance ({balance}).")
        return None

    try:
        risk_value_decimal = Decimal(str(risk_per_trade))
        if not (Decimal(0) <= risk_value_decimal <= Decimal(1)):
            raise ValueError(f"risk_per_trade ({risk_value_decimal}) must be between 0 and 1 (e.g., 0.01 for 1%).")
    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"Position sizing failed ({symbol}): Invalid risk_per_trade ('{risk_per_trade}'). Error: {e}")
        return None

    if not (isinstance(initial_stop_loss_price, Decimal) and initial_stop_loss_price > 0):
        lg.error(f"Position sizing failed ({symbol}): Invalid initial_stop_loss_price ({initial_stop_loss_price}).")
        return None
    if not (isinstance(entry_price, Decimal) and entry_price > 0):
        lg.error(f"Position sizing failed ({symbol}): Invalid entry_price ({entry_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Position sizing failed ({symbol}): Stop loss price cannot be equal to entry price.")
        return None
    if "limits" not in market_info or "precision" not in market_info:
        lg.error(f"Position sizing failed ({symbol}): Market info missing 'limits' or 'precision'.")
        return None

    try:
        risk_amount_quote = balance * risk_value_decimal
        if risk_value_decimal > 0 and risk_amount_quote <= 0:
            lg.warning(
                f"Calculated risk amount {risk_amount_quote} {quote_currency} for {symbol} (risk {risk_value_decimal:.2%}) is non-positive. Balance: {balance:.4f}"
            )

        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit <= 0:
            lg.error(f"Position sizing failed ({symbol}): Stop loss distance non-positive ({sl_distance_per_unit}).")
            return None

        contract_size_str = market_info.get("contractSize", "1")
        try:
            contract_size = Decimal(str(contract_size_str))
            if contract_size <= 0:
                contract_size = Decimal("1")
                lg.warning(f"Contract size invalid for {symbol}, defaulted to 1.")
        except (InvalidOperation, ValueError, TypeError):
            contract_size = Decimal("1")
            lg.warning(f"Contract size parse error for {symbol}, defaulted to 1.")

        calculated_size: Optional[Decimal] = None
        if market_info.get("linear", True) or not is_contract:
            denominator = sl_distance_per_unit * contract_size
            if denominator > 0:
                calculated_size = risk_amount_quote / denominator
            else:
                lg.error(f"Pos sizing denom err ({symbol}): SLDist={sl_distance_per_unit}, ContrSize={contract_size}.")
                return None
        elif market_info.get("inverse", False):
            if entry_price == 0 or initial_stop_loss_price == 0:
                lg.error(f"Pos sizing err ({symbol}): Entry or SL price zero for inverse contract.")
                return None
            loss_per_contract_in_quote = contract_size * abs(
                Decimal("1") / entry_price - Decimal("1") / initial_stop_loss_price
            )
            if loss_per_contract_in_quote > 0:
                calculated_size = risk_amount_quote / loss_per_contract_in_quote
            else:
                lg.error(f"Pos sizing err ({symbol}): Loss per contract zero/neg for inverse.")
                return None
        else:
            lg.error(f"Unsupported market type for sizing {symbol}. Market: {market_info.get('type')}")
            return None

        if not (calculated_size and calculated_size > 0):
            lg.error(
                f"Initial size calc zero/neg: {calculated_size}. RiskAmt={risk_amount_quote:.4f}, SLDist={sl_distance_per_unit}, ContrSize={contract_size}"
            )
            return None

        # Use get_price_precision for formatting the log output
        price_precision_for_log = get_price_precision(market_info, lg)
        lg.info(
            f"Position Sizing ({symbol}): Balance={balance:.2f}, Risk={risk_value_decimal:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}"
        )
        lg.info(
            f"  Entry={entry_price:.{price_precision_for_log}f}, SL={initial_stop_loss_price:.{price_precision_for_log}f}, SL Dist Per Unit={sl_distance_per_unit:.{price_precision_for_log}f}"
        )
        lg.info(f"  ContractSize={contract_size}, Initial Calculated Size = {calculated_size:.8f} {size_unit}")

        limits = market_info.get("limits", {})
        amount_limits = limits.get("amount", {}) if isinstance(limits, dict) else {}
        cost_limits = limits.get("cost", {}) if isinstance(limits, dict) else {}

        min_amount_str = amount_limits.get("min")
        max_amount_str = amount_limits.get("max")
        min_cost_str = cost_limits.get("min")
        max_cost_str = cost_limits.get("max")

        min_amount = Decimal("0")
        if min_amount_str is not None and str(min_amount_str).strip():
            try:
                val = Decimal(str(min_amount_str))
                if val >= 0:
                    min_amount = val  # Allow 0 min_amount
            except InvalidOperation:
                lg.warning(f"Invalid min_amount_str '{min_amount_str}' for {symbol}. Using 0.")
                min_amount = Decimal("0")

        max_amount: Optional[Decimal] = None
        if max_amount_str is not None and str(max_amount_str).strip():
            try:
                val = Decimal(str(max_amount_str))
                if val > 0:
                    max_amount = val
            except InvalidOperation:
                lg.warning(f"Invalid max_amount_str '{max_amount_str}' for {symbol}. No upper amount limit.")

        min_cost = Decimal("0")
        if min_cost_str is not None and str(min_cost_str).strip():
            try:
                val = Decimal(str(min_cost_str))
                if val >= 0:
                    min_cost = val  # Allow 0 min_cost
            except InvalidOperation:
                lg.warning(f"Invalid min_cost_str '{min_cost_str}' for {symbol}. Using 0.")
                min_cost = Decimal("0")

        max_cost: Optional[Decimal] = None
        if max_cost_str is not None and str(max_cost_str).strip():
            try:
                val = Decimal(str(max_cost_str))
                if val > 0:
                    max_cost = val
            except InvalidOperation:
                lg.warning(f"Invalid max_cost_str '{max_cost_str}' for {symbol}. No upper cost limit.")

        adjusted_size = calculated_size
        if min_amount > 0 and adjusted_size < min_amount:
            lg.warning(
                f"Calculated size {calculated_size:.8f} < min amount {min_amount}. Adjusting to min for {symbol}."
            )
            adjusted_size = min_amount
        if max_amount and adjusted_size > max_amount:
            lg.warning(
                f"Calculated size {calculated_size:.8f} > max amount {max_amount}. Adjusting to max for {symbol}."
            )
            adjusted_size = max_amount

        estimated_cost: Optional[Decimal] = None
        cost_calc_price = entry_price
        if market_info.get("linear", True) or not is_contract:
            if cost_calc_price > 0 and contract_size > 0:
                estimated_cost = adjusted_size * cost_calc_price * contract_size
        elif market_info.get("inverse", False):
            if cost_calc_price > 0 and contract_size > 0:
                estimated_cost = adjusted_size * contract_size / cost_calc_price

        if estimated_cost is None:
            lg.error(f"Could not estimate cost for {symbol} (price or contract size invalid).")
            return None
        lg.debug(f"  Size after amount limits: {adjusted_size:.8f}. Est. Cost: {estimated_cost:.4f} {quote_currency}")

        if min_cost > 0 and estimated_cost < min_cost:
            lg.warning(f"Est. cost {estimated_cost:.4f} < min_cost {min_cost} for {symbol}. Trying to meet min_cost.")
            required_size_for_min_cost: Optional[Decimal] = None

            cost_per_unit_denom: Optional[Decimal] = None
            if market_info.get("linear", True) or not is_contract:
                cost_per_unit_denom = cost_calc_price * contract_size
            elif market_info.get("inverse", False) and cost_calc_price > 0:
                cost_per_unit_denom = (
                    contract_size / cost_calc_price
                )  # This is value of 1 contract in base. We need cost per unit.
                # For inverse: Cost = Size * ContractSize / Price => Size = Cost * Price / ContractSize
                # So, if we want to find size for min_cost:
                if contract_size > 0:
                    required_size_for_min_cost = (min_cost * cost_calc_price) / contract_size
                cost_per_unit_denom = None  # Flag that required_size_for_min_cost was calculated differently

            if cost_per_unit_denom and cost_per_unit_denom > 0:  # For linear/spot if not already calculated
                required_size_for_min_cost = min_cost / cost_per_unit_denom

            if not (required_size_for_min_cost and required_size_for_min_cost > 0):
                lg.error(f"Cannot calc required size for min_cost for {symbol} (denom/price invalid).")
                return None
            lg.info(f"  Required size for min_cost {min_cost}: {required_size_for_min_cost:.8f} {size_unit}")

            if max_amount and required_size_for_min_cost > max_amount:
                lg.error(f"Cannot meet min_cost for {symbol} without exceeding max_amount. Aborted.")
                return None
            if (
                min_amount > 0 and required_size_for_min_cost < min_amount
            ):  # Check against actual min_amount, not calculated_size
                lg.error(
                    f"Req. size for min_cost {required_size_for_min_cost} < min_amount {min_amount}. Limits conflict for {symbol}. Aborted."
                )
                return None
            adjusted_size = required_size_for_min_cost
            lg.info(f"  Adjusted size to meet min_cost: {adjusted_size:.8f} {size_unit}")
            # Recalculate estimated_cost for max_cost check
            if market_info.get("linear", True) or not is_contract:
                estimated_cost = adjusted_size * cost_calc_price * contract_size
            elif market_info.get("inverse", False) and cost_calc_price > 0:
                estimated_cost = adjusted_size * contract_size / cost_calc_price
            if estimated_cost is None:
                lg.error(f"Could not re-estimate cost for {symbol} after min_cost adj.")
                return None

        if max_cost and estimated_cost > max_cost:
            lg.warning(f"Est. cost {estimated_cost:.4f} > max_cost {max_cost} for {symbol}. Reducing size.")
            size_for_max_cost: Optional[Decimal] = None
            cost_per_unit_denom_mc: Optional[Decimal] = None
            if market_info.get("linear", True) or not is_contract:
                cost_per_unit_denom_mc = cost_calc_price * contract_size
                if cost_per_unit_denom_mc and cost_per_unit_denom_mc > 0:
                    size_for_max_cost = max_cost / cost_per_unit_denom_mc
            elif market_info.get("inverse", False) and cost_calc_price > 0 and contract_size > 0:
                size_for_max_cost = (max_cost * cost_calc_price) / contract_size

            if not (size_for_max_cost and size_for_max_cost > 0):
                lg.error(f"Cannot calc max size for max_cost for {symbol}.")
                return None
            lg.info(f"  Reduced size by max_cost: {size_for_max_cost:.8f} {size_unit}")

            if min_amount > 0 and size_for_max_cost < min_amount:
                lg.error(
                    f"Size reduced for max_cost {size_for_max_cost} < min_amount {min_amount}. Aborted for {symbol}."
                )
                return None
            adjusted_size = size_for_max_cost

        final_size_str: Optional[str] = None
        try:
            # CCXT's amount_to_precision expects a float or string number as its second argument.
            # Passing Decimal directly might work for some exchanges but float is safer for CCXT's internal handling.
            final_size_str = exchange.amount_to_precision(symbol, float(adjusted_size))
            final_size = Decimal(final_size_str)
            lg.info(f"Applied exchange amount precision: {adjusted_size:.8f} -> {final_size} {size_unit}")
        except Exception as e_ccxt_prec:
            lg.error(
                f"Error applying CCXT amount_to_precision for {symbol} on size {adjusted_size}: {e_ccxt_prec}. Attempting manual quantization."
            )
            amount_precision_places = market_info.get("amountPrecision", 8)
            quant_factor = Decimal("1e-" + str(amount_precision_places))
            final_size = adjusted_size.quantize(quant_factor, rounding=ROUND_DOWN)
            lg.info(f"Applied manual amount quantization: {adjusted_size:.8f} -> {final_size} {size_unit}")

        if not (final_size and final_size > 0):
            lg.error(
                f"Final position size zero or negative ({final_size}) after all adjustments for {symbol}. Aborted."
            )
            return None
        if min_amount > 0 and final_size < min_amount:
            lg.error(f"Final size {final_size} < min amount {min_amount} after precision for {symbol}. Aborted.")
            return None

        final_cost_est: Optional[Decimal] = None
        cost_calc_price_final = entry_price  # Use the same price for final cost check consistency
        if market_info.get("linear", True) or not is_contract:
            if cost_calc_price_final > 0 and contract_size > 0:
                final_cost_est = final_size * cost_calc_price_final * contract_size
        elif market_info.get("inverse", False):
            if cost_calc_price_final > 0 and contract_size > 0:
                final_cost_est = final_size * contract_size / cost_calc_price_final

        if final_cost_est is not None and min_cost > 0 and final_cost_est < min_cost:
            lg.error(
                f"Final size {final_size} results in cost {final_cost_est:.4f} < min_cost {min_cost} for {symbol}. Aborted."
            )
            return None

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except (InvalidOperation, ValueError, TypeError) as e:
        lg.error(f"Error during position size calculation ({symbol}) (Decimal/Type Error): {e}", exc_info=False)
    except Exception as e:
        lg.error(f"Unexpected error calculating position size for {symbol}: {e}", exc_info=True)
    return None


import logging

# import time # Retained for time.time() if loop.time() is not strictly epoch, or if preferred for epoch.
# asyncio.get_event_loop().time() is generally preferred in async code.
import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple
import sys

import ccxt
import pandas as pd

try:
    from analysis import TradingAnalyzer
    from exchange_api import (
        fetch_balance,
        fetch_current_price_ccxt,
        fetch_klines_ccxt,
        fetch_orderbook_ccxt,
        get_market_info,
        get_open_position,
        place_trade,
        set_leverage_ccxt,
        set_trailing_stop_loss,
        _set_position_protection,
    )  # _set_position_protection is for fixed SL/TP
    from risk_manager import calculate_position_size
    from utils import (
        CCXT_INTERVAL_MAP,
        POSITION_CONFIRM_DELAY_SECONDS,
        get_min_tick_size,
        get_price_precision,
        DEFAULT_INDICATOR_PERIODS,
        NEON_GREEN,
        NEON_PURPLE,
        NEON_RED,
        NEON_YELLOW,
        RESET,
        NEON_BLUE,
        NEON_CYAN,
    )
except ImportError as e:
    _NEON_RED = "\033[1;91m" if "NEON_RED" not in globals() else NEON_RED
    _RESET = "\033[0m" if "RESET" not in globals() else RESET
    print(
        f"{_NEON_RED}CRITICAL ERROR: Failed to import required modules in trading_strategy.py: {e}{_RESET}",
        file=sys.stderr,
    )
    if "traceback" not in sys.modules:
        import traceback  # Import if not already imported
    traceback.print_exc(file=sys.stderr)
    raise
except Exception as e:
    _NEON_RED = "\033[1;91m" if "NEON_RED" not in globals() else NEON_RED
    _RESET = "\033[0m" if "RESET" not in globals() else RESET
    print(
        f"{_NEON_RED}CRITICAL ERROR: An unexpected error occurred during module import in trading_strategy.py: {e}{_RESET}",
        file=sys.stderr,
    )
    if "traceback" not in sys.modules:
        import traceback
    traceback.print_exc(file=sys.stderr)
    raise


# --- Formatting Helpers ---
def _format_signal(signal_text: str) -> str:
    if signal_text == "BUY":
        return f"{NEON_GREEN}{signal_text}{RESET}"
    if signal_text == "SELL":
        return f"{NEON_RED}{signal_text}{RESET}"
    if signal_text == "HOLD":
        return f"{NEON_YELLOW}{signal_text}{RESET}"
    return signal_text


def _format_side(side_text: Optional[str]) -> str:
    if side_text is None:
        return f"{NEON_YELLOW}UNKNOWN{RESET}"
    side_upper = side_text.upper()
    if side_upper == "LONG":
        return f"{NEON_GREEN}{side_upper}{RESET}"
    if side_upper == "SHORT":
        return f"{NEON_RED}{side_upper}{RESET}"
    return side_upper


def _format_price_or_na(price_val: Optional[Decimal], precision_places: int, label: str = "") -> str:
    color = NEON_CYAN
    if price_val is not None and isinstance(price_val, Decimal):
        if price_val == Decimal(0) and label:  # Explicitly show 0.0 for certain fields if it's a valid Decimal(0)
            return f"{NEON_YELLOW}0.0{RESET}"  # Or format with precision: f"{NEON_YELLOW}{price_val:.{precision_places}f}{RESET}"
        if price_val > 0 or (price_val == Decimal(0) and not label):  # Format 0 if no label, or any positive
            try:
                return f"{color}{price_val:.{precision_places}f}{RESET}"
            except Exception as e:
                return f"{NEON_YELLOW}{price_val} (fmt err for {label}: {e}){RESET}"
        # Potentially handle negative values if they are expected and need formatting
        return f"{NEON_YELLOW}{price_val} (unexpected value for {label}){RESET}"
    return f"{NEON_YELLOW}N/A{RESET}"


# --- Core Trading Logic ---
async def _execute_close_position(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict[str, Any],
    open_position: Dict[str, Any],
    logger: logging.Logger,
    reason: str = "exit signal",
) -> bool:
    lg = logger
    pos_side = open_position.get("side")  # Expected 'long' or 'short'
    pos_size = open_position.get("contractsDecimal")  # Expected Decimal

    if not pos_side or pos_side not in ["long", "short"]:
        lg.error(f"{NEON_RED}Cannot close position for {symbol}: Invalid position side '{pos_side}'.{RESET}")
        return False
    if not isinstance(pos_size, Decimal) or pos_size <= 0:
        lg.warning(
            f"{NEON_YELLOW}Attempted to close {symbol} ({reason}), but size invalid/zero ({pos_size}). No close order.{RESET}"
        )
        return False  # Or True if this state means "effectively closed" or no action needed. False implies failure.

    try:
        close_side_signal = "SELL" if pos_side == "long" else "BUY"
        amount_precision = market_info.get("amountPrecision", 8)
        if not (isinstance(amount_precision, int) and amount_precision >= 0):
            amount_precision = 8

        lg.info(f"{NEON_YELLOW}==> Closing {_format_side(pos_side)} position for {symbol} due to {reason} <==")
        lg.info(
            f"{NEON_YELLOW}==> Placing {_format_signal(close_side_signal)} MARKET order (reduceOnly=True) | Size: {pos_size:.{amount_precision}f} <=="
        )

        close_order = await place_trade(
            exchange=exchange,
            symbol=symbol,
            trade_signal=close_side_signal,
            position_size=pos_size,
            market_info=market_info,
            logger=lg,
            order_type="market",
            reduce_only=True,
        )

        if close_order and close_order.get("id"):
            lg.info(f"{NEON_GREEN}Position CLOSE order placed for {symbol}. Order ID: {close_order['id']}{RESET}")
            # Consider adding a small delay and confirming closure if critical
            return True
        else:
            lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Placement returned None/no ID.{RESET}")
            lg.warning(f"{NEON_RED}Manual check/intervention required for {symbol}!{RESET}")
            return False
    except Exception as close_err:
        lg.error(
            f"{NEON_RED}Error attempting to close position for {symbol} ({reason}): {close_err}{RESET}", exc_info=True
        )
        lg.warning(f"{NEON_RED}Manual intervention may be needed for {symbol}!{RESET}")
        return False


async def _fetch_and_prepare_market_data(
    exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], lg: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Fetches market info, klines, current price, and orderbook."""
    market_info = await get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"{NEON_RED}Failed to get market info for {symbol}. Skipping cycle.{RESET}")
        return None

    interval_config_val = config.get("interval")
    if interval_config_val is None:
        lg.error(f"{NEON_RED}Interval not specified for {symbol}. Skipping.{RESET}")
        return None
    interval_str = str(interval_config_val)
    ccxt_interval = CCXT_INTERVAL_MAP.get(interval_str)
    if not ccxt_interval:
        lg.error(f"{NEON_RED}Invalid interval '{interval_str}' for {symbol}. Skipping.{RESET}")
        return None

    kline_limit = config.get("kline_limit", 500)
    klines_df = await fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)
    min_kline_length = config.get("min_kline_length", 50)
    if klines_df is None or klines_df.empty or len(klines_df) < min_kline_length:
        lg.error(
            f"{NEON_RED}Insufficient kline data for {symbol} (got {len(klines_df) if klines_df is not None else 0}, need {min_kline_length}). Skipping.{RESET}"
        )
        return None

    current_price_decimal: Optional[Decimal] = None
    try:
        current_price_fetch = await fetch_current_price_ccxt(exchange, symbol, lg)
        if current_price_fetch and isinstance(current_price_fetch, Decimal) and current_price_fetch > 0:
            current_price_decimal = current_price_fetch
        else:
            lg.warning(
                f"{NEON_YELLOW}Failed ticker price fetch or invalid price ({current_price_fetch}) for {symbol}. Using last kline close.{RESET}"
            )
            if not klines_df.empty and "close" in klines_df.columns:
                last_close_val = klines_df["close"].iloc[-1]
                if isinstance(last_close_val, Decimal) and pd.notna(last_close_val) and last_close_val > 0:
                    current_price_decimal = last_close_val
                elif pd.notna(last_close_val):
                    try:
                        current_price_decimal = Decimal(str(last_close_val))
                        assert current_price_decimal > 0
                    except:
                        lg.error(f"{NEON_RED}Last kline close value '{last_close_val}' invalid for {symbol}.{RESET}")
                else:
                    lg.error(f"{NEON_RED}Last kline close value is NaN for {symbol}.{RESET}")
            else:
                lg.error(
                    f"{NEON_RED}Cannot use last kline close: DataFrame empty or 'close' missing for {symbol}.{RESET}"
                )
    except Exception as e:
        lg.error(f"{NEON_RED}Error fetching/processing current price for {symbol}: {e}{RESET}", exc_info=True)

    if not (current_price_decimal and isinstance(current_price_decimal, Decimal) and current_price_decimal > 0):
        lg.error(f"{NEON_RED}Cannot get valid current price for {symbol} ({current_price_decimal}). Skipping.{RESET}")
        return None
    lg.debug(f"Current price for {symbol}: {current_price_decimal}")

    orderbook_data = None
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    if config.get("indicators", {}).get("orderbook", False) and Decimal(str(active_weights.get("orderbook", "0"))) != 0:
        orderbook_data = await fetch_orderbook_ccxt(exchange, symbol, config.get("orderbook_limit", 100), lg)
        if not orderbook_data:
            lg.warning(f"{NEON_YELLOW}Failed to fetch orderbook for {symbol}, proceeding without.{RESET}")

    return {
        "market_info": market_info,
        "klines_df": klines_df,
        "current_price_decimal": current_price_decimal,
        "orderbook_data": orderbook_data,
        "price_precision": get_price_precision(market_info, lg),
        "min_tick_size": get_min_tick_size(market_info, lg),
        "amount_precision": market_info.get("amountPrecision", 8),
        "min_qty": Decimal(
            str(market_info.get("limits", {}).get("amount", {}).get("min", "0"))
        ),  # Simplified, add robust parsing if needed
    }


def _perform_trade_analysis(
    klines_df: pd.DataFrame,
    current_price_decimal: Decimal,
    orderbook_data: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    market_info: Dict[str, Any],
    lg: logging.Logger,
    price_precision: int,
) -> Optional[Dict[str, Any]]:
    """Performs trading analysis and generates signals."""
    analyzer = TradingAnalyzer(klines_df.copy(), lg, config, market_info)
    if not analyzer.indicator_values:
        lg.error(f"{NEON_RED}Indicator calculation failed for {symbol}. Skipping signal generation.{RESET}")
        return None  # symbol undefined here, use market_info['symbol']

    signal = analyzer.generate_trading_signal(current_price_decimal, orderbook_data)
    _, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price_decimal, signal)
    current_atr: Optional[Decimal] = analyzer.indicator_values.get("ATR")

    lg.info(f"--- {NEON_PURPLE}Analysis Summary ({market_info['symbol']}){RESET} ---")  # Use market_info['symbol']
    lg.info(f"  Current Price: {NEON_CYAN}{current_price_decimal:.{price_precision}f}{RESET}")
    analyzer_atr_period = analyzer.config.get("atr_period", DEFAULT_INDICATOR_PERIODS.get("atr_period", 14))
    atr_log_str = f"  ATR ({analyzer_atr_period}): {NEON_YELLOW}N/A{RESET}"
    if isinstance(current_atr, Decimal) and pd.notna(current_atr) and current_atr > 0:
        try:
            atr_log_str = (
                f"  ATR ({analyzer_atr_period}): {NEON_CYAN}{current_atr:.{max(0, price_precision + 2)}f}{RESET}"
            )
        except Exception as fmt_e:
            atr_log_str = f"  ATR ({analyzer_atr_period}): {NEON_YELLOW}{current_atr} (fmt err: {fmt_e}){RESET}"
    elif current_atr is not None:
        atr_log_str = f"  ATR ({analyzer_atr_period}): {NEON_YELLOW}{current_atr} (invalid/zero){RESET}"
    lg.info(atr_log_str)
    lg.info(f"  Initial SL (sizing): {_format_price_or_na(sl_calc, price_precision, 'SL Calc')}")
    lg.info(f"  Initial TP (target): {_format_price_or_na(tp_calc, price_precision, 'TP Calc')}")

    tsl_enabled_config = config.get("enable_trailing_stop", False)
    be_enabled_config = config.get("enable_break_even", False)
    time_exit_minutes_config = config.get("time_based_exit_minutes")
    tsl_conf_str = f"{NEON_GREEN}Enabled{RESET}" if tsl_enabled_config else f"{NEON_RED}Disabled{RESET}"
    be_conf_str = f"{NEON_GREEN}Enabled{RESET}" if be_enabled_config else f"{NEON_RED}Disabled{RESET}"
    time_exit_log_str = (
        f"{time_exit_minutes_config} min"
        if time_exit_minutes_config
        and isinstance(time_exit_minutes_config, (int, float))
        and time_exit_minutes_config > 0
        else "Disabled"
    )
    lg.info(f"  Config: TSL={tsl_conf_str}, BE={be_conf_str}, TimeExit={time_exit_log_str}")
    lg.info(f"  Generated Signal: {_format_signal(signal)}")
    lg.info("-----------------------------")

    return {
        "signal": signal,
        "tp_calc": tp_calc,
        "sl_calc": sl_calc,
        "analyzer": analyzer,
        "tsl_enabled_config": tsl_enabled_config,
        "be_enabled_config": be_enabled_config,
        "time_exit_minutes_config": time_exit_minutes_config,
        "current_atr": current_atr,
    }


async def _handle_no_open_position(
    exchange: ccxt.Exchange,
    symbol: str,
    config: Dict[str, Any],
    lg: logging.Logger,
    market_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
):
    """Handles logic when there is no open position."""
    signal = analysis_results["signal"]
    if signal not in ["BUY", "SELL"]:
        lg.info(f"Signal is {_format_signal(signal)} and no open position for {symbol}. No entry action.")
        return

    lg.info(f"{NEON_PURPLE}*** {_format_signal(signal)} Signal & No Position: Initiating Trade for {symbol} ***{RESET}")

    balance = await fetch_balance(exchange, config.get("quote_currency", "USDT"), lg)
    if not (balance and isinstance(balance, Decimal) and balance > 0):
        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid balance ({balance}).{RESET}")
        return

    try:
        risk_pct = Decimal(str(config.get("risk_per_trade", 0.0)))
        assert Decimal(0) <= risk_pct <= Decimal(1)
    except:
        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid risk_per_trade config.{RESET}")
        return

    risk_amt = balance * risk_pct
    sl_calc = analysis_results["sl_calc"]
    if not (sl_calc and isinstance(sl_calc, Decimal) and sl_calc > 0):
        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Initial SL invalid ({sl_calc}) for sizing.{RESET}")
        return
    if risk_amt <= 0 and risk_pct > 0:
        lg.warning(
            f"{NEON_YELLOW}Trade Aborted ({symbol} {signal}): Risk amount non-positive ({risk_amt}). Balance: {balance}, Risk %: {risk_pct}.{RESET}"
        )
        return

    market_info = market_data["market_info"]
    if not market_info.get("spot", True):
        lev = int(config.get("leverage", 1))
        if lev > 0:
            if not await set_leverage_ccxt(exchange, symbol, lev, market_info, lg):
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set leverage {lev}x.{RESET}")
                return

    current_price_decimal = market_data["current_price_decimal"]
    pos_size_dec = calculate_position_size(balance, risk_pct, sl_calc, current_price_decimal, market_info, exchange, lg)
    if not (pos_size_dec and isinstance(pos_size_dec, Decimal) and pos_size_dec > 0):
        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Position size calc invalid ({pos_size_dec}).{RESET}")
        return

    amount_precision = market_data["amount_precision"]
    try:
        quant_factor = Decimal(f"1e-{amount_precision}")
        pos_size_dec = pos_size_dec.quantize(quant_factor, ROUND_DOWN)
        lg.debug(f"Quantized position size for {symbol}: {pos_size_dec:.{amount_precision}f}")
    except Exception as e:
        lg.error(f"{NEON_RED}Error quantizing size {pos_size_dec} for {symbol}: {e}{RESET}", exc_info=True)
        return

    min_qty = market_data["min_qty"]
    if min_qty > 0 and pos_size_dec < min_qty:
        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Size {pos_size_dec} < MinQty {min_qty}.{RESET}")
        return

    entry_type = config.get("entry_order_type", "market").lower()
    limit_px: Optional[Decimal] = None
    min_tick_size = market_data["min_tick_size"]
    price_precision = market_data["price_precision"]

    if entry_type == "limit":
        if not (min_tick_size and isinstance(min_tick_size, Decimal) and min_tick_size > 0):
            lg.warning(f"{NEON_YELLOW}Min tick invalid for limit order {symbol}. Switching to Market.{RESET}")
            entry_type = "market"
        else:
            try:
                offset_key = f"limit_order_offset_{signal.lower()}"
                offset_pct = Decimal(str(config.get(offset_key, "0.0005")))
                if offset_pct < 0:
                    raise ValueError("Limit order offset percentage cannot be negative.")

                raw_px = current_price_decimal * (
                    Decimal(1) - offset_pct if signal == "BUY" else Decimal(1) + offset_pct
                )
                limit_px = (raw_px / min_tick_size).quantize(
                    Decimal("1"), ROUND_DOWN if signal == "BUY" else ROUND_UP
                ) * min_tick_size

                if not (limit_px and limit_px > 0):
                    raise ValueError(f"Limit price calc invalid: {limit_px}")
                lg.info(
                    f"Calculated Limit Entry for {signal} on {symbol}: {_format_price_or_na(limit_px, price_precision, 'Limit Px')}"
                )
            except Exception as e_lim:
                lg.error(
                    f"{NEON_RED}Error calc limit price for {symbol}: {e_lim}. Switching to Market.{RESET}",
                    exc_info=False,
                )
                entry_type = "market"
                limit_px = None

    limit_px_log_label = f"Limit Px {signal}"
    lg.info(
        f"{NEON_YELLOW}==> Placing {_format_signal(signal)} {entry_type.upper()} | Size: {pos_size_dec:.{amount_precision}f}"
        f"{f' @ {_format_price_or_na(limit_px, price_precision, limit_px_log_label)}' if limit_px else ''} <=="
    )
    trade_order = await place_trade(
        exchange, symbol, signal, pos_size_dec, market_info, lg, entry_type, limit_px, False
    )

    if not (trade_order and trade_order.get("id")):
        lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {_format_signal(signal)}). No order ID. ==={RESET}")
        return

    order_id, order_status = trade_order["id"], trade_order.get("status", "unknown")
    lg.info(f"{NEON_GREEN}Order placed for {symbol}: ID={order_id}, Status={order_status}{RESET}")

    if entry_type == "market" or (entry_type == "limit" and order_status == "closed"):
        confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
        lg.info(f"Waiting {confirm_delay}s for position confirmation ({symbol})...")
        await asyncio.sleep(confirm_delay)
        confirmed_pos = await get_open_position(exchange, symbol, market_info, lg)

        if not confirmed_pos:
            lg.error(
                f"{NEON_RED}Order {order_id} ({entry_type}) for {symbol} reported filled/placed, but FAILED TO CONFIRM open position! Manual check!{RESET}"
            )
            return

        lg.info(f"{NEON_GREEN}Position Confirmed for {symbol} after {entry_type.capitalize()} Order!{RESET}")
        try:
            entry_px_actual = confirmed_pos.get("entryPriceDecimal")
            if not (entry_px_actual and isinstance(entry_px_actual, Decimal) and entry_px_actual > 0):
                fallback_price_info = (
                    f"limit price {_format_price_or_na(limit_px, price_precision)}"
                    if entry_type == "limit" and limit_px and limit_px > 0
                    else f"initial estimate {current_price_decimal:.{price_precision}f}"
                )
                lg.warning(
                    f"Could not get valid actual entry price for {symbol}. Using {fallback_price_info} for protection."
                )
                entry_px_actual = (
                    limit_px if entry_type == "limit" and limit_px and limit_px > 0 else current_price_decimal
                )

            if not (entry_px_actual and isinstance(entry_px_actual, Decimal) and entry_px_actual > 0):
                raise ValueError("Cannot determine valid entry price for protection setup.")

            lg.info(
                f"Using Entry Price for Protection ({symbol}): {NEON_CYAN}{entry_px_actual:.{price_precision}f}{RESET}"
            )
            analyzer: TradingAnalyzer = analysis_results["analyzer"]
            _, tp_f, sl_f = analyzer.calculate_entry_tp_sl(entry_px_actual, signal)
            tp_f_s = _format_price_or_na(tp_f, price_precision, "Final TP")
            sl_f_s = _format_price_or_na(sl_f, price_precision, "Final SL")

            prot_ok = False
            tsl_enabled_config = analysis_results["tsl_enabled_config"]
            if tsl_enabled_config:
                lg.info(f"Setting TSL for {symbol} (TP target: {tp_f_s})...")
                prot_ok = await set_trailing_stop_loss(exchange, symbol, market_info, confirmed_pos, config, lg, tp_f)
            elif (sl_f and isinstance(sl_f, Decimal) and sl_f > 0) or (tp_f and isinstance(tp_f, Decimal) and tp_f > 0):
                lg.info(f"Setting Fixed SL ({sl_f_s}) and TP ({tp_f_s}) for {symbol}...")
                prot_ok = await _set_position_protection(exchange, symbol, market_info, confirmed_pos, lg, sl_f, tp_f)
            else:
                lg.debug(f"No valid SL/TP for fixed protection for {symbol}, and TSL disabled.")
                prot_ok = True

            if prot_ok:
                lg.info(
                    f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({symbol} {_format_signal(signal)}) ==={RESET}"
                )
            else:
                lg.error(
                    f"{NEON_RED}=== TRADE ({symbol} {_format_signal(signal)}) placed BUT FAILED TO SET PROTECTION ==={RESET}"
                )
                lg.warning(f"{NEON_RED}>>> MANUAL MONITORING REQUIRED! <<<")
        except Exception as post_err:
            lg.error(f"{NEON_RED}Error post-trade protection ({symbol}): {post_err}{RESET}", exc_info=True)
            lg.warning(f"{NEON_RED}Position open but protection failed. Manual check!{RESET}")
    elif entry_type == "limit" and order_status == "open":
        lg.info(
            f"{NEON_YELLOW}Limit order {order_id} for {symbol} OPEN @ {_format_price_or_na(limit_px, price_precision, f'Limit Px {signal}')}. Waiting for fill.{RESET}"
        )
    else:
        lg.error(
            f"{NEON_RED}Limit order {order_id} for {symbol} status: {order_status}. Trade not open as expected.{RESET}"
        )


async def _manage_existing_open_position(
    exchange: ccxt.Exchange,
    symbol: str,
    config: Dict[str, Any],
    lg: logging.Logger,
    market_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
    open_position: Dict[str, Any],
    loop: asyncio.AbstractEventLoop,
):
    """Manages an existing open position."""
    pos_side = open_position.get("side")
    pos_size_dec = open_position.get("contractsDecimal")
    entry_px_dec = open_position.get("entryPriceDecimal")
    pos_ts_ms = open_position.get("timestamp_ms")  # Assumed to be epoch ms

    market_info = market_data["market_info"]
    price_precision = market_data["price_precision"]
    amount_precision = market_data["amount_precision"]

    if not (
        pos_side
        and isinstance(pos_side, str)
        and pos_side in ["long", "short"]
        and isinstance(pos_size_dec, Decimal)
        and pos_size_dec > 0
        and isinstance(entry_px_dec, Decimal)
        and entry_px_dec > 0
    ):
        lg.error(
            f"{NEON_RED}Cannot manage {symbol}: Invalid pos details. Side='{pos_side}',Size='{pos_size_dec}',Entry='{entry_px_dec}'.{RESET}"
        )
        return

    quote_prec = config.get("quote_currency_precision", market_info.get("quotePrecision", 2))
    lg.info(f"{NEON_BLUE}--- Managing Position ({NEON_PURPLE}{symbol}{NEON_BLUE}) ---{RESET}")
    lg.info(
        f"  Side: {_format_side(pos_side)}, Size: {NEON_CYAN}{pos_size_dec:.{amount_precision}f}{RESET}, Entry: {_format_price_or_na(entry_px_dec, price_precision, 'Entry Px')}"
    )
    lg.info(
        f"  MarkPx: {_format_price_or_na(open_position.get('markPriceDecimal'), price_precision, 'Mark Px')}, LiqPx: {_format_price_or_na(open_position.get('liquidationPriceDecimal'), price_precision, 'Liq Px')}"
    )
    lg.info(
        f"  uPnL: {_format_price_or_na(open_position.get('unrealizedPnlDecimal'), quote_prec, 'uPnL')} {market_info.get('quote', 'USD')}"
    )
    lg.info(
        f"  Exchange SL: {_format_price_or_na(open_position.get('stopLossPriceDecimal'), price_precision, 'Exch SL')}, TP: {_format_price_or_na(open_position.get('takeProfitPriceDecimal'), price_precision, 'Exch TP')}"
    )
    lg.info(
        f"  TSL Active Val: {_format_price_or_na(open_position.get('trailingStopLossValue'), price_precision if open_position.get('trailingStopActivationPrice') else 2, 'TSL Val')}"
    )

    signal = analysis_results["signal"]
    if (pos_side == "long" and signal == "SELL") or (pos_side == "short" and signal == "BUY"):
        lg.warning(
            f"{NEON_YELLOW}*** EXIT Signal ({_format_signal(signal)}) opposes {_format_side(pos_side)} position for {symbol}. Closing... ***{RESET}"
        )
        await _execute_close_position(exchange, symbol, market_info, open_position, lg, "opposing signal")
        return

    lg.info(f"Signal ({_format_signal(signal)}) allows holding. Position management for {symbol}...")

    time_exit_minutes_config = analysis_results["time_exit_minutes_config"]
    if isinstance(time_exit_minutes_config, (int, float)) and time_exit_minutes_config > 0:
        if pos_ts_ms and isinstance(pos_ts_ms, int):
            try:
                # Using loop.time() if pos_ts_ms is also based on a monotonic clock,
                # or time.time() if pos_ts_ms is strictly epoch. Assuming loop.time() is appropriate here.
                current_time_ms = int(loop.time() * 1000)
                elapsed_min = (current_time_ms - pos_ts_ms) / 60000.0
                lg.debug(f"Time Exit Check ({symbol}): Elapsed={elapsed_min:.2f}m, Limit={time_exit_minutes_config}m")
                if elapsed_min >= time_exit_minutes_config:
                    lg.warning(
                        f"{NEON_YELLOW}*** TIME-BASED EXIT for {symbol} ({elapsed_min:.1f} >= {time_exit_minutes_config}m). Closing... ***{RESET}"
                    )
                    await _execute_close_position(exchange, symbol, market_info, open_position, lg, "time-based exit")
                    return
            except Exception as terr:
                lg.error(f"{NEON_RED}Time exit check error for {symbol}: {terr}{RESET}", exc_info=True)
        else:
            lg.warning(
                f"{NEON_YELLOW}Time exit enabled for {symbol} but position timestamp invalid/missing ({pos_ts_ms}).{RESET}"
            )

    is_tsl_exch_active = False
    tsl_val_raw = open_position.get("trailingStopLossValue") or open_position.get("info", {}).get("trailingStopValue")
    if tsl_val_raw and str(tsl_val_raw).strip() and str(tsl_val_raw) != "0":
        try:
            if Decimal(str(tsl_val_raw)) > 0:
                is_tsl_exch_active = True
                lg.debug(f"TSL appears active on exchange for {symbol}.")
        except:
            pass

    be_enabled_config = analysis_results["be_enabled_config"]
    current_price_decimal = market_data["current_price_decimal"]
    min_tick_size = market_data["min_tick_size"]
    current_atr = analysis_results["current_atr"]

    if be_enabled_config and not is_tsl_exch_active:
        lg.info(f"{NEON_BLUE}--- Break-Even Check ({NEON_PURPLE}{symbol}{NEON_BLUE}) ---{RESET}")
        try:
            if not (isinstance(current_atr, Decimal) and current_atr > 0):
                lg.warning(f"{NEON_YELLOW}BE check skipped for {symbol}: Current ATR invalid ({current_atr}).{RESET}")
            else:
                be_trig_atr = Decimal(str(config.get("break_even_trigger_atr_multiple", "1.0")))
                be_off_ticks = int(config.get("break_even_offset_ticks", 2))
                px_diff = (
                    (current_price_decimal - entry_px_dec)
                    if pos_side == "long"
                    else (entry_px_dec - current_price_decimal)
                )
                profit_atr = (
                    px_diff / current_atr if current_atr > 0 else Decimal("inf")
                )  # Avoid division by zero if ATR is somehow zero
                lg.info(
                    f"  BE Status ({symbol}): PxDiff={_format_price_or_na(px_diff, price_precision + 1, 'PxDiff')}, ProfitATRs={profit_atr:.2f}, TargetATRs={be_trig_atr:.2f}"
                )

                if profit_atr >= be_trig_atr:
                    lg.info(f"  {NEON_GREEN}BE Trigger Met for {symbol}! Calculating BE stop.{RESET}")
                    if not (min_tick_size and isinstance(min_tick_size, Decimal) and min_tick_size > 0):
                        lg.warning(f"  {NEON_YELLOW}Cannot calc BE offset for {symbol}: Min tick invalid.{RESET}")
                    else:
                        tick_off = min_tick_size * Decimal(be_off_ticks)
                        raw_be_px = entry_px_dec + tick_off if pos_side == "long" else entry_px_dec - tick_off
                        rnd_mode = (
                            ROUND_UP if pos_side == "long" else ROUND_DOWN
                        )  # Ensure BE stop is slightly in profit or at entry
                        be_px = (raw_be_px / min_tick_size).quantize(Decimal("1"), rnd_mode) * min_tick_size

                        if not (be_px and be_px > 0):
                            lg.error(f"  {NEON_RED}Calc BE stop invalid for {symbol}: {be_px}.{RESET}")
                        else:
                            lg.info(
                                f"  Target BE Stop Price for {symbol}: {NEON_CYAN}{be_px:.{price_precision}f}{RESET}"
                            )
                            cur_sl_dec = open_position.get("stopLossPriceDecimal")
                            upd_be_sl = False
                            if not (cur_sl_dec and isinstance(cur_sl_dec, Decimal) and cur_sl_dec > 0):
                                upd_be_sl = True
                                lg.info(f"  BE ({symbol}): No valid current SL. Setting BE SL.")
                            elif (pos_side == "long" and be_px > cur_sl_dec) or (
                                pos_side == "short" and be_px < cur_sl_dec
                            ):
                                upd_be_sl = True
                                lg.info(
                                    f"  BE ({symbol}): Target {_format_price_or_na(be_px, price_precision, 'BE Px')} better than Current {_format_price_or_na(cur_sl_dec, price_precision, 'Cur SL')}. Updating."
                                )
                            else:
                                lg.debug(
                                    f"  BE ({symbol}): Current SL {_format_price_or_na(cur_sl_dec, price_precision, 'Cur SL')} already better/equal."
                                )

                            if upd_be_sl:
                                lg.warning(
                                    f"{NEON_YELLOW}*** Moving SL to Break-Even for {symbol} at {be_px:.{price_precision}f} ***{RESET}"
                                )
                                cur_tp_dec = open_position.get("takeProfitPriceDecimal")  # Preserve existing TP if any
                                if await _set_position_protection(
                                    exchange, symbol, market_info, open_position, lg, be_px, cur_tp_dec
                                ):
                                    lg.info(f"{NEON_GREEN}BE SL set/updated successfully for {symbol}.{RESET}")
                                else:
                                    lg.error(f"{NEON_RED}Failed to set/update BE SL for {symbol}. Manual check!{RESET}")
                else:
                    lg.info(f"  BE Profit target not reached for {symbol} ({profit_atr:.2f} < {be_trig_atr:.2f} ATRs).")
        except Exception as be_e:
            lg.error(f"{NEON_RED}Error in BE check ({symbol}): {be_e}{RESET}", exc_info=True)
    elif is_tsl_exch_active:
        lg.debug(f"BE check skipped for {symbol}: TSL active on exchange.")
    else:
        lg.debug(f"BE check skipped for {symbol}: BE disabled in config.")

    tsl_enabled_config = analysis_results["tsl_enabled_config"]
    analyzer: TradingAnalyzer = analysis_results["analyzer"]
    if tsl_enabled_config and not is_tsl_exch_active:
        lg.info(f"{NEON_BLUE}--- Trailing Stop Loss Check ({NEON_PURPLE}{symbol}{NEON_BLUE}) ---{RESET}")
        lg.info("  Attempting to set/update TSL (enabled & not active on exch).")
        # For TSL, TP target is usually based on initial entry conditions or dynamic.
        # Here, using entry_px_dec and current pos_side to determine a relevant TP target for TSL.
        _, tsl_tp_target, _ = analyzer.calculate_entry_tp_sl(entry_px_dec, pos_side)  # pos_side is 'long' or 'short'
        if await set_trailing_stop_loss(exchange, symbol, market_info, open_position, config, lg, tsl_tp_target):
            lg.info(f"  {NEON_GREEN}TSL setup/update initiated successfully for {symbol}.{RESET}")
        else:
            lg.warning(f"  {NEON_YELLOW}Failed to initiate TSL setup/update for {symbol}.{RESET}")
    elif tsl_enabled_config and is_tsl_exch_active:
        lg.debug(f"TSL enabled but already appears active on exchange for {symbol}. No TSL action.")

    lg.info(f"{NEON_CYAN}------------------------------------{RESET}")


async def analyze_and_trade_symbol(
    exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger, enable_trading: bool
) -> None:
    lg = logger
    loop = asyncio.get_event_loop()
    cycle_start_time = loop.time()
    lg.info(
        f"{NEON_BLUE}---== Analyzing {NEON_PURPLE}{symbol}{NEON_BLUE} (Interval: {config.get('interval', 'N/A')}) Cycle Start ==---{RESET}"
    )

    market_data = await _fetch_and_prepare_market_data(exchange, symbol, config, lg)
    if not market_data:
        lg.info(
            f"{NEON_BLUE}---== Analysis Cycle End ({NEON_PURPLE}{symbol}{NEON_BLUE}, {loop.time() - cycle_start_time:.2f}s, Data Fetch Failed) ==---{RESET}\n"
        )
        return

    # Symbol is available in market_data['market_info']['symbol'] if needed by _perform_trade_analysis
    # Pass symbol explicitly if preferred over digging into market_info within the function.
    # For now, _perform_trade_analysis uses market_info['symbol'] for logging.
    analysis_results = _perform_trade_analysis(
        market_data["klines_df"],
        market_data["current_price_decimal"],
        market_data["orderbook_data"],
        config,
        market_data["market_info"],
        lg,
        market_data["price_precision"],
    )
    if not analysis_results:
        lg.info(
            f"{NEON_BLUE}---== Analysis Cycle End ({NEON_PURPLE}{symbol}{NEON_BLUE}, {loop.time() - cycle_start_time:.2f}s, Analysis Failed) ==---{RESET}\n"
        )
        return

    if not enable_trading:
        lg.debug(f"Trading disabled. Analysis complete for {symbol}.")
        lg.info(
            f"{NEON_BLUE}---== Analysis Cycle End ({NEON_PURPLE}{symbol}{NEON_BLUE}, {loop.time() - cycle_start_time:.2f}s, Trading Disabled) ==---{RESET}\n"
        )
        return

    open_position = await get_open_position(exchange, symbol, market_data["market_info"], lg)

    if open_position is None:
        await _handle_no_open_position(exchange, symbol, config, lg, market_data, analysis_results)
    else:
        await _manage_existing_open_position(
            exchange, symbol, config, lg, market_data, analysis_results, open_position, loop
        )

    lg.info(
        f"{NEON_BLUE}---== Analysis Cycle End ({NEON_PURPLE}{symbol}{NEON_BLUE}, {loop.time() - cycle_start_time:.2f}s) ==---{RESET}\n"
    )  # File: analysis.py


"""
Module for analyzing trading data, calculating technical indicators, and generating trading signals.

The TradingAnalyzer class takes historical OHLCV data, configuration, and market information
to compute various technical indicators. It then uses these indicators, along with configurable
weights, to generate BUY, SELL, or HOLD signals. It can also calculate Fibonacci levels and
suggest Take Profit/Stop Loss levels based on ATR.
"""

import logging
from decimal import Decimal
from typing import Any, Dict, Optional  # Added List for type hint

import numpy as np
import pandas as pd

from utils import (
    CCXT_INTERVAL_MAP,
    DEFAULT_INDICATOR_PERIODS,  # Ensure this includes psar_initial_af, psar_af_step, psar_max_af
    FIB_LEVELS,
    get_min_tick_size,
    get_price_precision,
    NEON_RED,
    NEON_YELLOW,
    NEON_GREEN,
    RESET,
    NEON_PURPLE,
    NEON_BLUE,
    NEON_CYAN,
    _format_signal,  # Assuming _format_signal is in utils for logging
)


class TradingAnalyzer:
    """
    Analyzes trading data, calculates technical indicators, and generates trading signals.
    It uses pandas-ta for most indicator calculations and allows for weighted scoring
    of these indicators to produce a final trading decision.
    """

    # Configuration for technical indicators.
    # - func_name: Name of the function in pandas_ta library or a custom method in this class.
    # - params_map: Maps internal configuration keys (value) to the TA function's parameter names (key).
    # - main_col_pattern: Pattern for the main output column name if the TA function returns a single Series
    #                     or if concat=False. Parameters are filled from params_map values.
    # - multi_cols: For TA functions returning a DataFrame. Maps internal friendly names (key) to
    #                 column name patterns (value) for specific columns from the TA result.
    # - type: Expected data type ("decimal" or "float") for storing the latest value of this indicator.
    # - pass_close_only: If True, only the 'close' price Series is passed to the TA function.
    # - min_data_param_key: The key in 'params_map' whose corresponding period value determines the
    #                       minimum data rows needed for the indicator.
    # - min_data: A fixed integer for minimum data rows if min_data_param_key is not applicable.
    # - concat: If True, the result of the TA function (Series or DataFrame) is concatenated to the
    #           main calculation DataFrame. If False, the result (must be a Series) is assigned as a
    #           new column.
    INDICATOR_CONFIG: Dict[str, Dict[str, Any]] = {
        "ATR": {
            "func_name": "atr",
            "params_map": {"length": "atr_period"},
            "main_col_pattern": "ATRr_{length}",
            "type": "decimal",
            "min_data_param_key": "length",
            "concat": False,
        },
        "EMA_Short": {
            "func_name": "ema",
            "params_map": {"length": "ema_short_period"},
            "main_col_pattern": "EMA_{length}",
            "type": "decimal",
            "pass_close_only": True,
            "min_data_param_key": "length",
            "concat": False,
        },
        "EMA_Long": {
            "func_name": "ema",
            "params_map": {"length": "ema_long_period"},
            "main_col_pattern": "EMA_{length}",
            "type": "decimal",
            "pass_close_only": True,
            "min_data_param_key": "length",
            "concat": False,
        },
        "Momentum": {
            "func_name": "mom",
            "params_map": {"length": "momentum_period"},
            "main_col_pattern": "MOM_{length}",
            "type": "float",
            "pass_close_only": True,
            "min_data_param_key": "length",
            "concat": False,
        },
        "CCI": {
            "func_name": "cci",
            "params_map": {"length": "cci_window", "c": "cci_constant"},
            "main_col_pattern": "CCI_{length}_{c:.3f}",
            "type": "float",
            "min_data_param_key": "length",
            "concat": False,
        },
        "Williams_R": {
            "func_name": "willr",
            "params_map": {"length": "williams_r_window"},
            "main_col_pattern": "WILLR_{length}",
            "type": "float",
            "min_data_param_key": "length",
            "concat": False,
        },
        "MFI": {
            "func_name": "mfi",
            "params_map": {"length": "mfi_window"},
            "main_col_pattern": "MFI_{length}",
            "type": "float",
            "concat": True,
            "min_data_param_key": "length",
        },
        "VWAP": {
            "func_name": "vwap",
            "params_map": {},
            "main_col_pattern": "VWAP_D",
            "type": "decimal",
            "concat": True,
            "min_data": 1,
        },
        "PSAR": {
            "func_name": "psar",
            # pandas-ta uses 'initial' (or af0), 'step' (or af), 'max' (or afmax)
            "params_map": {"initial": "psar_initial_af", "step": "psar_af_step", "max": "psar_max_af"},
            # pandas-ta psar returns: PSARl, PSARs, PSARaf, PSARr. We map PSARl and PSARs.
            # Column names from pandas-ta are like: PSARl_0.02_0.02_0.2
            "multi_cols": {
                "PSAR_long": "PSARl_{initial}_{step}_{max}",  # Keys match params_map keys
                "PSAR_short": "PSARs_{initial}_{step}_{max}",
            },
            "type": "decimal",
            "concat": True,
            "min_data": 2,  # PSAR typically needs few periods
        },
        "StochRSI": {
            "func_name": "stochrsi",
            "params_map": {
                "length": "stoch_rsi_window",
                "rsi_length": "stoch_rsi_rsi_window",
                "k": "stoch_rsi_k",
                "d": "stoch_rsi_d",
            },
            "multi_cols": {
                "StochRSI_K": "STOCHRSIk_{length}_{rsi_length}_{k}_{d}",
                "StochRSI_D": "STOCHRSId_{length}_{rsi_length}_{k}_{d}",
            },
            "type": "float",
            "concat": True,
            "min_data_param_key": "length",
        },
        "Bollinger_Bands": {
            "func_name": "bbands",
            "params_map": {"length": "bollinger_bands_period", "std": "bollinger_bands_std_dev"},
            # pandas-ta bbands also returns BBB_ (bandwidth) and BBP_ (percent) if needed
            "multi_cols": {
                "BB_Lower": "BBL_{length}_{std:.1f}",
                "BB_Middle": "BBM_{length}_{std:.1f}",
                "BB_Upper": "BBU_{length}_{std:.1f}",
            },
            "type": "decimal",
            "concat": True,
            "min_data_param_key": "length",
        },
        "Volume_MA": {
            "func_name": "_calculate_volume_ma",
            "params_map": {"length": "volume_ma_period"},
            "main_col_pattern": "VOL_SMA_{length}",
            "type": "decimal",
            "min_data_param_key": "length",
            "concat": False,
        },
        "SMA10": {
            "func_name": "sma",
            "params_map": {"length": "sma_10_window"},
            "main_col_pattern": "SMA_{length}",
            "type": "decimal",
            "pass_close_only": True,
            "min_data_param_key": "length",
            "concat": False,
        },
        "RSI": {
            "func_name": "rsi",
            "params_map": {"length": "rsi_period"},
            "main_col_pattern": "RSI_{length}",
            "type": "float",
            "pass_close_only": True,
            "min_data_param_key": "length",
            "concat": False,
        },
    }

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any],
    ) -> None:
        """
        Initializes the TradingAnalyzer.

        Args:
            df: Input DataFrame with OHLCV data. Timestamps should be in the index.
            logger: Logger instance for logging messages.
            config: Configuration dictionary for indicators, weights, and other settings.
            market_info: Market-specific information (symbol, precision, etc.).
        """
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")
        self.interval = str(config.get("interval", "5m"))
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval)

        self.indicator_values: Dict[str, Any] = {}
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1}  # HOLD=1 is default
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {}
        self.ta_column_names: Dict[str, str] = {}  # Stores mapping from internal key to actual DataFrame column name
        self.df_calculated: pd.DataFrame = pd.DataFrame()

        # Pre-build a map for quick lookup of an indicator's type (decimal/float)
        self.indicator_type_map: Dict[str, str] = {}
        for main_cfg_key, cfg_details in self.INDICATOR_CONFIG.items():
            default_type = cfg_details.get("type", "float")
            self.indicator_type_map[main_cfg_key] = default_type
            if "multi_cols" in cfg_details:
                for sub_key in cfg_details["multi_cols"].keys():
                    self.indicator_type_map[sub_key] = default_type

        if not isinstance(df, pd.DataFrame) or df.empty:
            self.logger.error(f"{NEON_RED}Input DataFrame for {self.symbol} is invalid or empty.{RESET}")
            raise ValueError("Input DataFrame must be a non-empty pandas DataFrame.")
        if not self.ccxt_interval:
            self.logger.error(
                f"{NEON_RED}Invalid interval '{self.interval}' for {self.symbol}. Not found in CCXT_INTERVAL_MAP.{RESET}"
            )
            raise ValueError(f"Interval '{self.interval}' not in CCXT_INTERVAL_MAP.")
        if not self.weights:
            self.logger.warning(
                f"{NEON_YELLOW}Weight set '{self.active_weight_set_name}' missing or empty for {self.symbol}. Scoring may be ineffective.{RESET}"
            )

        required_ohlcv_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_ohlcv_cols):
            missing_cols = [col for col in required_ohlcv_cols if col not in df.columns]
            self.logger.error(
                f"{NEON_RED}DataFrame for {self.symbol} missing required OHLCV columns: {missing_cols}.{RESET}"
            )
            raise ValueError(f"DataFrame must contain all OHLCV columns. Missing: {missing_cols}")

        self.df_original_ohlcv = df.copy()
        if self.df_original_ohlcv.index.tz is not None:
            self.df_original_ohlcv.index = self.df_original_ohlcv.index.tz_localize(None)

        self._validate_and_prepare_df_calculated()
        self._calculate_all_indicators()
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels()

    def _validate_and_prepare_df_calculated(self) -> None:
        """
        Validates OHLCV columns and prepares `df_calculated` with float types for pandas-ta.
        Also determines the maximum lookback period required by enabled indicators.
        """
        self.df_calculated = self.df_original_ohlcv.copy()
        required_cols = ["open", "high", "low", "close", "volume"]

        for col in required_cols:
            if col not in self.df_calculated.columns:
                self.logger.critical(
                    f"{NEON_RED}Critical column '{col}' missing for {self.symbol}. Analysis cannot proceed.{RESET}"
                )
                raise ValueError(f"Column '{col}' is missing from DataFrame.")
            try:
                self.df_calculated[col] = pd.to_numeric(self.df_calculated[col], errors="coerce")
                if not pd.api.types.is_float_dtype(self.df_calculated[col]):
                    self.df_calculated[col] = self.df_calculated[col].astype(float)
            except (ValueError, TypeError, AttributeError) as e:
                self.logger.error(
                    f"{NEON_RED}Failed to convert column '{col}' to numeric/float for {self.symbol}: {e}{RESET}",
                    exc_info=True,
                )
                raise ValueError(
                    f"Column '{col}' could not be converted to a suitable numeric type for TA calculations."
                )

            nan_count = self.df_calculated[col].isna().sum()
            if nan_count > 0:
                self.logger.warning(
                    f"{NEON_YELLOW}{nan_count} NaN values in '{col}' for {self.symbol} after prep. Total rows: {len(self.df_calculated)}{RESET}"
                )
            if not pd.api.types.is_numeric_dtype(self.df_calculated[col]):
                self.logger.error(
                    f"{NEON_RED}Column '{col}' is still not numeric after all processing for {self.symbol}. Type: {self.df_calculated[col].dtype}{RESET}"
                )
                raise ValueError(f"Column '{col}' must be numeric for TA calculations.")

        max_lookback = 1
        enabled_indicators_cfg = self.config.get("indicators", {})
        for ind_key_cfg, ind_cfg_details in self.INDICATOR_CONFIG.items():
            if enabled_indicators_cfg.get(ind_key_cfg.lower(), False):
                period_param_key = ind_cfg_details.get("min_data_param_key")
                if period_param_key and period_param_key in ind_cfg_details["params_map"]:
                    config_key_for_period = ind_cfg_details["params_map"][period_param_key]
                    period_val = self.get_period(config_key_for_period)
                    if isinstance(period_val, (int, float)) and period_val > 0:
                        max_lookback = max(max_lookback, int(period_val))
                elif isinstance(ind_cfg_details.get("min_data"), int):
                    max_lookback = max(max_lookback, ind_cfg_details.get("min_data", 1))

        min_required_rows = max_lookback + self.config.get("indicator_buffer_candles", 20)
        valid_ohlcv_rows = len(self.df_calculated.dropna(subset=required_cols))
        if valid_ohlcv_rows < min_required_rows:
            self.logger.warning(
                f"{NEON_YELLOW}Insufficient valid data rows ({valid_ohlcv_rows}) for {self.symbol} "
                f"(max lookback: {max_lookback}, min needed: {min_required_rows}). Some indicators may be all NaN.{RESET}"
            )

    def get_period(self, key: str) -> Any:
        """
        Safely retrieves a configuration value for an indicator period or parameter.
        Falls back to `DEFAULT_INDICATOR_PERIODS` if the key is not in `self.config`
        or if `self.config[key]` is None.
        """
        config_val = self.config.get(key)
        if config_val is not None:
            return config_val
        return DEFAULT_INDICATOR_PERIODS.get(key)

    def _format_ta_column_name(self, pattern: str, params: Dict[str, Any]) -> str:
        """
        Formats a technical analysis column name based on a pattern and parameters.
        This is used to generate column names consistent with pandas-ta or custom needs.
        """
        fmt_params = {}
        for k_param, v_param in params.items():
            if v_param is None:
                fmt_params[k_param] = "DEF"  # Placeholder for None parameters
                self.logger.debug(
                    f"Param '{k_param}' for column pattern '{pattern}' was None. Using placeholder 'DEF'."
                )
            elif isinstance(v_param, (float, Decimal)):
                # Convert Decimal to float for f-string formatting if specific precision is requested
                val_to_format = float(v_param) if isinstance(v_param, Decimal) else v_param
                # If pattern has specific float formatting (e.g., {std:.1f}), let f-string handle it.
                if f"{{{k_param}:." in pattern:
                    fmt_params[k_param] = val_to_format
                else:
                    # General float/Decimal to string conversion, keeping dots for pandas-ta compatibility.
                    fmt_params[k_param] = str(val_to_format)
            else:  # int, str, etc.
                fmt_params[k_param] = v_param
        try:
            return pattern.format(**fmt_params)
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error formatting TA column pattern '{pattern}' with params {fmt_params}: {e}")
            base_pattern_part = pattern.split("{")[0].rstrip("_") if pattern else "UNKNOWN_IND"
            param_keys_str = "_".join(map(str, params.values()))
            return f"{base_pattern_part}_{param_keys_str}_FORMAT_ERROR"

    def _calculate_volume_ma(self, df: pd.DataFrame, length: int) -> Optional[pd.Series]:
        """Calculates Simple Moving Average of volume. Custom TA function example."""
        if "volume" not in df.columns:
            self.logger.warning(f"Volume MA calculation failed for {self.symbol}: 'volume' column missing.")
            return None
        if not (isinstance(length, int) and length > 0):
            self.logger.warning(f"Volume MA calculation failed for {self.symbol}: Invalid length {length}.")
            return None

        volume_series = df["volume"].fillna(0).astype(float)
        if len(volume_series) < length:
            self.logger.debug(
                f"Not enough data points ({len(volume_series)}) for Volume MA with length {length} on {self.symbol}."
            )
            return pd.Series([np.nan] * len(df), index=df.index)
        return ta.sma(volume_series, length=length)

    def _calculate_all_indicators(self) -> None:
        """
        Calculates all technical indicators enabled in the configuration.
        Results are stored in `self.df_calculated`.
        Internal keys for indicators (e.g., "EMA_Short", "PSAR_long") are mapped to their
        actual column names in `self.ta_column_names`.
        """
        if self.df_calculated.empty:
            self.logger.warning(f"df_calculated is empty for {self.symbol}. Skipping indicator calculations.")
            return

        df_ta_intermediate = self.df_calculated.copy()  # Work on a copy
        enabled_cfg = self.config.get("indicators", {})

        for ind_cfg_key, ind_details in self.INDICATOR_CONFIG.items():
            # Use lowercased key for checking against config, as config keys might be lowercase
            if not enabled_cfg.get(ind_cfg_key.lower(), False):
                continue

            current_params_for_ta_func = {}
            valid_params = True
            # Prepare parameters for the TA function call
            for ta_func_param_name, config_key_for_value in ind_details["params_map"].items():
                param_value = self.get_period(config_key_for_value)
                if param_value is None:
                    self.logger.warning(
                        f"Parameter '{config_key_for_value}' for {ind_cfg_key} on {self.symbol} is None. Skipping this indicator."
                    )
                    valid_params = False
                    break
                try:  # Convert to types pandas-ta expects (usually float or int)
                    if isinstance(param_value, Decimal):
                        current_params_for_ta_func[ta_func_param_name] = float(param_value)
                    elif isinstance(param_value, str):
                        current_params_for_ta_func[ta_func_param_name] = (
                            float(param_value) if "." in param_value else int(param_value)
                        )
                    elif isinstance(param_value, (int, float)):
                        current_params_for_ta_func[ta_func_param_name] = param_value
                    else:
                        raise TypeError(f"Unsupported parameter type {type(param_value)} for {config_key_for_value}")
                except (ValueError, TypeError) as e:
                    self.logger.error(
                        f"Cannot convert parameter {config_key_for_value}='{param_value}' for {ind_cfg_key} on {self.symbol}: {e}"
                    )
                    valid_params = False
                    break
            if not valid_params:
                continue

            try:
                ta_func_name = ind_details["func_name"]
                ta_func_obj = (
                    getattr(ta, ta_func_name) if hasattr(ta, ta_func_name) else getattr(self, ta_func_name, None)
                )
                if ta_func_obj is None:
                    self.logger.error(
                        f"TA function '{ta_func_name}' for {ind_cfg_key} not found in pandas_ta or TradingAnalyzer class."
                    )
                    continue

                lookback_key = ind_details.get("min_data_param_key", "length")
                min_data_needed = int(current_params_for_ta_func.get(lookback_key, ind_details.get("min_data", 1)))
                if len(df_ta_intermediate.dropna(subset=["open", "high", "low", "close"])) < min_data_needed:
                    self.logger.debug(
                        f"Insufficient data for {ind_cfg_key} ({len(df_ta_intermediate.dropna(subset=['open', 'high', 'low', 'close']))} rows vs {min_data_needed} needed) for {self.symbol}. Skipping."
                    )
                    continue

                ta_input_args = {}  # Arguments like high, low, close, volume, open for TA function
                if ta_func_name != "_calculate_volume_ma":  # Custom functions might take df
                    import inspect  # To pass only necessary series to pandas-ta functions

                    sig_params = inspect.signature(ta_func_obj).parameters
                    if "high" in sig_params:
                        ta_input_args["high"] = df_ta_intermediate["high"]
                    if "low" in sig_params:
                        ta_input_args["low"] = df_ta_intermediate["low"]
                    if "close" in sig_params:
                        ta_input_args["close"] = df_ta_intermediate["close"]
                    if "volume" in sig_params and "volume" in df_ta_intermediate:
                        ta_input_args["volume"] = df_ta_intermediate["volume"]
                    if "open" in sig_params and "open" in df_ta_intermediate:
                        ta_input_args["open"] = df_ta_intermediate["open"]

                result_data = None
                if ta_func_name == "_calculate_volume_ma":
                    result_data = ta_func_obj(df_ta_intermediate, **current_params_for_ta_func)
                elif ind_details.get("pass_close_only", False):
                    result_data = ta_func_obj(close=df_ta_intermediate["close"], **current_params_for_ta_func)
                else:  # Standard pandas-ta call with specific series
                    result_data = ta_func_obj(**ta_input_args, **current_params_for_ta_func)

                if result_data is None:
                    self.logger.warning(f"{ind_cfg_key} calculation returned None for {self.symbol}.")
                    continue

                # Process and integrate the result
                should_concat = ind_details.get("concat", False)
                if should_concat:  # Result (Series or DataFrame) is concatenated
                    df_piece_to_add = None
                    col_name_for_series_concat = None  # If result is Series and concat=True

                    if isinstance(result_data, pd.Series):
                        if "main_col_pattern" not in ind_details:
                            self.logger.error(
                                f"Indicator {ind_cfg_key} (Series, concat=True) lacks main_col_pattern. Skipping."
                            )
                            continue
                        col_name_for_series_concat = self._format_ta_column_name(
                            ind_details["main_col_pattern"], current_params_for_ta_func
                        )
                        df_piece_to_add = result_data.to_frame(name=col_name_for_series_concat)
                    elif isinstance(result_data, pd.DataFrame):
                        df_piece_to_add = result_data.copy()  # Use a copy
                    else:
                        self.logger.warning(
                            f"Result for {ind_cfg_key} (concat=True) is not Series/DataFrame. Type: {type(result_data)}. Skipping."
                        )
                        continue

                    # Ensure numeric types, primarily float64, for concatenated piece
                    try:
                        df_piece_to_add = df_piece_to_add.astype("float64")
                    except Exception:  # If bulk cast fails, try column by column
                        self.logger.warning(
                            f"Could not cast all columns of piece for {ind_cfg_key} to float64. Trying column by column."
                        )
                        valid_cols_for_df = {}
                        for col_idx in df_piece_to_add.columns:
                            try:
                                valid_cols_for_df[col_idx] = pd.to_numeric(
                                    df_piece_to_add[col_idx], errors="raise"
                                ).astype("float64")
                            except Exception as e_col_cast:
                                self.logger.error(
                                    f"Failed to convert column {col_idx} for {ind_cfg_key} to float64: {e_col_cast}. Dropping this column."
                                )
                        df_piece_to_add = pd.DataFrame(valid_cols_for_df, index=df_piece_to_add.index)
                        if df_piece_to_add.empty:
                            self.logger.error(
                                f"Piece for {ind_cfg_key} became empty after type conversion attempts. Skipping."
                            )
                            continue

                    # Drop columns from df_ta_intermediate if they already exist to avoid duplicates from pandas-ta direct naming
                    cols_to_drop_if_exist = [
                        col for col in df_piece_to_add.columns if col in df_ta_intermediate.columns
                    ]
                    if cols_to_drop_if_exist:
                        df_ta_intermediate.drop(columns=cols_to_drop_if_exist, inplace=True, errors="ignore")

                    df_ta_intermediate = pd.concat([df_ta_intermediate, df_piece_to_add], axis=1)

                    # Map internal keys to actual column names
                    if "multi_cols" in ind_details:  # For indicators like PSAR, StochRSI, BBands
                        for internal_key, col_pattern in ind_details["multi_cols"].items():
                            actual_col_name = self._format_ta_column_name(col_pattern, current_params_for_ta_func)
                            if actual_col_name in df_ta_intermediate.columns:
                                self.ta_column_names[internal_key] = actual_col_name
                            else:
                                self.logger.warning(
                                    f"Multi-col '{actual_col_name}' (for {internal_key} of {ind_cfg_key}) not found in df_ta_intermediate. Available: {df_ta_intermediate.columns.tolist()}"
                                )
                    elif col_name_for_series_concat:  # For single Series results with concat=True (e.g., MFI, VWAP)
                        if col_name_for_series_concat in df_ta_intermediate.columns:
                            self.ta_column_names[ind_cfg_key] = col_name_for_series_concat
                        else:
                            self.logger.error(
                                f"Internal: Column {col_name_for_series_concat} for {ind_cfg_key} not found after concat."
                            )

                else:  # Not concat (concat=False). Result must be a Series, assigned as a new column.
                    if "main_col_pattern" not in ind_details:
                        self.logger.error(f"Indicator {ind_cfg_key} (concat=False) lacks main_col_pattern. Skipping.")
                        continue
                    actual_col_name = self._format_ta_column_name(
                        ind_details["main_col_pattern"], current_params_for_ta_func
                    )
                    if isinstance(result_data, pd.Series):
                        if actual_col_name in df_ta_intermediate.columns:
                            self.logger.debug(
                                f"Overwriting column '{actual_col_name}' for {ind_cfg_key} in df_ta_intermediate."
                            )
                        df_ta_intermediate[actual_col_name] = result_data.astype("float64")  # Ensure float type
                        self.ta_column_names[ind_cfg_key] = actual_col_name  # Map main config key
                    else:
                        self.logger.warning(
                            f"Result for {ind_cfg_key} (concat=False, col '{actual_col_name}') not pd.Series. Type: {type(result_data)}. Skipping."
                        )

            except Exception as e:
                self.logger.error(
                    f"Error calculating indicator {ind_cfg_key} for {self.symbol} with params {current_params_for_ta_func}: {e}",
                    exc_info=True,
                )

        self.df_calculated = df_ta_intermediate
        self.logger.debug(
            f"Indicator calculation complete for {self.symbol}. Resulting columns: {self.df_calculated.columns.tolist()}"
        )
        self.logger.debug(f"Final mapped TA column names for {self.symbol}: {self.ta_column_names}")

    def _update_latest_indicator_values(self) -> None:
        """
        Updates `self.indicator_values` with the latest calculated indicator values
        and OHLCV data. Values are converted to Decimal or float based on `INDICATOR_CONFIG`.
        OHLCV values are sourced from `self.df_original_ohlcv` to preserve Decimal types.
        """
        df_indicators_src = self.df_calculated
        df_ohlcv_src = self.df_original_ohlcv  # Use original for OHLCV to preserve Decimal type

        ohlcv_keys_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        # Initialize all potential keys to NaN to ensure they exist
        all_expected_keys = set(self.indicator_type_map.keys()) | set(ohlcv_keys_map.keys())
        temp_indicator_values: Dict[str, Any] = {k: np.nan for k in all_expected_keys}

        if df_indicators_src.empty:
            self.logger.warning(
                f"Cannot update latest values for {self.symbol}: Indicator DataFrame (df_calculated) is empty."
            )
            self.indicator_values = temp_indicator_values
            return
        if df_ohlcv_src.empty:  # Should not happen if df_indicators_src is not empty, but good check
            self.logger.warning(
                f"Cannot update latest values for {self.symbol}: Original OHLCV DataFrame (df_original_ohlcv) is empty."
            )
            self.indicator_values = temp_indicator_values
            return

        try:
            if df_indicators_src.index.empty or df_ohlcv_src.index.empty:
                self.logger.error(f"Cannot get latest row for {self.symbol}: DataFrame index is empty.")
                self.indicator_values = temp_indicator_values
                return

            latest_indicator_row = df_indicators_src.iloc[-1]
            latest_ohlcv_row = df_ohlcv_src.iloc[
                -1
            ]  # Assumes synchronized indices or picking latest irrespective of exact timestamp match

            self.logger.debug(
                f"Updating latest values for {self.symbol} from indicator row dated: {latest_indicator_row.name}, OHLCV row dated: {latest_ohlcv_row.name}"
            )

            # Process TA indicators from df_calculated (mostly floats)
            for internal_key, actual_col_name in self.ta_column_names.items():
                if actual_col_name and actual_col_name in latest_indicator_row.index:
                    value = latest_indicator_row[actual_col_name]
                    indicator_target_type = self.indicator_type_map.get(internal_key, "float")  # Default to float

                    if pd.notna(value):
                        try:
                            if indicator_target_type == "decimal":
                                temp_indicator_values[internal_key] = Decimal(str(value))
                            else:
                                temp_indicator_values[internal_key] = float(value)
                        except (ValueError, TypeError, InvalidOperation) as e_conv:
                            self.logger.warning(
                                f"Conversion error for {internal_key} ('{actual_col_name}':{value}) to {indicator_target_type}: {e_conv}. Storing as NaN."
                            )
                            temp_indicator_values[internal_key] = np.nan
                    else:
                        temp_indicator_values[internal_key] = np.nan
                else:
                    self.logger.debug(
                        f"Internal key '{internal_key}' (mapped to '{actual_col_name}') not found in latest indicator row for {self.symbol} or actual_col_name is empty. Storing as NaN."
                    )
                    temp_indicator_values[internal_key] = np.nan

            # Process OHLCV values from df_original_ohlcv (preferring Decimals)
            for display_key, source_col_name in ohlcv_keys_map.items():
                value_ohlcv = latest_ohlcv_row.get(source_col_name)
                if pd.notna(value_ohlcv):
                    try:
                        if isinstance(value_ohlcv, Decimal):
                            temp_indicator_values[display_key] = value_ohlcv
                        else:
                            temp_indicator_values[display_key] = Decimal(str(value_ohlcv))
                    except InvalidOperation:
                        self.logger.warning(
                            f"Failed to convert original OHLCV value '{source_col_name}' ({value_ohlcv}) to Decimal for {self.symbol}. Storing as NaN."
                        )
                        temp_indicator_values[display_key] = np.nan
                else:
                    temp_indicator_values[display_key] = np.nan

            self.indicator_values = temp_indicator_values

            if "ATR" in self.indicator_values and pd.notna(self.indicator_values.get("ATR")):
                self.logger.info(
                    f"DEBUG ATR for {self.symbol}: Final ATR in self.indicator_values: {self.indicator_values.get('ATR')}, Type: {type(self.indicator_values.get('ATR'))}"
                )

            price_prec_log = get_price_precision(self.market_info, self.logger)
            log_output_details = {}
            # Define keys that usually represent prices or price-like values (e.g., ATR, EMA, VWAP, PSAR, BB)
            decimal_like_keys = [k for k, v_type in self.indicator_type_map.items() if v_type == "decimal"] + [
                "Open",
                "High",
                "Low",
                "Close",
            ]
            volume_like_keys = ["Volume", "Volume_MA"]  # Explicitly list volume-like keys

            for k_log, v_val_log in self.indicator_values.items():
                if isinstance(v_val_log, (Decimal, float)) and pd.notna(v_val_log):
                    if k_log in decimal_like_keys:
                        fmt_str = f"{Decimal(str(v_val_log)):.{price_prec_log}f}"  # Ensure Decimal then format
                    elif k_log in volume_like_keys:
                        fmt_str = f"{Decimal(str(v_val_log)):.8f}"  # Volume often needs more precision
                    else:
                        fmt_str = f"{float(v_val_log):.4f}"  # Default for other floats (RSI, MFI, etc.)
                    log_output_details[k_log] = fmt_str
                else:
                    log_output_details[k_log] = str(v_val_log)  # NaN or other types
            self.logger.debug(f"Latest indicator values updated for {self.symbol}: {log_output_details}")

        except IndexError:  # If iloc[-1] fails
            self.logger.error(
                f"IndexError accessing latest row for {self.symbol}. Check DataFrame integrity and length."
            )
            self.indicator_values = temp_indicator_values  # Fallback to NaN-initialized dict
        except Exception as e:
            self.logger.error(f"Unexpected error updating latest indicators for {self.symbol}: {e}", exc_info=True)
            if not self.indicator_values:  # If it's still empty, ensure it's initialized
                self.indicator_values = temp_indicator_values

    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """
        Calculates Fibonacci retracement levels based on the high/low over a specified window.
        The levels are quantized according to the market's price precision or minimum tick size.
        """
        cfg_window = self.get_period("fibonacci_window")
        window_val = window if isinstance(window, int) and window > 0 else cfg_window

        if not (isinstance(window_val, int) and window_val > 0):
            self.logger.warning(f"Invalid Fibonacci window ({window_val}) for {self.symbol}. No levels calculated.")
            self.fib_levels_data = {}
            return {}
        if len(self.df_original_ohlcv) < window_val:
            self.logger.debug(
                f"Not enough data ({len(self.df_original_ohlcv)} rows) for Fibonacci window {window_val} on {self.symbol}."
            )
            self.fib_levels_data = {}
            return {}

        df_slice = self.df_original_ohlcv.tail(window_val)
        try:
            h_series = pd.to_numeric(df_slice["high"], errors="coerce").dropna()
            l_series = pd.to_numeric(df_slice["low"], errors="coerce").dropna()

            if h_series.empty or l_series.empty:
                self.logger.warning(f"No valid high/low data for Fibonacci calculation in window for {self.symbol}.")
                self.fib_levels_data = {}
                return {}

            period_high = Decimal(str(h_series.max()))
            period_low = Decimal(str(l_series.min()))
            diff = period_high - period_low
            levels: Dict[str, Decimal] = {}

            price_precision = get_price_precision(self.market_info, self.logger)
            min_tick_size = get_min_tick_size(self.market_info, self.logger)
            quantize_factor = (
                min_tick_size if min_tick_size and min_tick_size > Decimal("0") else Decimal(f"1e-{price_precision}")
            )

            if diff > Decimal("0"):
                for level_pct in FIB_LEVELS:  # e.g., [Decimal('0.236'), Decimal('0.382'), ...]
                    level_price_raw = period_high - (diff * Decimal(str(level_pct)))  # Ensure level_pct is Decimal
                    levels[f"Fib_{level_pct * 100:.1f}%"] = (level_price_raw / quantize_factor).quantize(
                        Decimal("1"), rounding=ROUND_DOWN
                    ) * quantize_factor
            else:  # If high and low are the same, or low > high (data error)
                level_price_quantized = (period_high / quantize_factor).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * quantize_factor
                for level_pct in FIB_LEVELS:
                    levels[f"Fib_{level_pct * 100:.1f}%"] = level_price_quantized

            self.fib_levels_data = levels
            log_levels_str = {k: f"{v:.{price_precision}f}" for k, v in levels.items()}
            self.logger.debug(
                f"Calculated Fibonacci levels for {self.symbol} (Window: {window_val}, High: {period_high:.{price_precision}f}, Low: {period_low:.{price_precision}f}): {log_levels_str}"
            )
            return levels
        except Exception as e:
            self.logger.error(f"Fibonacci calculation error for {self.symbol}: {e}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> List[Tuple[str, Decimal]]:  # Changed to List
        """Finds the N nearest Fibonacci levels to the current price."""
        if not self.fib_levels_data:
            self.logger.debug(f"No Fibonacci levels available for {self.symbol} to find nearest.")
            return []
        if not (isinstance(current_price, Decimal) and pd.notna(current_price) and current_price > Decimal("0")):
            self.logger.warning(f"Invalid current_price ({current_price}) for Fibonacci comparison on {self.symbol}.")
            return []
        if num_levels <= 0:
            return []

        try:
            distances = []
            for name, level_price in self.fib_levels_data.items():
                if isinstance(level_price, Decimal) and level_price > Decimal("0"):
                    distances.append({"name": name, "level": level_price, "distance": abs(current_price - level_price)})
            if not distances:
                return []  # No valid levels to compare against

            distances.sort(key=lambda x: x["distance"])
            return [(item["name"], item["level"]) for item in distances[:num_levels]]
        except Exception as e:
            self.logger.error(f"Error finding nearest Fibonacci levels for {self.symbol}: {e}", exc_info=True)
            return []

    def calculate_ema_alignment_score(self) -> float:
        """
        Calculates a score based on EMA alignment (short EMA vs long EMA) and price position
        relative to the short EMA. Returns a float score between -1.0 and 1.0, or np.nan.
        """
        ema_s_val = self.indicator_values.get("EMA_Short")
        ema_l_val = self.indicator_values.get("EMA_Long")
        close_price_val = self.indicator_values.get("Close")

        # Ensure all values are valid Decimals before comparison
        if not all(isinstance(val, Decimal) and pd.notna(val) for val in [ema_s_val, ema_l_val, close_price_val]):
            self.logger.debug(
                f"EMA alignment score skipped for {self.symbol}: one or more values (EMA_Short, EMA_Long, Close) are invalid/NaN."
            )
            return np.nan

        # Type casting for mypy after validation
        ema_s: Decimal = ema_s_val
        ema_l: Decimal = ema_l_val
        close_price: Decimal = close_price_val

        if close_price > ema_s and ema_s > ema_l:
            return 1.0  # Strong bullish: Price > ShortEMA > LongEMA
        if close_price < ema_s and ema_s < ema_l:
            return -1.0  # Strong bearish: Price < ShortEMA < LongEMA
        # Could add more nuanced scores for partial alignments (e.g., price between EMAs)
        return 0.0  # Neutral or mixed alignment

    def generate_trading_signal(self, current_price: Decimal, orderbook_data: Optional[Dict]) -> str:
        """
        Generates a trading signal (BUY, SELL, HOLD) based on a weighted sum of scores
        from various enabled indicator check methods.
        """
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}  # Reset, default to HOLD
        current_score_decimal, total_weight_decimal = Decimal("0"), Decimal("0")
        active_checks_count, nan_checks_count = 0, 0
        debug_scores_log: Dict[str, str] = {}

        if not self.indicator_values:
            self.logger.warning(f"No indicator values available for {self.symbol}. Defaulting to HOLD signal.")
            return "HOLD"

        atr_val = self.indicator_values.get("ATR")
        if not (isinstance(atr_val, Decimal) and pd.notna(atr_val) and atr_val > Decimal("0")):
            self.logger.warning(
                f"Signal generation for {self.symbol}: ATR is invalid ({atr_val}). This might affect subsequent TP/SL calculations if ATR is used."
            )

        # Count valid core indicators (those in INDICATOR_CONFIG and successfully calculated)
        valid_core_indicator_count = 0
        for ind_key_main_cfg in self.INDICATOR_CONFIG.keys():
            if "multi_cols" in self.INDICATOR_CONFIG[ind_key_main_cfg]:
                if any(
                    pd.notna(self.indicator_values.get(sub_key))
                    for sub_key in self.INDICATOR_CONFIG[ind_key_main_cfg]["multi_cols"]
                ):
                    valid_core_indicator_count += 1
            elif pd.notna(self.indicator_values.get(ind_key_main_cfg)):  # Check main key itself
                valid_core_indicator_count += 1

        num_configured_inds_enabled = sum(
            1 for enabled_flag in self.config.get("indicators", {}).values() if enabled_flag
        )
        min_active_inds_for_signal = self.config.get(
            "min_active_indicators_for_signal", max(1, int(num_configured_inds_enabled * 0.6))
        )

        if valid_core_indicator_count < min_active_inds_for_signal:
            self.logger.warning(
                f"Signal for {self.symbol}: Only {valid_core_indicator_count}/{num_configured_inds_enabled} core indicators are valid (min required: {min_active_inds_for_signal}). Defaulting to HOLD."
            )
            return "HOLD"
        if not (isinstance(current_price, Decimal) and pd.notna(current_price) and current_price > Decimal("0")):
            self.logger.warning(
                f"Invalid current_price ({current_price}) for {self.symbol} signal generation. Defaulting to HOLD."
            )
            return "HOLD"

        active_weights_dict = self.weights
        if not active_weights_dict:
            self.logger.error(
                f"Weight set '{self.active_weight_set_name}' is empty for {self.symbol}. Cannot generate signal. Defaulting to HOLD."
            )
            return "HOLD"

        # Iterate through check methods corresponding to enabled indicators in config
        for indicator_check_key_lower in active_weights_dict.keys():  # Keys in weights are expected to be lowercase
            # Check if this indicator is actually enabled in the main "indicators" config section
            if not self.config.get("indicators", {}).get(indicator_check_key_lower, False):
                continue  # Skip if not enabled, even if a weight exists

            weight_str_val = active_weights_dict.get(indicator_check_key_lower)
            if weight_str_val is None:
                continue
            try:
                weight_decimal = Decimal(str(weight_str_val))
            except InvalidOperation:
                self.logger.warning(
                    f"Invalid weight '{weight_str_val}' for {indicator_check_key_lower} for {self.symbol}. Skipping this check."
                )
                continue
            if weight_decimal == Decimal("0"):
                continue  # Zero weight, no contribution

            check_method_name_str = f"_check_{indicator_check_key_lower}"
            if not hasattr(self, check_method_name_str) or not callable(getattr(self, check_method_name_str)):
                self.logger.warning(
                    f"No check method '{check_method_name_str}' found for enabled indicator {indicator_check_key_lower} ({self.symbol})."
                )
                continue

            method_to_call_obj = getattr(self, check_method_name_str)
            individual_indicator_score_float = np.nan
            try:
                if indicator_check_key_lower == "orderbook":  # Special case for orderbook data
                    individual_indicator_score_float = method_to_call_obj(orderbook_data, current_price)
                else:
                    individual_indicator_score_float = method_to_call_obj()
            except Exception as e_check_method:
                self.logger.error(
                    f"Error in check method {check_method_name_str} for {self.symbol}: {e_check_method}", exc_info=True
                )

            debug_scores_log[indicator_check_key_lower] = (
                f"{individual_indicator_score_float:.3f}" if pd.notna(individual_indicator_score_float) else "NaN"
            )
            if pd.notna(individual_indicator_score_float):
                try:
                    indicator_score_decimal = Decimal(str(individual_indicator_score_float))
                    clamped_score = max(Decimal("-1"), min(Decimal("1"), indicator_score_decimal))  # Clamp score
                    current_score_decimal += clamped_score * weight_decimal
                    total_weight_decimal += abs(weight_decimal)
                    active_checks_count += 1
                except InvalidOperation:
                    nan_checks_count += 1
                    self.logger.error(
                        f"Error processing score for {indicator_check_key_lower} (value: {individual_indicator_score_float})."
                    )
            else:
                nan_checks_count += 1

        final_signal_decision_str = "HOLD"
        signal_score_threshold = Decimal(str(self.get_period("signal_score_threshold") or "0.7"))

        if total_weight_decimal == Decimal("0") and active_checks_count == 0:
            self.logger.warning(
                f"No weighted indicators contributed to the score for {self.symbol}. Defaulting to HOLD."
            )
        elif current_score_decimal >= signal_score_threshold:
            final_signal_decision_str = "BUY"
        elif current_score_decimal <= -signal_score_threshold:
            final_signal_decision_str = "SELL"

        price_prec = get_price_precision(self.market_info, self.logger)
        self.logger.info(
            f"Signal ({self.symbol} @ {current_price:.{price_prec}f}): "
            f"Set='{self.active_weight_set_name}', Checks[Act:{active_checks_count},NaN:{nan_checks_count}], "
            f"TotalWeightAbs={total_weight_decimal:.2f}, Score={current_score_decimal:.4f} (Threshold:{signal_score_threshold:.2f}) "
            f"==> {_format_signal(final_signal_decision_str)}"
        )
        self.logger.debug(f"Individual Scores ({self.symbol}): {debug_scores_log}")

        self.signals = {
            "BUY": int(final_signal_decision_str == "BUY"),
            "SELL": int(final_signal_decision_str == "SELL"),
            "HOLD": int(final_signal_decision_str == "HOLD"),
        }
        return final_signal_decision_str

    # --- Individual Indicator Check Methods (_check_...) ---
    # Each method should:
    # 1. Retrieve necessary value(s) from `self.indicator_values`.
    # 2. Handle potential `np.nan` or invalid data.
    # 3. Retrieve any specific thresholds from `self.get_period()`.
    # 4. Implement logic to produce a score, typically between -1.0 (strong sell/bearish)
    #    and 1.0 (strong buy/bullish), or `np.nan` if not applicable/calculable.
    # Method names must be `_check_{key}` where `key` is lowercase and matches
    # keys in `self.config["indicators"]` and `self.config["weight_sets"][active_set]`.

    def _check_ema_alignment(self) -> float:
        """Checks EMA alignment and price position. Score: 1.0 bullish, -1.0 bearish, 0.0 neutral."""
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        """Checks momentum relative to price. Score scaled by momentum strength."""
        momentum_val = self.indicator_values.get("Momentum")
        last_close_val = self.indicator_values.get("Close")
        if pd.isna(momentum_val) or not (isinstance(last_close_val, Decimal) and last_close_val > Decimal("0")):
            return np.nan

        try:
            momentum_decimal = Decimal(str(momentum_val))
            mom_pct = (momentum_decimal / last_close_val) * Decimal("100")
            threshold_pct = Decimal(str(self.get_period("momentum_threshold_pct") or "0.1"))
        except (ZeroDivisionError, InvalidOperation, TypeError):
            return 0.0

        if threshold_pct == Decimal("0"):
            return 0.0
        # Scale score: Full score at threshold_pct * 5, linear in between
        scaling_factor = threshold_pct * Decimal("5")
        if scaling_factor == Decimal("0"):
            return 0.0  # Avoid division by zero if threshold_pct is very small

        score_unclamped = mom_pct / scaling_factor
        return float(max(Decimal("-1"), min(Decimal("1"), score_unclamped)))

    def _check_volume_confirmation(self) -> float:
        """Checks if current volume is significantly above its moving average."""
        current_volume_val = self.indicator_values.get("Volume")
        volume_ma_val = self.indicator_values.get("Volume_MA")
        try:
            multiplier_val = Decimal(str(self.get_period("volume_confirmation_multiplier") or "1.5"))
        except (InvalidOperation, TypeError):
            return np.nan

        if not all(isinstance(v, Decimal) and pd.notna(v) for v in [current_volume_val, volume_ma_val, multiplier_val]):
            return np.nan
        # Type cast for mypy
        current_volume, volume_ma, multiplier = current_volume_val, volume_ma_val, multiplier_val
        if current_volume < Decimal("0") or volume_ma <= Decimal("0") or multiplier <= Decimal("0"):
            return np.nan

        try:
            ratio = current_volume / volume_ma
            if ratio > multiplier:  # Significantly higher volume
                base_score, scale_top_ratio = Decimal("0.5"), multiplier * Decimal("5")  # Score from 0.5 to 1.0
                if scale_top_ratio == multiplier:  # Avoid division by zero if multiplier is large or scale is too small
                    return 1.0 if ratio >= multiplier else 0.5
                additional_score_pct = (ratio - multiplier) / (scale_top_ratio - multiplier)
                return float(min(Decimal("1.0"), base_score + additional_score_pct * Decimal("0.5")))
            if ratio < (Decimal("1") / multiplier if multiplier > Decimal("0") else Decimal("0")):
                return -0.4  # Low volume
            return 0.0  # Neutral volume
        except (ZeroDivisionError, InvalidOperation, TypeError):
            return np.nan

    def _check_stoch_rsi(self) -> float:
        """Checks StochRSI K & D lines for overbought/oversold conditions and crosses."""
        k_val = self.indicator_values.get("StochRSI_K")
        d_val = self.indicator_values.get("StochRSI_D")
        if pd.isna(k_val) or pd.isna(d_val):
            return np.nan

        k_float, d_float = float(k_val), float(d_val)
        oversold_thresh = float(self.get_period("stoch_rsi_oversold_threshold") or 20)
        overbought_thresh = float(self.get_period("stoch_rsi_overbought_threshold") or 80)
        cross_thresh_val = self.get_period("stoch_rsi_cross_threshold")
        cross_thresh = (
            float(cross_thresh_val) if isinstance(cross_thresh_val, (int, float)) and cross_thresh_val > 0 else 5.0
        )

        score = 0.0
        if k_float < oversold_thresh and d_float < oversold_thresh:
            score = 0.8  # Base for oversold
        elif k_float > overbought_thresh and d_float > overbought_thresh:
            score = -0.8  # Base for overbought

        diff = k_float - d_float  # K - D
        if score > 0 and diff > 0:
            score = 1.0  # K crossing up D in oversold is stronger buy
        elif score < 0 and diff < 0:
            score = -1.0  # K crossing down D in overbought is stronger sell
        elif abs(diff) > cross_thresh:  # Significant cross outside OB/OS zones
            score = 0.6 if diff > 0 else -0.6
        elif k_float > d_float and score == 0.0:
            score = 0.2  # K above D, not OB/OS
        elif k_float < d_float and score == 0.0:
            score = -0.2  # K below D, not OB/OS

        if 40 < k_float < 60 and 40 < d_float < 60:
            score *= 0.5  # Dampen if in neutral zone
        return score

    def _check_rsi(self) -> float:
        """Checks RSI for overbought/oversold conditions."""
        rsi_val = self.indicator_values.get("RSI")
        if pd.isna(rsi_val):
            return np.nan
        rsi_float = float(rsi_val)

        oversold = float(self.get_period("rsi_oversold_threshold") or 30)
        overbought = float(self.get_period("rsi_overbought_threshold") or 70)
        near_oversold = float(self.get_period("rsi_near_oversold_threshold") or 40)
        near_overbought = float(self.get_period("rsi_near_overbought_threshold") or 60)

        if rsi_float <= oversold:
            return 1.0
        if rsi_float >= overbought:
            return -1.0
        if rsi_float < near_oversold:
            return 0.5
        if rsi_float > near_overbought:
            return -0.5

        mid_point = (near_overbought + near_oversold) / 2.0
        span = (near_overbought - near_oversold) / 2.0
        if span > 0 and near_oversold <= rsi_float <= near_overbought:
            # Score from -0.3 (at near_overbought) to +0.3 (at near_oversold)
            return ((rsi_float - mid_point) / span) * -0.3
        return 0.0

    def _check_cci(self) -> float:
        """Checks CCI for extreme levels indicating potential trend reversals or continuations."""
        cci_val = self.indicator_values.get("CCI")
        if pd.isna(cci_val):
            return np.nan
        cci_float = float(cci_val)

        strong_os = float(self.get_period("cci_strong_oversold") or -150)
        strong_ob = float(self.get_period("cci_strong_overbought") or 150)
        moderate_os = float(self.get_period("cci_moderate_oversold") or -100)  # Standard CCI level
        moderate_ob = float(self.get_period("cci_moderate_overbought") or 100)  # Standard CCI level

        if cci_float <= strong_os:
            return 1.0
        if cci_float >= strong_ob:
            return -1.0
        if cci_float < moderate_os:
            return 0.6
        if cci_float > moderate_ob:
            return -0.6
        if moderate_os <= cci_float < 0:
            return 0.1  # Rising from moderate oversold
        if 0 < cci_float <= moderate_ob:
            return -0.1  # Falling from moderate overbought
        return 0.0

    def _check_wr(self) -> float:  # Williams %R
        """Checks Williams %R for overbought/oversold conditions."""
        wr_val = self.indicator_values.get("Williams_R")  # Typically -100 (most oversold) to 0 (most overbought)
        if pd.isna(wr_val):
            return np.nan
        wr_float = float(wr_val)

        oversold = float(self.get_period("wr_oversold_threshold") or -80)  # e.g., -80 to -100
        overbought = float(self.get_period("wr_overbought_threshold") or -20)  # e.g., -20 to 0
        midpoint = float(self.get_period("wr_midpoint_threshold") or -50)

        if wr_float <= oversold:
            return 1.0  # Strongly oversold
        if wr_float >= overbought:
            return -1.0  # Strongly overbought
        if oversold < wr_float < midpoint:
            return 0.4  # Approaching midpoint from oversold
        if midpoint < wr_float < overbought:
            return -0.4  # Approaching midpoint from overbought
        return 0.0  # Near midpoint

    def _check_psar(self) -> float:
        """Checks Parabolic SAR trend direction relative to price."""
        psar_long_val = self.indicator_values.get("PSAR_long")
        psar_short_val = self.indicator_values.get("PSAR_short")
        close_price_val = self.indicator_values.get("Close")

        if not isinstance(close_price_val, Decimal) or pd.isna(close_price_val):
            return np.nan
        close_price: Decimal = close_price_val

        is_long_trend_active = isinstance(psar_long_val, Decimal) and pd.notna(psar_long_val)
        is_short_trend_active = isinstance(psar_short_val, Decimal) and pd.notna(psar_short_val)

        if is_long_trend_active and not is_short_trend_active and close_price > psar_long_val:
            return 1.0  # Uptrend: PSAR dot is below price
        if is_short_trend_active and not is_long_trend_active and close_price < psar_short_val:
            return -1.0  # Downtrend: PSAR dot is above price

        if not is_long_trend_active and not is_short_trend_active:  # Both NaN (e.g. start of data)
            return np.nan

        # Ambiguous: e.g., both PSAR values present, or price crossed active PSAR (flip might be imminent or just happened)
        self.logger.debug(
            f"PSAR ambiguous state for {self.symbol}: PSARl={psar_long_val}, PSARs={psar_short_val}, Close={close_price}"
        )
        return 0.0

    def _check_sma_10(self) -> float:
        """Checks price position relative to SMA10."""
        sma_val = self.indicator_values.get("SMA10")
        last_close_val = self.indicator_values.get("Close")
        if not all(isinstance(v, Decimal) and pd.notna(v) for v in [sma_val, last_close_val]):
            return np.nan
        sma, last_close = sma_val, last_close_val  # mypy cast

        if last_close > sma:
            return 0.6
        if last_close < sma:
            return -0.6
        return 0.0

    def _check_vwap(self) -> float:
        """Checks price position relative to VWAP."""
        vwap_val = self.indicator_values.get("VWAP")
        last_close_val = self.indicator_values.get("Close")
        if not all(isinstance(v, Decimal) and pd.notna(v) for v in [vwap_val, last_close_val]):
            return np.nan
        vwap, last_close = vwap_val, last_close_val  # mypy cast

        if last_close > vwap:
            return 0.7
        if last_close < vwap:
            return -0.7
        return 0.0

    def _check_mfi(self) -> float:  # Money Flow Index
        """Checks MFI for overbought/oversold conditions, indicating buying/selling pressure."""
        mfi_val = self.indicator_values.get("MFI")
        if pd.isna(mfi_val):
            return np.nan
        mfi_float = float(mfi_val)

        oversold = float(self.get_period("mfi_oversold_threshold") or 20)
        overbought = float(self.get_period("mfi_overbought_threshold") or 80)
        near_os = float(self.get_period("mfi_near_oversold_threshold") or 35)
        near_ob = float(self.get_period("mfi_near_overbought_threshold") or 65)

        if mfi_float <= oversold:
            return 1.0
        if mfi_float >= overbought:
            return -1.0
        if mfi_float < near_os:
            return 0.4
        if mfi_float > near_ob:
            return -0.4
        return 0.0

    def _check_bollinger_bands(self) -> float:
        """Checks price position relative to Bollinger Bands. Extreme touches might signal reversals."""
        lower_bb_val = self.indicator_values.get("BB_Lower")
        middle_bb_val = self.indicator_values.get("BB_Middle")
        upper_bb_val = self.indicator_values.get("BB_Upper")
        last_close_val = self.indicator_values.get("Close")

        if not all(
            isinstance(v, Decimal) and pd.notna(v) for v in [lower_bb_val, middle_bb_val, upper_bb_val, last_close_val]
        ):
            return np.nan
        lower_bb, middle_bb, upper_bb, last_close = (
            lower_bb_val,
            middle_bb_val,
            upper_bb_val,
            last_close_val,
        )  # mypy cast

        if last_close <= lower_bb:
            return 1.0  # Touched/below lower band (potential buy)
        if last_close >= upper_bb:
            return -1.0  # Touched/above upper band (potential sell)

        band_width = upper_bb - lower_bb
        if band_width > Decimal("0"):
            try:  # Normalize position: -1 (lower_bb) to +1 (upper_bb), 0 at middle_bb. Scale score within bands.
                position_score_raw = (last_close - middle_bb) / (band_width / Decimal("2"))
                # Max score of +/-0.7 within bands, scaled linearly
                return float(max(Decimal("-1"), min(Decimal("1"), position_score_raw)) * Decimal("0.7"))
            except (ZeroDivisionError, InvalidOperation, TypeError):
                return 0.0
        return 0.0  # Flat bands or error

    def _check_orderbook(
        self, orderbook_data: Optional[Dict], current_price: Decimal
    ) -> float:  # current_price unused here but part of signature pattern
        """Analyzes order book depth for short-term pressure. Score is Order Book Imbalance (OBI)."""
        if not orderbook_data:
            return np.nan
        try:
            bids = orderbook_data.get("bids", [])  # List of [price_decimal, quantity_decimal]
            asks = orderbook_data.get("asks", [])
            if not bids or not asks:
                return np.nan

            num_levels = self.config.get("orderbook_check_levels", 10)

            # Sum quantities from Decimal entries
            total_bid_qty = sum(
                b[1] for b in bids[:num_levels] if len(b) == 2 and isinstance(b[1], Decimal) and pd.notna(b[1])
            )
            total_ask_qty = sum(
                a[1] for a in asks[:num_levels] if len(a) == 2 and isinstance(a[1], Decimal) and pd.notna(a[1])
            )

            # Fallback for string quantities if Decimals not directly available
            if total_bid_qty == Decimal("0") and total_ask_qty == Decimal("0"):
                total_bid_qty = sum(Decimal(str(b[1])) for b in bids[:num_levels] if len(b) == 2 and pd.notna(b[1]))
                total_ask_qty = sum(Decimal(str(a[1])) for a in asks[:num_levels] if len(a) == 2 and pd.notna(a[1]))

            total_qty_in_levels = total_bid_qty + total_ask_qty
            if total_qty_in_levels == Decimal("0"):
                return 0.0

            obi = (total_bid_qty - total_ask_qty) / total_qty_in_levels
            return float(max(Decimal("-1"), min(Decimal("1"), obi)))  # Clamp OBI just in case
        except (TypeError, ValueError, InvalidOperation, IndexError) as e:
            self.logger.warning(f"Order book analysis error for {self.symbol}: {e}", exc_info=False)
            return np.nan

    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates Take Profit (TP) and Stop Loss (SL) levels.
        Uses ATR and configurable multipliers. Quantizes results to market's tick size/precision.
        """
        if signal not in ["BUY", "SELL"]:
            return entry_price_estimate, None, None

        atr_value_orig = self.indicator_values.get("ATR")
        # Validate entry price first, as it's needed for fallback ATR
        if not (
            isinstance(entry_price_estimate, Decimal)
            and pd.notna(entry_price_estimate)
            and entry_price_estimate > Decimal("0")
        ):
            self.logger.warning(
                f"Cannot calculate TP/SL for {self.symbol} {signal}: Entry price estimate invalid ({entry_price_estimate})."
            )
            return entry_price_estimate, None, None

        atr_value_final: Optional[Decimal] = None
        if isinstance(atr_value_orig, Decimal) and pd.notna(atr_value_orig) and atr_value_orig > Decimal("0"):
            atr_value_final = atr_value_orig
        else:
            self.logger.warning(
                f"TP/SL Calc ({self.symbol} {signal}): ATR invalid ({atr_value_orig}). Using default ATR based on price percentage."
            )
            default_atr_pct_str = str(self.get_period("default_atr_percentage_of_price") or "0.01")  # e.g., 1%
            try:
                atr_value_final = entry_price_estimate * Decimal(default_atr_pct_str)
                if not (atr_value_final > Decimal("0")):  # Check if calculated ATR is positive
                    self.logger.error(
                        f"Default ATR calculation resulted in non-positive value ({atr_value_final}) for {self.symbol}. Cannot set TP/SL."
                    )
                    return entry_price_estimate, None, None
            except InvalidOperation:
                self.logger.error(f"Invalid 'default_atr_percentage_of_price': {default_atr_pct_str}. No TP/SL.")
                return entry_price_estimate, None, None
            self.logger.debug(f"Using price-percentage based ATR for {self.symbol} TP/SL: {atr_value_final}")

        if atr_value_final is None or not (atr_value_final > Decimal("0")):  # Should be caught above, but final check
            self.logger.error(f"Final ATR value ({atr_value_final}) is invalid for {self.symbol}. Cannot set TP/SL.")
            return entry_price_estimate, None, None

        try:
            tp_multiplier = Decimal(str(self.get_period("take_profit_multiple") or "1.5"))
            sl_multiplier = Decimal(str(self.get_period("stop_loss_multiple") or "1.0"))

            price_precision = get_price_precision(self.market_info, self.logger)
            min_tick = get_min_tick_size(self.market_info, self.logger)
            quantize_unit = min_tick if min_tick and min_tick > Decimal("0") else Decimal(f"1e-{price_precision}")
            if not (quantize_unit > Decimal("0")):
                quantize_unit = Decimal(f"1e-{price_precision}")  # Safety for quantize_unit

            tp_offset = atr_value_final * tp_multiplier
            sl_offset = atr_value_final * sl_multiplier

            raw_tp, raw_sl = (Decimal("0"), Decimal("0"))
            if signal == "BUY":
                raw_tp = entry_price_estimate + tp_offset
                raw_sl = entry_price_estimate - sl_offset
                quantized_tp = (raw_tp / quantize_unit).quantize(Decimal("1"), rounding=ROUND_UP) * quantize_unit
                quantized_sl = (raw_sl / quantize_unit).quantize(Decimal("1"), rounding=ROUND_DOWN) * quantize_unit
            else:  # SELL
                raw_tp = entry_price_estimate - tp_offset
                raw_sl = entry_price_estimate + sl_offset
                quantized_tp = (raw_tp / quantize_unit).quantize(Decimal("1"), rounding=ROUND_DOWN) * quantize_unit
                quantized_sl = (raw_sl / quantize_unit).quantize(Decimal("1"), rounding=ROUND_UP) * quantize_unit

            # Sanity checks
            if min_tick and min_tick > Decimal("0"):  # Adjust SL if quantization pushed it through entry
                if signal == "BUY" and quantized_sl >= entry_price_estimate:
                    quantized_sl = ((entry_price_estimate - min_tick) / quantize_unit).quantize(
                        Decimal("1"), rounding=ROUND_DOWN
                    ) * quantize_unit
                elif signal == "SELL" and quantized_sl <= entry_price_estimate:
                    quantized_sl = ((entry_price_estimate + min_tick) / quantize_unit).quantize(
                        Decimal("1"), rounding=ROUND_UP
                    ) * quantize_unit

            final_tp, final_sl = quantized_tp, quantized_sl

            if (signal == "BUY" and final_tp <= entry_price_estimate) or (
                signal == "SELL" and final_tp >= entry_price_estimate
            ):
                self.logger.warning(
                    f"{signal} TP ({final_tp}) is not profitable vs entry ({entry_price_estimate}) for {self.symbol}. Setting TP to None."
                )
                final_tp = None

            if final_sl is not None and final_sl <= Decimal("0"):
                self.logger.error(f"Calculated SL ({final_sl}) is not positive for {self.symbol}. Setting SL to None.")
                final_sl = None
            if final_tp is not None and final_tp <= Decimal("0"):
                self.logger.warning(
                    f"Calculated TP ({final_tp}) is not positive for {self.symbol}. Setting TP to None."
                )
                final_tp = None

            tp_log = f"{final_tp:.{price_precision}f}" if final_tp else "None"
            sl_log = f"{final_sl:.{price_precision}f}" if final_sl else "None"
            self.logger.debug(
                f"Calculated TP/SL for {self.symbol} ({signal}): Entry={entry_price_estimate:.{price_precision}f}, "
                f"ATR={atr_value_final:.{price_precision + 2}f}, TP_raw={raw_tp:.{price_precision + 2}f}, SL_raw={raw_sl:.{price_precision + 2}f}, "
                f"TP_quant={tp_log}, SL_quant={sl_log}"
            )
            return entry_price_estimate, final_tp, final_sl

        except (InvalidOperation, TypeError, Exception) as e:
            self.logger.error(f"Error calculating TP/SL for {self.symbol} ({signal}): {e}", exc_info=True)
            return entry_price_estimate, None, None
