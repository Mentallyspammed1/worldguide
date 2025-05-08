```python
# File: utils.py
"""
Utility functions, constants, and shared classes for the trading bot.

Includes:
- Configuration constants (file paths, default values).
- API behavior constants (retries, delays).
- Trading constants (intervals, Fibonacci levels).
- Default indicator periods.
- Timezone handling (using zoneinfo or fallback).
- Custom logging formatter for redacting sensitive data.
- Helper functions for market data (precision, tick size).
- Color constants for terminal output.
- Formatting utilities (signals).
- Exponential backoff utility.
"""

import logging
import os
import sys  # Added for sys.stderr
import datetime as dt
import random
from decimal import Decimal, getcontext, InvalidOperation
from typing import Any, Dict, Optional, Type, List, Union # Union was in original comment, added here

# --- Attempt to import zoneinfo (Python 3.9+) ---
# Define type aliases for clarity
_ZoneInfoType = Optional[Type[dt.tzinfo]]
_ZoneInfoErrorType = Optional[Type[Exception]] # Type of the exception class, not an instance

_ZoneInfo: _ZoneInfoType = None
_ZoneInfoNotFoundError: _ZoneInfoErrorType = None
_zoneinfo_available: bool = False

try:
    from zoneinfo import ZoneInfo as _ZI, ZoneInfoNotFoundError as _ZINF
    _ZoneInfo = _ZI
    _ZoneInfoNotFoundError = _ZINF # This is zoneinfo.ZoneInfoNotFoundError class
    _zoneinfo_available = True
except ImportError:
    _zoneinfo_available = False
    # Fallback: Try importing pytz if zoneinfo is not available
    try:
        import pytz  # type: ignore

        # Define a custom exception type for pytz, mimicking ZoneInfoNotFoundError
        class PytzCustomZoneInfoNotFoundError(Exception):
            pass
        _ZoneInfoNotFoundError = PytzCustomZoneInfoNotFoundError

        # Create a wrapper that mimics ZoneInfo constructor and behavior for pytz
        class PytzZoneInfoWrapper(dt.tzinfo):
            def __init__(self, key: str):
                try:
                    self._tz = pytz.timezone(key)
                except pytz.UnknownTimeZoneError as e:
                    raise PytzCustomZoneInfoNotFoundError(f"pytz: Unknown timezone '{key}'") from e
                except Exception as e: # Catch other pytz errors during timezone loading
                    raise PytzCustomZoneInfoNotFoundError(f"pytz: Error loading timezone '{key}': {e}") from e

            def utcoffset(self, d: Optional[dt.datetime]) -> Optional[dt.timedelta]:
                return self._tz.utcoffset(d)

            def dst(self, d: Optional[dt.datetime]) -> Optional[dt.timedelta]:
                return self._tz.dst(d)

            def tzname(self, d: Optional[dt.datetime]) -> Optional[str]:
                return self._tz.tzname(d)

            def __repr__(self) -> str:
                return f"<PytzZoneInfoWrapper zone='{self._tz.zone}'>"

            def __str__(self) -> str: # For user-friendly string representation
                return self._tz.zone

        _ZoneInfo = PytzZoneInfoWrapper
        # Use sys.stderr for import-time warnings as logger might not be configured yet
        print("Warning: 'zoneinfo' not found. Using 'pytz' for timezone support.", file=sys.stderr)
    except ImportError:
        print("Warning: Neither 'zoneinfo' nor 'pytz' found. Using basic UTC fallback only.", file=sys.stderr)

        # Define a compatible error type for the UTC fallback case
        class UTCFallbackZoneInfoNotFoundError(Exception):
            pass
        _ZoneInfoNotFoundError = UTCFallbackZoneInfoNotFoundError

        # Define a basic UTC tzinfo class if both zoneinfo and pytz fail
        class _UTCFallback(dt.tzinfo):
            _ZERO_OFFSET = dt.timedelta(0)

            def utcoffset(self, d: Optional[dt.datetime]) -> Optional[dt.timedelta]:
                return self._ZERO_OFFSET

            def dst(self, d: Optional[dt.datetime]) -> Optional[dt.timedelta]:
                return self._ZERO_OFFSET # UTC has no DST

            def tzname(self, d: Optional[dt.datetime]) -> Optional[str]:
                return "UTC"

            def __repr__(self) -> str:
                return "<UTCFallback tzinfo>"

            def __str__(self) -> str:
                return "UTC"

        _ZoneInfo = _UTCFallback


# --- Attempt to initialize Colorama ---
try:
    from colorama import Fore, Style, init
    init(autoreset=True)  # Initialize Colorama for cross-platform colored output
    NEON_GREEN: str = Fore.LIGHTGREEN_EX
    NEON_BLUE: str = Fore.LIGHTBLUE_EX
    NEON_PURPLE: str = Fore.LIGHTMAGENTA_EX
    NEON_YELLOW: str = Fore.LIGHTYELLOW_EX
    NEON_RED: str = Fore.LIGHTRED_EX
    NEON_CYAN: str = Fore.LIGHTCYAN_EX
    RESET_ALL_STYLE: str = Style.RESET_ALL
except ImportError:
    print("Warning: 'colorama' not installed. Colored output will be disabled.", file=sys.stderr)
    # Define fallback empty strings if colorama is not available
    NEON_GREEN = NEON_BLUE = NEON_PURPLE = NEON_YELLOW = NEON_RED = NEON_CYAN = RESET_ALL_STYLE = ""

# --- Module-level logger ---
_module_logger = logging.getLogger(__name__)

# --- Decimal Context ---
# Set precision early; affects all Decimal operations in the current context.
try:
    getcontext().prec = 38  # High precision suitable for financial calculations
except Exception as e: # Catch generic Exception as various issues can occur
    _module_logger.error(f"Failed to set Decimal precision: {e}", exc_info=True)

# --- Configuration Constants ---
CONFIG_FILE: str = "config.json"
LOG_DIRECTORY: str = "bot_logs"
DEFAULT_TIMEZONE: str = "America/Chicago"  # Default if not in env or config

# --- API and Bot Behavior Constants ---
MAX_API_RETRIES: int = 3
RETRY_DELAY_SECONDS: float = 5.0
MAX_RETRY_DELAY_SECONDS: float = 60.0

# --- Trading Constants ---
VALID_INTERVALS: List[str] = [
    "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"
]
CCXT_INTERVAL_MAP: Dict[str, str] = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "360": "6h", "720": "12h",
    "D": "1d", "W": "1w", "M": "1M", # CCXT uses '1M' for month
}
FIB_LEVELS: List[Decimal] = [
    Decimal(str(f_val)) for f_val in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
]

# --- Indicator Default Periods (Centralized Source of Truth) ---
DEFAULT_INDICATOR_PERIODS: Dict[str, Union[int, Decimal, str]] = {
    # Moving Averages & Trend
    "ema_short_period": 9,
    "ema_long_period": 21,
    "sma_10_window": 10,
    "psar_initial_af": Decimal("0.02"),
    "psar_af_step": Decimal("0.02"),
    "psar_max_af": Decimal("0.2"),
    "vwap_anchor": "D",  # Default VWAP anchor (Daily)
    # Oscillators
    "rsi_period": 14,
    "stoch_rsi_window": 14,
    "stoch_rsi_rsi_window": 14,
    "stoch_rsi_k": 3,
    "stoch_rsi_d": 3,
    "cci_window": 20,
    "cci_constant": Decimal("0.015"),
    "williams_r_window": 14,
    "mfi_window": 14,
    "momentum_period": 10,
    # Volatility
    "atr_period": 14,
    "bollinger_bands_period": 20,
    "bollinger_bands_std_dev": Decimal("2.0"),
    # Volume
    "volume_ma_period": 20,
    # Other / Strategy Specific
    "fibonacci_window": 50, # Window for high/low in Fibonacci calculation
}

# Global timezone object, lazily initialized by get_timezone()
_TIMEZONE: Optional[dt.tzinfo] = None


def _exponential_backoff(
    attempt: int,
    base_delay: float = RETRY_DELAY_SECONDS,
    max_delay: float = MAX_RETRY_DELAY_SECONDS,
    jitter: bool = True,
) -> float:
    """
    Calculates the delay for an exponential backoff strategy with jitter.

    Args:
        attempt (int): The current retry attempt number (0-indexed).
        base_delay (float): The starting delay in seconds.
        max_delay (float): The maximum delay cap in seconds.
        jitter (bool): Whether to add random jitter (+/- 30%).

    Returns:
        float: The calculated delay in seconds.
    """
    if attempt < 0:
        _module_logger.error(
            "Exponential backoff attempt must be non-negative. Using base_delay."
        )
        return base_delay
    try:
        # Calculate exponential delay: base * 2^attempt
        delay = base_delay * (2**attempt)
        # Apply jitter: random value between 70% and 130% of the calculated delay
        if jitter:
            delay = random.uniform(delay * 0.7, delay * 1.3)
        # Cap the delay at the maximum allowed value
        return min(delay, max_delay)
    except OverflowError:
        # Handle potential overflow if attempt number is excessively large
        _module_logger.warning(
            f"Exponential backoff calculation overflowed for attempt {attempt}. Using max_delay."
        )
        return max_delay


def set_timezone(tz_str: str) -> None:
    """
    Sets the global timezone object used by the application.
    Uses zoneinfo (Python 3.9+), pytz if available, or falls back to basic UTC.

    Args:
        tz_str (str): The timezone string (e.g., "America/Chicago", "UTC").
    """
    global _TIMEZONE

    if _ZoneInfo is None or _ZoneInfoNotFoundError is None:
        # This state indicates a fundamental issue with the timezone provider setup.
        _module_logger.critical(
            "Timezone provider system not properly initialized. Forcing basic UTC."
        )
        _TIMEZONE = dt.timezone.utc
        return

    if _ZoneInfo is _UTCFallback:  # Check if we are using the most basic fallback
        if tz_str.upper() == "UTC":
            _TIMEZONE = _UTCFallback() # Instantiate the fallback UTC tzinfo
            _module_logger.info("Timezone set to UTC (using UTCFallback provider).")
        else:
            _module_logger.warning(
                f"Timezone '{tz_str}' requested, but only UTC is supported with UTCFallback. Using UTC."
            )
            _TIMEZONE = _UTCFallback()
        return

    # Proceed with zoneinfo or PytzZoneInfoWrapper
    try:
        # _ZoneInfo is known to be a Type[dt.tzinfo] here, Pylance might need help.
        _TIMEZONE = _ZoneInfo(tz_str) # type: ignore
        provider_name = "zoneinfo" if _zoneinfo_available else "pytz"
        _module_logger.info(
            f"Timezone successfully set to '{tz_str}' using {provider_name}. Effective timezone: {str(_TIMEZONE)}"
        )
    except _ZoneInfoNotFoundError as e: # Catch the aliased/defined error for the current provider
        provider_name = "zoneinfo" if _zoneinfo_available else "pytz"
        _module_logger.error(
            f"Timezone '{tz_str}' not found by {provider_name}: {e}. "
            f"Falling back to UTC using {provider_name}."
        )
        try:
            _TIMEZONE = _ZoneInfo("UTC") # type: ignore
        except Exception as e_utc: # If even "UTC" fails with the current provider
            _module_logger.critical(
                f"Failed to set timezone to UTC using {provider_name}: {e_utc}. "
                "Using basic dt.timezone.utc as final fallback.", exc_info=True
            )
            _TIMEZONE = dt.timezone.utc # Absolute fallback
    except Exception as tz_err: # Catch any other unexpected errors
        _module_logger.error(
            f"Unexpected error loading timezone '{tz_str}'. Using basic dt.timezone.utc. Error: {tz_err}",
            exc_info=True
        )
        _TIMEZONE = dt.timezone.utc


def get_timezone() -> dt.tzinfo:
    """
    Retrieves the global timezone object, initializing it from environment
    variable 'TIMEZONE' or DEFAULT_TIMEZONE if called for the first time.

    Returns:
        datetime.tzinfo: The configured timezone object. Defaults to a UTC
                         implementation if initialization fails.
    """
    global _TIMEZONE
    if _TIMEZONE is None:
        # Initialize timezone if it hasn't been set yet
        env_tz = os.getenv("TIMEZONE")
        chosen_tz_str = env_tz if env_tz else DEFAULT_TIMEZONE
        _module_logger.info(
            f"Initializing timezone. Env='{env_tz}', Default='{DEFAULT_TIMEZONE}', Chosen='{chosen_tz_str}'"
        )
        set_timezone(chosen_tz_str)
        # Ensure _TIMEZONE is not None after initialization attempt
        if _TIMEZONE is None: # Should be set by set_timezone, even to a fallback
            _module_logger.critical(
                "Timezone initialization failed critically. Forcing basic dt.timezone.utc."
            )
            _TIMEZONE = dt.timezone.utc # Final safeguard
    return _TIMEZONE


class SensitiveFormatter(logging.Formatter):
    """
    Custom logging formatter that redacts sensitive strings (e.g., API keys/secrets).
    """
    # Class attribute to store (original_secret, placeholder_text) tuples
    _secrets_to_redact: List[tuple[str, str]] = []

    @classmethod
    def set_sensitive_data(cls, *args: Optional[str]) -> None:
        """
        Registers sensitive strings to be redacted in log messages.
        Existing secrets are cleared and replaced.

        Args:
            *args (Optional[str]): Variable number of sensitive strings.
                                     None, empty, or very short strings are ignored.
        """
        current_secrets: List[tuple[str, str]] = []
        for s_val in args:
            # Process only non-empty strings of reasonable length
            if s_val and isinstance(s_val, str) and len(s_val) > 3:
                placeholder = "***REDACTED***"
                # Example for partial redaction (can be enabled if desired):
                # if len(s_val) > 8:
                #     placeholder = f"{s_val[:3]}...{s_val[-3:]}"
                # else:
                #     placeholder = "***REDACTED***"
                current_secrets.append((s_val, placeholder))

        cls._secrets_to_redact = current_secrets # Atomically update the list

        if cls._secrets_to_redact:
            _module_logger.debug(
                f"Sensitive data registered for redaction: {len(cls._secrets_to_redact)} items."
            )
        else:
            _module_logger.debug(
                "No sensitive data (or only short/empty strings) provided for redaction."
            )

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, applying redaction to the message."""
        # Format the message using the standard Formatter first
        formatted_message = super().format(record)

        # Access class attribute via type(self) for instance methods
        secrets_list = type(self)._secrets_to_redact
        if not secrets_list:  # Optimization: No secrets to redact
            return formatted_message

        temp_message = formatted_message
        try:
            for original, placeholder in secrets_list:
                # Check if original string exists before replacing to avoid unnecessary operations
                if original in temp_message:
                    temp_message = temp_message.replace(original, placeholder)
            return temp_message
        except Exception as e:
            # Avoid crashing the logger if redaction fails.
            # Log to a specific sub-logger to prevent potential recursion
            # if _module_logger itself uses this SensitiveFormatter.
            logging.getLogger(__name__ + ".SensitiveFormatter").error(
                f"Error during log message redaction: {e}", exc_info=False # exc_info=False to keep it concise
            )
            return formatted_message # Return original formatted message on error


def get_price_precision(
    market_info: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> int:
    """
    Determines price precision (number of decimal places) from CCXT market info.
    Handles various ways precision might be specified.

    Args:
        market_info: Market dictionary from CCXT.
        logger: Optional logger instance for context-specific logging.

    Returns:
        Price precision (number of decimal places), defaulting to 8.
    """
    lg = logger or _module_logger
    symbol = market_info.get("symbol", "N/A") # Use N/A if symbol is not present
    default_precision = 8  # Common default if extraction fails

    try:
        precision_data = market_info.get("precision", {})
        price_precision_value = precision_data.get("price")

        # Case 1: 'precision.price' is an integer (explicit decimal places)
        if isinstance(price_precision_value, int):
            if price_precision_value >= 0:
                return price_precision_value
            lg.warning(
                f"[{symbol}] Negative integer precision '{price_precision_value}' found, ignoring."
            )
        # Case 2: 'precision.price' is float/str (usually tick size)
        elif price_precision_value is not None:
            try:
                tick_size = Decimal(str(price_precision_value))
                if tick_size > Decimal("0"):
                    # Calculate decimal places from tick size's exponent
                    return abs(tick_size.normalize().as_tuple().exponent)
                lg.warning(f"[{symbol}] Non-positive tick size '{price_precision_value}' in precision data.")
            except InvalidOperation:
                lg.warning(
                    f"[{symbol}] Invalid value '{price_precision_value}' for tick size in precision data."
                )
        else: # price_precision_value is None or 'price' key is missing
            lg.debug(f"[{symbol}] No 'precision.price' info. Checking other fields.")

        # Fallback 1: Check 'limits.price.min' (often reflects tick size or minimum price unit)
        min_price_str = market_info.get("limits", {}).get("price", {}).get("min")
        if min_price_str:
            try:
                min_price_tick = Decimal(str(min_price_str))
                if min_price_tick > Decimal("0"):
                    lg.debug(f"[{symbol}] Using precision from limits.price.min: {min_price_tick}")
                    return abs(min_price_tick.normalize().as_tuple().exponent)
            except InvalidOperation:
                lg.debug(f"[{symbol}] Invalid value for limits.price.min: '{min_price_str}', skipping.")

        # Fallback 2: Check for less common 'decimal_places' or 'price_decimals' fields
        explicit_places = market_info.get("decimal_places") or market_info.get("price_decimals")
        if isinstance(explicit_places, int) and explicit_places >= 0:
            lg.debug(f"[{symbol}] Using explicit 'decimal_places'/'price_decimals' field: {explicit_places}")
            return explicit_places

    except (TypeError, ValueError, AttributeError) as e: # Common errors during dict access or conversion
        lg.warning(
            f"[{symbol}] Error determining price precision from market info: {e}. Using default {default_precision}."
        )
    except Exception as e: # Catch-all for truly unexpected issues
        lg.error(
            f"[{symbol}] Unexpected error getting price precision: {e}. Using default {default_precision}.",
            exc_info=True
        )

    lg.warning(f"[{symbol}] Could not determine price precision, using default: {default_precision}")
    return default_precision


def get_min_tick_size(
    market_info: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> Decimal:
    """
    Determines minimum price increment (tick size) as Decimal from market info.

    Args:
        market_info: Market dictionary from CCXT.
        logger: Optional logger instance.

    Returns:
        Minimum tick size as Decimal, defaulting to Decimal('1e-8').
    """
    lg = logger or _module_logger
    symbol = market_info.get("symbol", "N/A")
    default_tick_size = Decimal("1e-8") # Corresponds to 8 decimal places

    try:
        precision_data = market_info.get("precision", {})
        price_precision_value = precision_data.get("price")

        # Case 1: 'precision.price' is explicitly the tick size (float/str/Decimal)
        if isinstance(price_precision_value, (str, float, Decimal)):
            try:
                tick_size = Decimal(str(price_precision_value))
                if tick_size > Decimal("0"):
                    return tick_size
                lg.warning(f"[{symbol}] Non-positive tick size '{price_precision_value}' in precision data.")
            except InvalidOperation:
                lg.warning(
                    f"[{symbol}] Invalid value '{price_precision_value}' for tick size in precision data."
                )
        # Case 2: 'precision.price' is integer (decimal places), convert to tick size
        elif isinstance(price_precision_value, int) and price_precision_value >= 0:
            return Decimal("1e-" + str(price_precision_value))

        # Fallback 1: Check 'limits.price.min' (often represents the tick size directly)
        min_price_str = market_info.get("limits", {}).get("price", {}).get("min")
        if min_price_str:
            try:
                min_price_tick = Decimal(str(min_price_str))
                if min_price_tick > Decimal("0"):
                    return min_price_tick
            except InvalidOperation:
                lg.debug(f"[{symbol}] Invalid value for limits.price.min: '{min_price_str}', skipping.")

        # Fallback 2: Derive from calculated precision places (less direct, but an estimate)
        # This is useful if 'precision.price' is missing or not directly the tick size.
        price_decimal_places = get_price_precision(market_info, lg) # Reuse precision logic
        derived_tick = Decimal("1e-" + str(price_decimal_places))
        lg.debug(f"[{symbol}] Derived min tick size {derived_tick} from calculated precision places.")
        return derived_tick

    except (TypeError, ValueError, AttributeError) as e:
        lg.warning(
            f"[{symbol}] Error determining min tick size: {e}. Using default {default_tick_size}."
        )
    except Exception as e:
        lg.error(
            f"[{symbol}] Unexpected error getting min tick size: {e}. Using default {default_tick_size}.",
            exc_info=True
        )

    lg.warning(f"[{symbol}] Could not determine min tick size, using default: {default_tick_size}")
    return default_tick_size


def format_signal(signal_text: Any, success: bool = True) -> str:
    """Formats trading signals (BUY, SELL, HOLD) or other statuses with color."""
    text_to_format = str(signal_text)  # Ensure input is string for consistent handling
    signal_str_upper = text_to_format.upper() # Use uppercase for comparisons
    color = RESET_ALL_STYLE  # Default to no color or reset

    if success:
        if signal_str_upper == "BUY":
            color = NEON_GREEN
        elif signal_str_upper == "SELL":
            color = NEON_RED
        elif signal_str_upper == "HOLD":
            color = NEON_YELLOW
        elif signal_str_upper in ["ACTIVE", "CONFIRMED", "OK", "SUCCESS"]:
            color = NEON_GREEN
        elif signal_str_upper in ["PENDING", "WAITING", "NEUTRAL"]:
            color = NEON_YELLOW
        else: # Default color for other successful or informational messages
            color = NEON_CYAN
    else:  # Not success (error, failure, warning)
        color = NEON_RED

    return f"{color}{text_to_format}{RESET_ALL_STYLE}" # Use original casing in output


# --- End of utils.py ---
```