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
import datetime as dt  # Use dt alias for datetime module
import random
from decimal import Decimal, getcontext, InvalidOperation
from typing import Any, Dict, Optional, Type, List  # Added List, Union

# --- Attempt to import zoneinfo (Python 3.9+) ---
_ZoneInfo: Optional[Type[dt.tzinfo]] = None
_ZoneInfoNotFoundError: Optional[Type[Exception]] = None
try:
    from zoneinfo import ZoneInfo as _ZI, ZoneInfoNotFoundError as _ZINF

    _ZoneInfo = _ZI
    _ZoneInfoNotFoundError = _ZINF
    _zoneinfo_available = True
except ImportError:
    _zoneinfo_available = False
    # Fallback: Try importing pytz if zoneinfo is not available
    try:
        import pytz  # type: ignore

        # Create a wrapper that mimics ZoneInfo constructor
        class PytzZoneInfoWrapper:
            def __init__(self, key: str):
                try:
                    self._tz = pytz.timezone(key)
                except pytz.UnknownTimeZoneError:
                    raise ZoneInfoNotFoundError(
                        f"pytz: Unknown timezone '{key}'"
                    ) from None  # Mimic ZoneInfo error
                except Exception as e:
                    raise ZoneInfoNotFoundError(f"pytz error for '{key}': {e}") from e

            def __getattr__(self, name):
                return getattr(self._tz, name)  # Delegate methods

            def __str__(self):
                return self._tz.zone

        _ZoneInfo = PytzZoneInfoWrapper

        # Define a compatible error type for the except block in set_timezone
        class ZoneInfoNotFoundError(Exception):
            pass  # Define if pytz is used

        _ZoneInfoNotFoundError = ZoneInfoNotFoundError
        print(
            "Warning: 'zoneinfo' not found. Using 'pytz' for timezone support.",
            file=sys.stderr,
        )
    except ImportError:
        print(
            "Warning: Neither 'zoneinfo' nor 'pytz' found. Using basic UTC fallback only.",
            file=sys.stderr,
        )

        # Define a basic UTC tzinfo class if both fail
        class _UTCFallback(dt.tzinfo):
            def utcoffset(self, d: Optional[dt.datetime]) -> Optional[dt.timedelta]:
                return dt.timedelta(0)

            def dst(self, d: Optional[dt.datetime]) -> Optional[dt.timedelta]:
                return dt.timedelta(0)

            def tzname(self, d: Optional[dt.datetime]) -> Optional[str]:
                return "UTC"

            def __repr__(self) -> str:
                return "<UTCFallback tzinfo>"

        _ZoneInfo = _UTCFallback

        # Define a dummy error class for the except block consistency
        class ZoneInfoNotFoundError(Exception):
            pass

        _ZoneInfoNotFoundError = ZoneInfoNotFoundError


# --- Attempt to initialize Colorama ---
try:
    from colorama import Fore, Style, init

    init(autoreset=True)  # Initialize Colorama for cross-platform colored output
    # Color constants using Colorama
    NEON_GREEN = Fore.LIGHTGREEN_EX
    NEON_BLUE = Fore.LIGHTBLUE_EX
    NEON_PURPLE = Fore.LIGHTMAGENTA_EX
    NEON_YELLOW = Fore.LIGHTYELLOW_EX
    NEON_RED = Fore.LIGHTRED_EX
    NEON_CYAN = Fore.LIGHTCYAN_EX
    RESET_ALL_STYLE = Style.RESET_ALL
except ImportError:
    print(
        "Warning: 'colorama' not installed. Colored output will be disabled.",
        file=sys.stderr,
    )
    # Define fallback empty strings if colorama is not available
    NEON_GREEN = NEON_BLUE = NEON_PURPLE = NEON_YELLOW = NEON_RED = NEON_CYAN = (
        RESET_ALL_STYLE
    ) = ""

# --- Module-level logger ---
# It's generally better for utils to not log directly unless necessary,
# or to accept a logger instance. For now, keep it simple.
_module_logger = logging.getLogger(__name__)  # Use standard dunder name

# --- Decimal Context ---
# Set precision early, affects all subsequent Decimal operations in this context
try:
    getcontext().prec = 38  # High precision for financial calculations
except Exception as e:
    _module_logger.error(f"Failed to set Decimal precision: {e}")

# --- Configuration Constants ---
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
DEFAULT_TIMEZONE = "America/Chicago"  # Default timezone if not set in env or config

# --- API and Bot Behavior Constants ---
MAX_API_RETRIES = 3  # Default max retries (can be overridden by config)
RETRY_DELAY_SECONDS = 5.0  # Default base delay for non-rate-limit retries (use float)
MAX_RETRY_DELAY_SECONDS = (
    60.0  # Default max delay cap for exponential backoff (use float)
)
# POSITION_CONFIRM_DELAY_SECONDS is now defined in config_loader defaults

# --- Trading Constants ---
VALID_INTERVALS = [
    "1",
    "3",
    "5",
    "15",
    "30",
    "60",
    "120",
    "240",
    "360",
    "720",
    "D",
    "W",
    "M",
]  # Added more common intervals
CCXT_INTERVAL_MAP = {  # Map user-friendly intervals to CCXT timeframe codes
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "360": "6h",
    "720": "12h",
    "D": "1d",
    "W": "1w",
    "M": "1M",
}
FIB_LEVELS = [
    Decimal(str(f)) for f in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
]  # Standard Fibonacci levels as Decimals

# --- Indicator Default Periods (Centralized Source of Truth) ---
# These are loaded by config_loader and can be overridden in config.json
DEFAULT_INDICATOR_PERIODS = {
    # Moving Averages & Trend
    "ema_short_period": 9,
    "ema_long_period": 21,
    "sma_10_window": 10,
    "psar_initial_af": Decimal("0.02"),
    "psar_af_step": Decimal("0.02"),
    "psar_max_af": Decimal("0.2"),
    "vwap_anchor": "D",  # Default VWAP anchor (Daily) - Note: pandas_ta vwap might not use anchor directly
    # Oscillators
    "rsi_period": 14,
    "stoch_rsi_window": 14,
    "stoch_rsi_rsi_window": 14,  # Often same as rsi_period
    "stoch_rsi_k": 3,
    "stoch_rsi_d": 3,
    "cci_window": 20,
    "cci_constant": Decimal("0.015"),
    "williams_r_window": 14,
    "mfi_window": 14,
    "momentum_period": 10,  # Common default
    # Volatility
    "atr_period": 14,
    "bollinger_bands_period": 20,
    "bollinger_bands_std_dev": Decimal("2.0"),
    # Volume
    "volume_ma_period": 20,
    # Other / Strategy Specific
    "fibonacci_window": 50,  # Window for high/low used in Fib calculation
    # --- Thresholds moved to default_config in config_loader.py ---
    # These are strategy parameters rather than indicator calculation periods.
    # Keeping them in config_loader allows easier user tuning without touching utils.py.
    # Example threshold keys that would now live ONLY in config_loader's default_config:
    # "stoch_rsi_oversold_threshold": 25,
    # "stoch_rsi_overbought_threshold": 75,
    # "rsi_oversold_threshold": 30,
    # ... etc ...
    # "default_atr_percentage_of_price": Decimal("0.01")
}

# Global timezone object, lazily initialized by get_timezone()
_TIMEZONE: Optional[dt.tzinfo] = None


def _exponential_backoff(
    attempt: int,
    base_delay: float = RETRY_DELAY_SECONDS,  # Use constant from this module
    max_delay: float = MAX_RETRY_DELAY_SECONDS,  # Use constant from this module
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
        _module_logger.error("Exponential backoff attempt must be non-negative.")
        return base_delay  # Return base delay on invalid input
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
    Uses zoneinfo (Python 3.9+) or pytz if available, falls back to basic UTC.

    Args:
        tz_str (str): The timezone string (e.g., "America/Chicago", "UTC", "Europe/London").
    """
    global _TIMEZONE
    if _ZoneInfo is None:  # If neither zoneinfo nor pytz was available
        _module_logger.error(
            "No valid timezone implementation available. Cannot set timezone."
        )
        _TIMEZONE = None
        return
    try:
        _TIMEZONE = _ZoneInfo(
            tz_str
        )  # Use the detected ZoneInfo class (zoneinfo or pytz wrapper)
        _module_logger.info(
            f"Timezone successfully set using '{tz_str}'. Effective timezone: {str(_TIMEZONE)}"
        )
    except _ZoneInfoNotFoundError:  # Catch the specific error type
        _module_logger.error(
            f"Timezone '{tz_str}' not found by the available provider ({'zoneinfo' if _zoneinfo_available else 'pytz' if _ZoneInfo.__name__ == 'PytzZoneInfoWrapper' else 'fallback'}). Using UTC."
        )
        _TIMEZONE = _ZoneInfo("UTC")  # Fallback to UTC using the available provider
    except Exception as tz_err:
        _module_logger.error(
            f"Unexpected error loading timezone '{tz_str}'. Using UTC. Error: {tz_err}",
            exc_info=True,
        )
        try:
            _TIMEZONE = _ZoneInfo("UTC")  # Attempt UTC with available provider
        except:
            _TIMEZONE = (
                dt.timezone.utc
            )  # Absolute fallback if even UTC fails with provider


def get_timezone() -> dt.tzinfo:
    """
    Retrieves the global timezone object, initializing it from environment
    variable 'TIMEZONE' or DEFAULT_TIMEZONE if called for the first time.

    Returns:
        datetime.tzinfo: The configured timezone object (defaults to UTC on error).
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
        if _TIMEZONE is None:
            _module_logger.critical(
                "Timezone initialization failed critically. Forcing basic UTC."
            )
            _TIMEZONE = dt.timezone.utc  # Provide a basic UTC object if all else fails

    return _TIMEZONE


class SensitiveFormatter(logging.Formatter):
    """
    Custom logging formatter that redacts sensitive strings (API keys/secrets).
    Uses partial redaction for longer secrets if configured.
    """

    _secrets_to_redact: List[
        tuple[str, str]
    ] = []  # Store tuples of (original, placeholder)

    @classmethod
    def set_sensitive_data(cls, *args: Optional[str]) -> None:
        """
        Registers sensitive strings to be redacted in log messages.

        Args:
            *args (Optional[str]): Variable number of sensitive strings (e.g., API key, secret).
                                     None or empty strings are ignored.
        """
        cls._secrets_to_redact = []
        for s in args:
            if s and isinstance(s, str):  # Only process non-empty strings
                # Simple placeholder: "***KEY***" or "***SECRET***" etc.
                placeholder = f"***{s.__class__.__name__.upper()}***"
                # More secure placeholder (shows start/end, hides middle)
                # if len(s) > 8:
                #     placeholder = f"{s[:3]}...{s[-3:]}"
                # else: # Too short to partially redact meaningfully
                #     placeholder = "***REDACTED***"
                cls._secrets_to_redact.append((s, placeholder))
        if cls._secrets_to_redact:
            _module_logger.debug(
                f"Sensitive data registered for redaction: {len(cls._secrets_to_redact)} items."
            )
        else:
            _module_logger.debug("No sensitive data provided for redaction.")

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, applying redaction to the message."""
        # Format the message using the standard Formatter first
        formatted_message = super().format(record)
        # Apply redaction using the stored secrets
        temp_message = formatted_message
        try:
            # Use the class attribute directly if available
            secrets = getattr(SensitiveFormatter, "_secrets_to_redact", [])
            for original, placeholder in secrets:
                # Check if original string exists before replacing
                if original in temp_message:
                    temp_message = temp_message.replace(original, placeholder)
            return temp_message
        except Exception as e:
            # Avoid crashing the logger if redaction fails
            _module_logger.error(
                f"Error during log message redaction: {e}", exc_info=False
            )
            # Return the original formatted message or a generic error string
            return formatted_message  # Safest fallback


def get_price_precision(
    market_info: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> int:
    """
    Determines price precision (decimal places) from CCXT market info.
    Handles various ways precision might be specified (int, float/str tick size).

    Args:
        market_info: Market dictionary from CCXT.
        logger: Optional logger instance.

    Returns:
        Price precision (number of decimal places), defaulting to 8.
    """
    lg = logger or _module_logger
    symbol = market_info.get("symbol", "?")
    default_prec = 8  # Default if extraction fails

    try:
        precision_dict = market_info.get("precision", {})
        price_precision_info = precision_dict.get("price")

        if isinstance(price_precision_info, int):  # Explicit decimal places
            if price_precision_info >= 0:
                return price_precision_info
            else:
                lg.warning(
                    f"Negative integer precision {price_precision_info} for {symbol}. Ignoring."
                )
        elif price_precision_info is not None:  # Treat as tick size (str/float/Decimal)
            tick_size = Decimal(str(price_precision_info))
            if tick_size > Decimal("0"):
                # Calculate decimal places from tick size exponent
                # normalize() removes trailing zeros, exponent gives position of last digit
                return abs(tick_size.normalize().as_tuple().exponent)
            else:
                lg.warning(
                    f"Non-positive tick size '{price_precision_info}' in precision for {symbol}."
                )
        else:  # price precision info is None or missing
            lg.debug(f"No 'precision.price' info for {symbol}. Checking limits.")

        # Fallback 1: Check limits.price.min
        min_price_str = market_info.get("limits", {}).get("price", {}).get("min")
        if min_price_str:
            min_price_tick = Decimal(str(min_price_str))
            if min_price_tick > Decimal("0"):
                lg.debug(
                    f"Using precision from limits.price.min ({min_price_tick}) for {symbol}."
                )
                return abs(min_price_tick.normalize().as_tuple().exponent)

        # Fallback 2: If market info has 'decimal_places' or 'price_decimals' (less common CCXT fields)
        places = market_info.get("decimal_places") or market_info.get("price_decimals")
        if isinstance(places, int) and places >= 0:
            lg.debug(
                f"Using explicit decimal_places/price_decimals field for {symbol}: {places}"
            )
            return places

    except (InvalidOperation, TypeError, ValueError, AttributeError) as e:
        lg.warning(
            f"Could not reliably determine price precision for {symbol} from market info: {e}. Using default {default_prec}."
        )
    except Exception as e:
        lg.error(
            f"Unexpected error getting price precision for {symbol}: {e}", exc_info=True
        )

    lg.warning(
        f"Could not determine price precision for {symbol}, using default: {default_prec}"
    )
    return default_prec


def get_min_tick_size(
    market_info: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> Decimal:
    """
    Determines minimum price increment (tick size) as Decimal from market info.
    Handles various ways tick size might be specified.

    Args:
        market_info: Market dictionary from CCXT.
        logger: Optional logger instance.

    Returns:
        Minimum tick size as Decimal, defaulting to Decimal('1e-8').
    """
    lg = logger or _module_logger
    symbol = market_info.get("symbol", "?")
    default_tick = Decimal("1e-8")  # Match default precision

    try:
        precision_dict = market_info.get("precision", {})
        price_precision_info = precision_dict.get("price")

        # Case 1: Precision info is explicitly the tick size (float/str)
        if isinstance(price_precision_info, (str, float)):
            tick_size = Decimal(str(price_precision_info))
            if tick_size > 0:
                return tick_size
        # Case 2: Precision info is integer decimal places
        elif isinstance(price_precision_info, int) and price_precision_info >= 0:
            return Decimal("1e-" + str(price_precision_info))

        # Fallback 1: Check limits.price.min (often represents tick size)
        min_price_str = market_info.get("limits", {}).get("price", {}).get("min")
        if min_price_str:
            min_price_tick = Decimal(str(min_price_str))
            if min_price_tick > 0:
                return min_price_tick

        # Fallback 2: Derive from calculated precision places (less accurate but better than fixed default)
        price_prec_places = get_price_precision(market_info, lg)  # Use the helper
        derived_tick = Decimal("1e-" + str(price_prec_places))
        lg.debug(
            f"Derived min tick size {derived_tick} from precision places for {symbol}."
        )
        return derived_tick

    except (InvalidOperation, TypeError, ValueError, AttributeError) as e:
        lg.warning(
            f"Could not reliably determine min tick size for {symbol}: {e}. Using default {default_tick}."
        )
    except Exception as e:
        lg.error(
            f"Unexpected error getting min tick size for {symbol}: {e}", exc_info=True
        )

    lg.warning(
        f"Could not determine min tick size for {symbol}, using default: {default_tick}"
    )
    return default_tick


def format_signal(signal_text: Any, success: bool = True) -> str:
    """Formats trading signals (BUY, SELL, HOLD) or other statuses with color."""
    signal_str = str(signal_text).upper()
    color = RESET_ALL_STYLE  # Default to no color

    if success:
        if signal_str == "BUY":
            color = NEON_GREEN
        elif signal_str == "SELL":
            color = NEON_RED
        elif signal_str == "HOLD":
            color = NEON_YELLOW
        elif signal_str in ["ACTIVE", "CONFIRMED", "OK"]:
            color = NEON_GREEN
        elif signal_str in ["PENDING", "WAITING"]:
            color = NEON_YELLOW
        else:
            color = NEON_CYAN  # Neutral/Informational color
    else:  # Not success (error, failure, warning)
        color = NEON_RED

    return f"{color}{signal_text}{RESET_ALL_STYLE}"  # Use original casing in output


# --- End of utils.py ---
