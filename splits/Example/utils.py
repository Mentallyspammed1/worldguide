# File: utils.py
# -*- coding: utf-8 -*-

"""
Utility Functions and Placeholders for Bybit Trading Bot Helpers.

This module provides various helper functions commonly needed when interacting
with the Bybit exchange via the CCXT library. It includes functions for
safe type conversions, formatting, market validation, and placeholders
for features like API call retries and SMS alerts.

Note: Several functions, particularly the retry decorator and SMS alert,
are implemented as placeholders or simulations. They MUST be replaced with
robust implementations in a production environment.
"""

import logging
import sys
from decimal import Decimal, InvalidOperation, getcontext
from typing import (
    Optional,
    Dict,
    Tuple,
    Any,
    Literal,
    TypeVar,
    Callable,
    Union,
    TYPE_CHECKING,
)
from functools import wraps

# Conditional import for type checking to avoid circular dependencies
if TYPE_CHECKING:
    from config import Config  # Assuming 'config.py' defines a Config class

# --- Dependencies Handling ---
try:
    import ccxt

    # Specific check for bybit class to ensure it's available
    if not hasattr(ccxt, "bybit"):
        print(
            "Error: ccxt library is installed, but 'bybit' exchange class is missing."
        )
        print("Ensure you have a compatible version of ccxt.")
        sys.exit(1)
except ImportError:
    print("Error: CCXT library not found.")
    print("Please install it using: pip install ccxt")
    sys.exit(1)

try:
    from colorama import Fore, Style, Back, init as colorama_init

    colorama_init(autoreset=True)  # Initialize colorama for automatic style reset
except ImportError:
    print("Warning: colorama library not found. Log messages will not be colored.")
    print("Install it using: pip install colorama")

    # Create a dummy class that returns empty strings for color attributes
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            """Return empty string for any color attribute."""
            return ""

    Fore = Style = Back = DummyColor()

# --- Constants ---
DEFAULT_DECIMAL_PRECISION = 28
# Common Bybit V5 categories
BybitV5Category = Literal["linear", "inverse", "spot", "option"]
# Common CCXT market types
MarketType = Literal["swap", "future", "spot", "option"]

# --- Configuration ---
# Set Decimal context precision for calculations
getcontext().prec = DEFAULT_DECIMAL_PRECISION
# Consider rounding mode if needed, e.g., getcontext().rounding = ROUND_HALF_UP

# --- Logging Setup ---
# Configure logging for this specific module
# Note: The actual handler configuration (e.g., StreamHandler, FileHandler)
# should ideally be done in the main application entry point.
logger = logging.getLogger(__name__)
# Example basic config if no other logging is set up (useful for standalone testing)
# if not logging.getLogger().hasHandlers():
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Critical Placeholders ---

# Define a generic TypeVar for decorator return types
T = TypeVar("T")


# Placeholder: Retry Decorator
# IMPORTANT: Replace this with a robust implementation using libraries like 'tenacity'
# or a custom solution handling specific CCXT/network exceptions.
def retry_api_call(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions_to_retry: Tuple[type[Exception], ...] = (
        ccxt.NetworkError,
        ccxt.ExchangeError,
        ccxt.RateLimitExceeded,
    ),
    **decorator_kwargs: Any,  # Allow passing extra args if needed by real decorator
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    **PLACEHOLDER** Decorator to retry a function call on specific exceptions.

    This is a non-functional placeholder. The actual implementation MUST:
    1. Catch specific exceptions (e.g., RateLimitExceeded, NetworkError).
    2. Implement exponential backoff with jitter.
    3. Log retry attempts and failures clearly.
    4. Respect the `max_retries` limit.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Delay before the first retry (in seconds).
        backoff_factor: Multiplier for the delay increase on each retry.
        exceptions_to_retry: Tuple of exception types that trigger a retry.
        **decorator_kwargs: Additional arguments for potential real decorator.

    Returns:
        A decorator function.
    """
    logger.warning(
        f"{Fore.YELLOW}Using PLACEHOLDER retry_api_call decorator. "
        f"NO RETRY LOGIC IS ACTIVE. Implement a real retry mechanism!{Style.RESET_ALL}"
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Placeholder: Just call the function directly without retry logic.
            # logger.debug(f"Placeholder retry decorator executing for {func.__name__}")
            try:
                return func(*args, **kwargs)
            except Exception:
                # logger.error(f"Placeholder retry caught unhandled exception in {func.__name__}: {e}", exc_info=False) # Set exc_info=True for full traceback
                # In a real implementation, check if e is in exceptions_to_retry
                # and implement the retry loop here.
                raise  # Re-raise immediately in this placeholder

        return wrapper

    return decorator


# Placeholder: SMS Alert Function
# IMPORTANT: Replace this with a real SMS sending implementation (e.g., Twilio, Termux API).
def send_sms_alert(message: str, config: Optional["Config"] = None) -> bool:
    """
    **PLACEHOLDER/SIMULATION** Sends an SMS alert.

    This function simulates sending an SMS. In a real application, replace
    this with code that interacts with an SMS gateway or service (like Twilio)
    or uses device-specific APIs (like Termux:API on Android).

    Reads configuration (ENABLE_SMS_ALERTS, SMS_RECIPIENT_NUMBER, SMS_TIMEOUT_SECONDS)
    either from the passed `config` object or by importing the default `Config`.

    Args:
        message: The text message content to send.
        config: Optional Config object containing SMS settings. If None, imports
                the default Config.

    Returns:
        True if the simulation was successful (or if SMS is disabled),
        False if SMS is enabled but improperly configured (e.g., no recipient).
    """
    # Import config locally only if needed, reducing import coupling
    if config is None:
        try:
            from config import Config as ConfigClass

            cfg: "Config" = ConfigClass()
        except ImportError:
            logger.error("send_sms_alert: Default Config class could not be imported.")
            return False
    else:
        cfg = config

    is_enabled: bool = getattr(cfg, "ENABLE_SMS_ALERTS", False)

    if not is_enabled:
        # logger.debug(f"SMS alert suppressed (disabled): {message[:50]}...") # Log less verbosely if disabled
        return True  # Consider disabled as a "success" in terms of execution flow

    recipient: Optional[str] = getattr(cfg, "SMS_RECIPIENT_NUMBER", None)
    timeout: int = getattr(cfg, "SMS_TIMEOUT_SECONDS", 30)

    if not recipient:
        logger.warning(
            f"{Fore.YELLOW}SMS alerts enabled, but no recipient number found in config (SMS_RECIPIENT_NUMBER).{Style.RESET_ALL}"
        )
        return False

    # --- Simulation Logic ---
    logger.info(
        f"{Back.YELLOW}{Fore.BLACK}--- SIMULATING SMS Alert ---{Style.RESET_ALL}"
    )
    logger.info(f"Recipient: {recipient}")
    logger.info(f"Timeout:   {timeout}s")
    logger.info(f"Message:   {message}")
    print(
        f"{Back.GREEN}{Fore.BLACK}SIMULATED SMS: {message}{Style.RESET_ALL}"
    )  # Also print to console for visibility

    # Placeholder for actual API call (e.g., Termux, Twilio)
    # try:
    #     # Example: import os; os.system(f"termux-sms-send -n {recipient} '{message}'")
    #     # Example: from twilio.rest import Client; client.messages.create(...)
    #     pass # Replace with actual implementation
    #     logger.info(f"SMS simulation successful.")
    #     return True
    # except Exception as e:
    #     logger.error(f"Failed to send simulated SMS: {e}", exc_info=True)
    #     return False
    # --- End Simulation ---

    return True  # Simulation considered successful


# --- Core Utility Functions ---


def safe_decimal_conversion(
    value: Any, default: Optional[Decimal] = None
) -> Optional[Decimal]:
    """
    Safely converts various types (str, int, float) to a Decimal object.

    Handles None input and potential conversion errors gracefully.

    Args:
        value: The value to convert (string, number, or None).
        default: The value to return if conversion fails or input is None.
                 Defaults to None.

    Returns:
        The Decimal representation of the value, or the default value on failure.

    Example:
        >>> safe_decimal_conversion("123.45")
        Decimal('123.45')
        >>> safe_decimal_conversion(100)
        Decimal('100')
        >>> safe_decimal_conversion("invalid", default=Decimal('0'))
        Decimal('0')
        >>> safe_decimal_conversion(None)
        None
    """
    if value is None:
        return default
    try:
        # Convert to string first to handle floats more predictably
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        # logger.debug(f"safe_decimal_conversion failed for value '{value}' (type: {type(value)}): {e}")
        return default


def format_price(
    exchange: ccxt.Exchange, symbol: str, price: Any, fallback_precision: int = 4
) -> str:
    """
    Formats a price value according to the exchange's precision rules for a symbol.

    Uses `exchange.price_to_precision()`. Provides fallbacks if precision
    information is unavailable or conversion fails.

    Args:
        exchange: An initialized ccxt Exchange object.
        symbol: The market symbol (e.g., 'BTC/USDT').
        price: The price value to format (can be Decimal, float, str, int).
        fallback_precision: Number of decimal places for fallback formatting
                             if exchange precision fails.

    Returns:
        A string representation of the formatted price, or "N/A" if input is None.
    """
    if price is None:
        return "N/A"

    try:
        # Attempt to use CCXT's built-in precision formatting
        return exchange.price_to_precision(symbol, price)
    except (ccxt.ExchangeError, ccxt.BadSymbol, AttributeError, TypeError, ValueError):
        # Log the reason for fallback if needed
        # logger.warning(f"Could not format price using exchange precision for {symbol}: {e}. Using fallback.")
        # Fallback 1: Try converting to Decimal and formatting
        try:
            price_decimal = safe_decimal_conversion(price)
            if price_decimal is not None:
                # Create format string dynamically, e.g., "{:.4f}"
                format_spec = f":.{fallback_precision}f"
                return f"{price_decimal:{format_spec}}"
            else:
                raise ValueError("Decimal conversion failed in fallback.")
        except (InvalidOperation, ValueError, TypeError):
            # Fallback 2: Convert to string as a last resort
            logger.warning(
                f"Fallback decimal formatting failed for price '{price}' of {symbol}. Returning raw string."
            )
            return str(price)


def format_amount(
    exchange: ccxt.Exchange, symbol: str, amount: Any, fallback_precision: int = 8
) -> str:
    """
    Formats an amount (quantity) value according to the exchange's precision rules.

    Uses `exchange.amount_to_precision()`. Provides fallbacks similar to format_price.

    Args:
        exchange: An initialized ccxt Exchange object.
        symbol: The market symbol (e.g., 'BTC/USDT').
        amount: The amount value to format.
        fallback_precision: Number of decimal places for fallback formatting.

    Returns:
        A string representation of the formatted amount, or "N/A" if input is None.
    """
    if amount is None:
        return "N/A"

    try:
        # Attempt to use CCXT's built-in precision formatting
        return exchange.amount_to_precision(symbol, amount)
    except (ccxt.ExchangeError, ccxt.BadSymbol, AttributeError, TypeError, ValueError):
        # logger.warning(f"Could not format amount using exchange precision for {symbol}: {e}. Using fallback.")
        # Fallback 1: Try converting to Decimal and formatting
        try:
            amount_decimal = safe_decimal_conversion(amount)
            if amount_decimal is not None:
                format_spec = f":.{fallback_precision}f"
                return f"{amount_decimal:{format_spec}}"
            else:
                raise ValueError("Decimal conversion failed in fallback.")
        except (InvalidOperation, ValueError, TypeError):
            # Fallback 2: Convert to string
            logger.warning(
                f"Fallback decimal formatting failed for amount '{amount}' of {symbol}. Returning raw string."
            )
            return str(amount)


def format_order_id(order_id: Optional[Union[str, int]]) -> str:
    """
    Formats an order ID for concise logging, typically showing the last few characters.

    Args:
        order_id: The order ID (string or integer) or None.

    Returns:
        A shortened string representation of the order ID (e.g., last 6 chars),
        the full ID if short, or "N/A" if None.
    """
    if order_id is None:
        return "N/A"
    id_str = str(order_id)
    # Return last 6 characters if long enough, otherwise the whole ID
    return id_str[-6:] if len(id_str) > 6 else id_str


# --- Internal Helper Functions ---


def _get_v5_category(market: Dict[str, Any]) -> Optional[BybitV5Category]:
    """
    Internal helper to determine the Bybit V5 API category from a CCXT market object.

    Bybit's V5 API requires specifying a category ('linear', 'inverse', 'spot', 'option')
    for many endpoints. This function attempts to deduce it from market info.

    Args:
        market: The market dictionary object obtained from `exchange.market(symbol)`.

    Returns:
        The determined category ('linear', 'inverse', 'spot', 'option') or None
        if the category cannot be reliably determined.
    """
    if not market:
        return None

    # 1. Check explicit flags (most reliable if present)
    if market.get("linear"):
        return "linear"
    if market.get("inverse"):
        return "inverse"
    if market.get("spot"):
        return "spot"
    if market.get("option"):
        return "option"

    # 2. Fallback based on market 'type' (less reliable, makes assumptions)
    market_type: Optional[MarketType] = market.get("type")
    if market_type == "spot":
        return "spot"
    if market_type == "option":
        return "option"
    if market_type == "swap" or market_type == "future":
        # Assumption: If not explicitly inverse, assume linear for swaps/futures.
        # This might be incorrect for some exchanges or symbols.
        # Bybit usually distinguishes via 'linear'/'inverse' flags or symbol naming conventions.
        # A safer approach might involve checking settle currency (e.g., USDT vs BTC).
        settle_currency = market.get("settle")
        if settle_currency == market.get(
            "base"
        ):  # e.g., settle=BTC for BTC/USD inverse
            return "inverse"
        else:  # Assume linear if settle is quote (USDT, USDC) or unknown
            return "linear"

    # 3. If type is unknown or doesn't fit, log a warning
    logger.warning(
        f"_get_v5_category: Could not determine category for market: "
        f"{market.get('symbol')}. Market details: type='{market_type}', "
        f"linear={market.get('linear')}, inverse={market.get('inverse')}"
    )
    return None


# --- Market Validation ---


# No API call needed here *if* markets are already loaded, so no retry decorator needed directly.
# However, the `load_markets` call *inside* this function might benefit from retries
# if it were separated or if this function handled the network errors more granularly.
def validate_market(
    exchange: ccxt.bybit,
    symbol: str,
    config: Optional["Config"] = None,
    expected_type: Optional[MarketType] = None,
    expected_logic: Optional[
        BybitV5Category
    ] = None,  # For Bybit V5 category ('linear', 'inverse')
    check_active: bool = True,
    require_contract: bool = True,
    reload_markets: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Validates if a symbol exists, is active, and optionally matches expected criteria.

    Checks include:
    - Symbol existence on the exchange.
    - Market active status (optional).
    - Market type (e.g., 'swap', 'spot') (optional).
    - Contract status (e.g., is it a derivative?) (optional).
    - Bybit V5 category ('linear' or 'inverse') for contracts (optional).

    Loads markets using `exchange.load_markets()` if they haven't been loaded yet
    or if `reload_markets` is True.

    Args:
        exchange: An initialized ccxt.bybit exchange instance.
        symbol: The trading symbol to validate (e.g., "BTC/USDT:USDT").
        config: Optional Config object. Used to get default expected type/logic if
                specific arguments are None.
        expected_type: If provided, check if market['type'] matches this value.
        expected_logic: If provided (and market is a contract), check if the
                       deduced V5 category ('linear'/'inverse') matches.
        check_active: If True, check if `market['active']` is True. Logs a warning
                      if inactive but doesn't fail validation by default.
        require_contract: If True, check if `market['contract']` is True. Fails
                          validation if it's not a contract market.
        reload_markets: If True, force reloading markets even if already loaded.

    Returns:
        The market dictionary if validation passes, otherwise None.
        Logs errors or warnings detailing the validation outcome.
    """
    func_name = "validate_market"  # For logging context

    # Import default config if specific config is not provided
    if config is None:
        try:
            from config import Config as ConfigClass

            cfg: "Config" = ConfigClass()
        except ImportError:
            logger.warning(
                f"[{func_name}] Default Config class not found, cannot get default expectations."
            )
            cfg = None  # type: ignore # Assign None explicitly
    else:
        cfg = config

    # Determine effective expectations (use args if provided, else config defaults)
    eff_expected_type = expected_type
    if eff_expected_type is None and cfg:
        eff_expected_type = getattr(cfg, "EXPECTED_MARKET_TYPE", None)

    eff_expected_logic = expected_logic
    if eff_expected_logic is None and cfg:
        eff_expected_logic = getattr(cfg, "EXPECTED_MARKET_LOGIC", None)

    logger.debug(
        f"[{func_name}] Validating '{symbol}'. Checks: Type='{eff_expected_type or 'Any'}', "
        f"Logic='{eff_expected_logic or 'Any'}', Active={check_active}, "
        f"Contract={require_contract}, Reload={reload_markets}"
    )

    try:
        # Load or reload markets if necessary
        # Check both attributes as their presence might vary slightly across ccxt versions/states
        needs_load = (
            reload_markets
            or not getattr(exchange, "markets", None)
            or not getattr(exchange, "markets_by_id", None)
        )
        if needs_load:
            logger.info(f"[{func_name}] Loading/Reloading markets for validation...")
            # This call might raise NetworkError, ExchangeError etc.
            # Consider adding retries around load_markets if needed elsewhere too.
            exchange.load_markets(reload=reload_markets)
            if not getattr(exchange, "markets", None):  # Verify load was successful
                logger.error(
                    f"{Fore.RED}[{func_name}] Failed to load markets.{Style.RESET_ALL}"
                )
                return None
            logger.info(f"[{func_name}] Markets loaded successfully.")

        # Get the market details - raises ccxt.BadSymbol if not found
        market = exchange.market(symbol)

        # --- Perform Checks ---
        validation_passed = True
        error_messages = []

        # 1. Check Active Status
        is_active = market.get("active", False)
        if check_active and not is_active:
            # Log as warning, doesn't necessarily fail validation unless required by caller logic
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] Validation Warning: Market '{symbol}' is INACTIVE.{Style.RESET_ALL}"
            )
            # If inactive should cause failure, uncomment below:
            # error_messages.append(f"Market '{symbol}' is inactive.")
            # validation_passed = False

        # 2. Check Market Type
        actual_type: Optional[MarketType] = market.get("type")
        if eff_expected_type and actual_type != eff_expected_type:
            msg = f"Type mismatch for '{symbol}'. Expected '{eff_expected_type}', Got '{actual_type}'."
            error_messages.append(msg)
            validation_passed = False

        # 3. Check if Contract Market (if required)
        is_contract = market.get("contract", False)
        if require_contract and not is_contract:
            msg = f"Market '{symbol}' is not a contract market, but contract was required."
            error_messages.append(msg)
            validation_passed = False

        # 4. Check Contract Logic (Linear/Inverse) - only if it's a contract and logic is specified
        actual_logic_str: Optional[BybitV5Category] = None
        if is_contract and eff_expected_logic:
            actual_logic_str = _get_v5_category(market)  # Use internal helper
            if actual_logic_str != eff_expected_logic:
                msg = (
                    f"Contract logic mismatch for '{symbol}'. Expected '{eff_expected_logic}', "
                    f"Got '{actual_logic_str or 'Undetermined'}'."
                )
                error_messages.append(msg)
                validation_passed = False
        elif is_contract:  # Get logic even if not checking, for logging
            actual_logic_str = _get_v5_category(market)

        # --- Final Verdict ---
        if validation_passed:
            logger.info(
                f"{Fore.GREEN}[{func_name}] Market OK: '{symbol}' "
                f"(Type: {actual_type or 'N/A'}, "
                f"Logic: {actual_logic_str or 'N/A'}, "  # Display N/A if not contract or undetermined
                f"Active: {is_active}).{Style.RESET_ALL}"
            )
            return market
        else:
            for error in error_messages:
                logger.error(
                    f"{Fore.RED}[{func_name}] Validation Failed: {error}{Style.RESET_ALL}"
                )
            return None

    except ccxt.BadSymbol as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Validation Failed: Symbol '{symbol}' not found on exchange. CCXT Error: {e}{Style.RESET_ALL}"
        )
        return None
    except ccxt.NetworkError as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Network error during market validation/loading for '{symbol}': {e}{Style.RESET_ALL}"
        )
        # Depending on application logic, you might want to retry here or raise
        return None
    except ccxt.ExchangeError as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Exchange error during market validation/loading for '{symbol}': {e}{Style.RESET_ALL}"
        )
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Unexpected error validating market '{symbol}': {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- END OF FILE utils.py ---
