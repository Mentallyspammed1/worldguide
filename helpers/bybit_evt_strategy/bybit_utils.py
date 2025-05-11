# --- START OF FILE bybit_utils.py ---

import functools
import logging
import random  # For jitter
import subprocess  # For Termux API call
import time
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any, TypeVar

import ccxt

try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init

    colorama_init(autoreset=True)
except ImportError:

    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()  # type: ignore

# Assume logger is configured in the importing scope (e.g., strategy script)
# If not, create a basic placeholder
# Note: In this project structure, logger is set up in main.py, so this might run before config.
# It's better to get the logger by name if possible.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():  # Add a basic handler if none exist yet
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Default to INFO if not configured
    logger.info(
        f"Placeholder logger initialized for {__name__}. Main logger setup expected."
    )


# Placeholder TypeVar for Config object (structure defined in importing script)
# Replace with actual import if possible, or define attributes needed by this module
try:
    from config_models import AppConfig  # Try importing the main config

    ConfigPlaceholder = AppConfig  # Use the real type hint
except ImportError:

    class ConfigPlaceholder:  # type: ignore # Fallback definition
        # Define attributes expected by functions in this file
        RETRY_COUNT: int = 3
        RETRY_DELAY_SECONDS: float = 1.0
        ENABLE_SMS_ALERTS: bool = False
        SMS_RECIPIENT_NUMBER: str | None = None
        SMS_TIMEOUT_SECONDS: int = 30
        # Add others if needed by analyze_order_book, etc.

# --- Utility Functions ---


def safe_decimal_conversion(
    value: Any, default: Decimal | None = None
) -> Decimal | None:
    """Convert various inputs to Decimal, returning default or None on failure."""
    if value is None:
        return default
    try:
        # Handle potential scientific notation strings
        if isinstance(value, str) and "e" in value.lower():
            return Decimal(value)
        # Convert float to string first for precision
        if isinstance(value, float):
            value = str(value)
        d = Decimal(value)
        if d.is_nan() or d.is_infinite():
            return default  # Reject NaN/Inf
        return d
    except (ValueError, TypeError, InvalidOperation):
        # logger.warning(f"safe_decimal_conversion failed for value: {value}", exc_info=True) # Optional: log failures
        return default


def format_price(exchange: ccxt.Exchange, symbol: str, price: Any) -> str:
    """Format a price value according to the market's precision rules."""
    price_dec = safe_decimal_conversion(price)
    if price_dec is None:
        return "N/A"
    try:
        # Use CCXT's method if available
        return exchange.price_to_precision(symbol, float(price_dec))
    except (AttributeError, KeyError, TypeError, ValueError, ccxt.ExchangeError):
        # logger.warning(f"Market data/precision issue for '{symbol}' in format_price. Using fallback.", exc_info=True)
        # Fallback: format to a reasonable number of decimal places
        return f"{price_dec:.8f}"
    except Exception as e:
        logger.critical(
            f"Error formatting price '{price}' for {symbol}: {e}", exc_info=True
        )
        return "Error"


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Any) -> str:
    """Format an amount value according to the market's precision rules."""
    amount_dec = safe_decimal_conversion(amount)
    if amount_dec is None:
        return "N/A"
    try:
        # Use CCXT's method if available
        return exchange.amount_to_precision(symbol, float(amount_dec))
    except (AttributeError, KeyError, TypeError, ValueError, ccxt.ExchangeError):
        # logger.warning(f"Market data/precision issue for '{symbol}' in format_amount. Using fallback.", exc_info=True)
        # Fallback: format to a reasonable number of decimal places
        return f"{amount_dec:.8f}"
    except Exception as e:
        logger.critical(
            f"Error formatting amount '{amount}' for {symbol}: {e}", exc_info=True
        )
        return "Error"


def format_order_id(order_id: Any) -> str:
    """Format an order ID for concise logging (shows last 6 digits)."""
    try:
        id_str = str(order_id).strip() if order_id else ""
        if not id_str:
            return "UNKNOWN"
        return "..." + id_str[-6:] if len(id_str) > 6 else id_str
    except Exception as e:
        logger.error(f"Error formatting order ID {order_id}: {e}")
        return "UNKNOWN"


def send_sms_alert(message: str, config: ConfigPlaceholder | None = None) -> bool:
    """Send an SMS alert using Termux API."""
    if not config:
        logger.warning("SMS alert config missing.")
        return False

    # Use attributes directly from ConfigPlaceholder (or the real AppConfig)
    enabled = getattr(
        config,
        "ENABLE_SMS_ALERTS",
        getattr(config.sms_config, "enable_sms_alerts", False),
    )
    if not enabled:
        return True  # Return True if disabled, as no action failed

    recipient = getattr(
        config,
        "SMS_RECIPIENT_NUMBER",
        getattr(config.sms_config, "sms_recipient_number", None),
    )
    if not recipient:
        logger.warning("SMS alerts enabled but no SMS_RECIPIENT_NUMBER configured.")
        return False

    use_termux = getattr(
        config, "use_termux_api", getattr(config.sms_config, "use_termux_api", False)
    )  # Check if Termux is the method
    if not use_termux:
        logger.warning("SMS enabled but Termux method not selected/configured.")
        return False  # Only Termux is implemented here

    timeout = getattr(
        config,
        "SMS_TIMEOUT_SECONDS",
        getattr(config.sms_config, "sms_timeout_seconds", 30),
    )

    try:
        logger.info(f"Attempting to send SMS alert via Termux to {recipient}...")
        command = ["termux-sms-send", "-n", recipient, message]
        result = subprocess.run(
            command, timeout=timeout, check=True, capture_output=True, text=True
        )
        log_output = result.stdout.strip() if result.stdout else "(No output)"
        logger.info(
            f"{Fore.GREEN}SMS Alert Sent Successfully via Termux.{Style.RESET_ALL} Output: {log_output}"
        )
        return True
    except FileNotFoundError:
        logger.error(
            f"{Fore.RED}Termux API command 'termux-sms-send' not found. Is Termux:API installed and configured?{Style.RESET_ALL}"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error(
            f"{Fore.RED}Termux SMS command timed out after {timeout} seconds.{Style.RESET_ALL}"
        )
        return False
    except subprocess.CalledProcessError as e:
        stderr_log = e.stderr.strip() if e.stderr else "(No stderr)"
        logger.error(
            f"{Fore.RED}Termux SMS command failed with exit code {e.returncode}.{Style.RESET_ALL}"
        )
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Stderr: {stderr_log}")
        return False
    except Exception as e:
        logger.critical(
            f"{Fore.RED}Unexpected error sending SMS alert via Termux: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return False


# --- Retry Decorator Factory ---
T = TypeVar("T")


def retry_api_call(
    max_retries_override: int | None = None,
    initial_delay_override: float | None = None,
    handled_exceptions=(
        ccxt.RateLimitExceeded,
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.RequestTimeout,
    ),
    error_message_prefix: str = "API Call Failed",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory to retry synchronous API calls with configurable settings.
    Requires config object passed to decorated function (positional or kwarg `app_config`).
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Find config object (accept AppConfig or legacy Config)
            config_obj = kwargs.get("app_config") or kwargs.get("config")
            if not config_obj:
                config_obj = next(
                    (
                        arg
                        for arg in args
                        if isinstance(arg, (AppConfig, ConfigPlaceholder))
                    ),
                    None,
                )  # type: ignore

            if not config_obj:
                logger.error(
                    f"{Fore.RED}No config object (AppConfig or compatible) found for retry_api_call in {func.__name__}{Style.RESET_ALL}"
                )
                raise ValueError("Config object required for retry_api_call")

            # Extract retry settings from config (handle both AppConfig and legacy)
            if isinstance(config_obj, AppConfig):
                api_conf = config_obj.api_config
                sms_conf = config_obj.sms_config
                effective_max_retries = (
                    max_retries_override
                    if max_retries_override is not None
                    else api_conf.retry_count
                )
                effective_base_delay = (
                    initial_delay_override
                    if initial_delay_override is not None
                    else api_conf.retry_delay_seconds
                )
            else:  # Assume legacy Config object
                effective_max_retries = (
                    max_retries_override
                    if max_retries_override is not None
                    else getattr(config_obj, "RETRY_COUNT", 3)
                )
                effective_base_delay = (
                    initial_delay_override
                    if initial_delay_override is not None
                    else getattr(config_obj, "RETRY_DELAY_SECONDS", 1.0)
                )
                sms_conf = config_obj  # Pass the whole legacy config for SMS

            func_name = func.__name__
            last_exception = None  # Store last exception for final raise

            for attempt in range(effective_max_retries + 1):
                try:
                    if attempt > 0:
                        logger.debug(
                            f"Retrying {func_name} (Attempt {attempt + 1}/{effective_max_retries + 1})"
                        )
                    return func(
                        *args, **kwargs
                    )  # Execute the original synchronous function
                except handled_exceptions as e:
                    last_exception = e
                    if attempt >= effective_max_retries:
                        logger.error(
                            f"{Fore.RED}{error_message_prefix}: Max retries ({effective_max_retries + 1}) reached for {func_name}. Last error: {type(e).__name__} - {e}{Style.RESET_ALL}"
                        )
                        # Use the potentially legacy config directly for SMS
                        send_sms_alert(
                            f"{error_message_prefix}: Max retries for {func_name} ({type(e).__name__})",
                            sms_conf,
                        )
                        break  # Exit loop, will raise last_exception below

                    # Calculate delay with exponential backoff + jitter
                    delay = (effective_base_delay * (2**attempt)) + (
                        effective_base_delay * random.uniform(0.1, 0.5)
                    )
                    log_level = logging.WARNING
                    log_color = Fore.YELLOW

                    if isinstance(e, ccxt.RateLimitExceeded):
                        log_color = Fore.YELLOW + Style.BRIGHT
                        logger.log(
                            log_level,
                            f"{log_color}Rate limit exceeded in {func_name}. Retry {attempt + 1}/{effective_max_retries + 1} after {delay:.2f}s: {e}{Style.RESET_ALL}",
                        )
                    elif isinstance(
                        e,
                        (
                            ccxt.NetworkError,
                            ccxt.ExchangeNotAvailable,
                            ccxt.RequestTimeout,
                        ),
                    ):
                        log_level = logging.ERROR
                        log_color = Fore.RED
                        logger.log(
                            log_level,
                            f"{log_color}{type(e).__name__} in {func_name}. Retry {attempt + 1}/{effective_max_retries + 1} after {delay:.2f}s: {e}{Style.RESET_ALL}",
                        )
                    else:
                        # Generic handled exception
                        logger.log(
                            log_level,
                            f"{log_color}{type(e).__name__} in {func_name}. Retry {attempt + 1}/{effective_max_retries + 1} after {delay:.2f}s: {e}{Style.RESET_ALL}",
                        )

                    time.sleep(delay)  # Synchronous sleep

                except Exception as e:
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}Unexpected critical error in {func_name}: {type(e).__name__} - {e}{Style.RESET_ALL}",
                        exc_info=True,
                    )
                    send_sms_alert(
                        f"CRITICAL Error in {func_name}: {type(e).__name__}", sms_conf
                    )
                    raise e  # Re-raise unexpected exceptions immediately

            # If loop finished without returning, raise the last handled exception
            if last_exception:
                raise last_exception
            else:  # Should not happen unless max_retries is negative, but safeguard
                raise Exception(
                    f"Failed to execute {func_name} after {effective_max_retries + 1} retries (unknown error)"
                )

        return wrapper

    return decorator


# --- Order Book Analysis ---
# Note: This is synchronous. If used in an async context, run in executor.
@retry_api_call()  # Use default retry settings from config
def analyze_order_book(
    exchange: ccxt.Exchange,  # Expects synchronous exchange instance
    symbol: str,
    depth: int,  # Analysis depth (levels from top)
    fetch_limit: int,  # How many levels to fetch initially
    config: ConfigPlaceholder,  # Accepts legacy or AppConfig
) -> dict[str, Decimal | None]:
    """Fetches and analyzes the L2 order book (synchronous)."""
    func_name = "analyze_order_book"
    log_prefix = f"[{func_name}({symbol}, Depth:{depth}, Fetch:{fetch_limit})]"
    logger.debug(f"{log_prefix} Analyzing...")

    # Determine config type for accessing settings
    if isinstance(config, AppConfig):
        sms_conf = config.sms_config
    else:  # Assume legacy config
        sms_conf = config

    analysis_result = {
        "best_bid": None,
        "best_ask": None,
        "mid_price": None,
        "spread": None,
        "spread_pct": None,
        "bid_volume_depth": None,
        "ask_volume_depth": None,
        "bid_ask_ratio_depth": None,
        "timestamp": None,
    }

    try:
        logger.debug(f"{log_prefix} Fetching order book data...")
        # Ensure fetch_limit is at least the analysis depth
        effective_fetch_limit = max(depth, fetch_limit)
        # Bybit V5 needs category for fetch_order_book
        category = market_cache.get_category(
            symbol
        )  # Assumes market cache is populated elsewhere
        if not category:
            logger.warning(
                f"{log_prefix} Category unknown for {symbol}. Fetch may fail."
            )
            params = {}
        else:
            params = {"category": category}

        order_book = exchange.fetch_order_book(
            symbol, limit=effective_fetch_limit, params=params
        )

        if (
            not isinstance(order_book, dict)
            or "bids" not in order_book
            or "asks" not in order_book
        ):
            raise ValueError("Invalid order book structure received.")

        analysis_result["timestamp"] = order_book.get(
            "timestamp"
        )  # Store timestamp if available

        bids_raw = order_book["bids"]
        asks_raw = order_book["asks"]
        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            raise ValueError("Order book 'bids' or 'asks' data is not a list.")
        if not bids_raw or not asks_raw:
            logger.warning(
                f"{log_prefix} Order book side empty (Bids:{len(bids_raw)}, Asks:{len(asks_raw)})."
            )
            return analysis_result  # Return defaults

        # Process bids and asks, converting to Decimal
        bids: list[tuple[Decimal, Decimal]] = []
        asks: list[tuple[Decimal, Decimal]] = []
        for p, a in bids_raw:
            price = safe_decimal_conversion(p)
            amount = safe_decimal_conversion(a)
            if price and amount and price > 0 and amount >= 0:
                bids.append((price, amount))
        for p, a in asks_raw:
            price = safe_decimal_conversion(p)
            amount = safe_decimal_conversion(a)
            if price and amount and price > 0 and amount >= 0:
                asks.append((price, amount))

        # Ensure lists are sorted correctly by CCXT (bids descending, asks ascending)
        # bids.sort(key=lambda x: x[0], reverse=True) # Optional verification
        # asks.sort(key=lambda x: x[0]) # Optional verification

        if not bids or not asks:
            logger.warning(
                f"{log_prefix} Order book validated bids/asks empty after conversion."
            )
            return analysis_result

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        analysis_result["best_bid"] = best_bid
        analysis_result["best_ask"] = best_ask

        if best_bid >= best_ask:
            logger.error(
                f"{Fore.RED}{log_prefix} Order book crossed! Bid ({best_bid}) >= Ask ({best_ask}).{Style.RESET_ALL}"
            )
            # Return crossed data for potential handling upstream
            return analysis_result

        analysis_result["mid_price"] = (best_bid + best_ask) / Decimal("2")
        analysis_result["spread"] = best_ask - best_bid
        analysis_result["spread_pct"] = (
            (analysis_result["spread"] / analysis_result["mid_price"]) * Decimal("100")
            if analysis_result["mid_price"] > 0
            else Decimal("inf")
        )  # Use mid price for stability

        # Calculate volume within the specified depth
        bid_vol_depth = sum(b[1] for b in bids[:depth] if b[1] is not None)
        ask_vol_depth = sum(a[1] for a in asks[:depth] if a[1] is not None)
        analysis_result["bid_volume_depth"] = bid_vol_depth
        analysis_result["ask_volume_depth"] = ask_vol_depth
        analysis_result["bid_ask_ratio_depth"] = (
            (bid_vol_depth / ask_vol_depth)
            if ask_vol_depth and ask_vol_depth > 0
            else None
        )

        ratio_str = (
            f"{analysis_result['bid_ask_ratio_depth']:.2f}"
            if analysis_result["bid_ask_ratio_depth"] is not None
            else "N/A"
        )
        logger.debug(
            f"{log_prefix} Analysis OK: Spread={analysis_result['spread_pct']:.4f}%, Depth Ratio(d{depth})={ratio_str}"
        )

        return analysis_result

    except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, TypeError) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Error fetching/analyzing: {e}{Style.RESET_ALL}"
        )
        raise  # Re-raise handled exceptions for the decorator
    except Exception as e:
        logger.critical(
            f"{Back.RED}{Fore.WHITE}{log_prefix} Unexpected error analyzing order book: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(f"CRITICAL: OB Analysis failed for {symbol}", sms_conf)
        return analysis_result  # Return default on critical failure


# --- END OF FILE bybit_utils.py ---
