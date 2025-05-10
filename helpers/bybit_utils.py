import functools
import logging
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

    Fore = Style = Back = DummyColor()

# Assume logger is configured in the importing scope (e.g., strategy script)
# If not, create a basic placeholder
if "logger" not in globals():
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.info("Placeholder logger initialized for bybit_utils.py.")
else:
    # If logger exists, ensure it's the correct one for this module context
    logger = logging.getLogger(__name__)


# Placeholder TypeVar for Config object (structure defined in importing script)
ConfigPlaceholder = TypeVar("ConfigPlaceholder")

# --- Utility Functions ---


def safe_decimal_conversion(
    value: Any, default: Decimal | None = None
) -> Decimal | None:
    """Convert various inputs to Decimal, returning default or None on failure.

    Args:
        value: Input value (e.g., str, float, int, Decimal).
        default: Value to return if conversion fails (default: None).

    Returns:
        Decimal value, or default/None on failure.
    """
    if value is None:
        return default
    try:
        # Handle potential scientific notation in strings
        if isinstance(value, str) and "e" in value.lower():
            return Decimal(value)
        # Convert other types to string first for robust Decimal conversion
        return Decimal(str(value))
    except (ValueError, TypeError, InvalidOperation):
        # logger.debug(f"Failed to convert {type(value)} '{value}' to Decimal, returning {default}")
        return default


def format_price(exchange: ccxt.Exchange, symbol: str, price: Any) -> str:
    """Format a price value according to the market's precision rules.

    Args:
        exchange: CCXT exchange object (e.g., ccxt.bybit).
        symbol: Trading pair (e.g., 'BTC/USDT:USDT').
        price: Price value (e.g., Decimal, float, str).

    Returns:
        Formatted price string adhering to market precision, or "N/A".
    """
    if price is None:
        return "N/A"
    try:
        # price_to_precision is the preferred method in ccxt
        return exchange.price_to_precision(symbol, price)
    except (KeyError, AttributeError):
        logger.warning(
            f"{Fore.YELLOW}Market data or precision method not found for symbol '{symbol}' in format_price. Using fallback.{Style.RESET_ALL}"
        )
        # Fallback formatting
        price_dec = safe_decimal_conversion(price)
        return (
            f"{price_dec:.8f}" if price_dec is not None else "Invalid"
        )  # Use more decimals as fallback
    except Exception as e:
        logger.critical(
            f"{Fore.RED}Error formatting price '{price}' for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return "Error"  # Return error string


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Any) -> str:
    """Format an amount value according to the market's precision rules.

    Args:
        exchange: CCXT exchange object (e.g., ccxt.bybit).
        symbol: Trading pair (e.g., 'BTC/USDT:USDT').
        amount: Amount value (e.g., Decimal, float, str).

    Returns:
        Formatted amount string adhering to market precision, or "N/A".
    """
    if amount is None:
        return "N/A"
    try:
        # amount_to_precision is the preferred method in ccxt
        return exchange.amount_to_precision(symbol, amount)
    except (KeyError, AttributeError):
        logger.warning(
            f"{Fore.YELLOW}Market data or precision method not found for symbol '{symbol}' in format_amount. Using fallback.{Style.RESET_ALL}"
        )
        amount_dec = safe_decimal_conversion(amount)
        return (
            f"{amount_dec:.8f}" if amount_dec is not None else "Invalid"
        )  # Use more decimals as fallback
    except Exception as e:
        logger.critical(
            f"{Fore.RED}Error formatting amount '{amount}' for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return "Error"


def format_order_id(order_id: Any) -> str:
    """Format an order ID for concise logging (shows last 6 digits).

    Args:
        order_id: Order ID (e.g., str, int).

    Returns:
        Formatted order ID string, or 'UNKNOWN' if invalid.
    """
    try:
        id_str = str(order_id).strip() if order_id else ""
        if not id_str:
            return "UNKNOWN"
        return "..." + id_str[-6:] if len(id_str) > 6 else id_str  # Add ellipsis
    except Exception as e:
        logger.error(
            f"{Fore.RED}Error formatting order ID {order_id}: {e}{Style.RESET_ALL}"
        )
        return "UNKNOWN"


def send_sms_alert(message: str, config: ConfigPlaceholder | None = None) -> bool:
    """Send an SMS alert using Termux API.

    Args:
        message: Message to send.
        config: Config object with SMS settings:
            - ENABLE_SMS_ALERTS (bool)
            - SMS_RECIPIENT_NUMBER (str)
            - SMS_TIMEOUT_SECONDS (int, optional, default 30)

    Returns:
        True if alert sent successfully, False otherwise or if disabled.
    """
    enabled = getattr(config, "ENABLE_SMS_ALERTS", False) if config else False
    if not enabled:
        # logger.debug(f"SMS alert suppressed (disabled): {message}")
        return False

    recipient = getattr(config, "SMS_RECIPIENT_NUMBER", None)
    if not recipient:
        logger.warning("SMS alerts enabled but no SMS_RECIPIENT_NUMBER configured.")
        return False

    timeout = getattr(config, "SMS_TIMEOUT_SECONDS", 30)  # Default 30s timeout

    try:
        logger.info(f"Attempting to send SMS alert via Termux to {recipient}...")
        command = ["termux-sms-send", "-n", recipient, message]
        # Use subprocess.run to execute the command
        result = subprocess.run(
            command,
            timeout=timeout,
            check=True,  # Raise CalledProcessError on non-zero exit code
            capture_output=True,  # Capture stdout/stderr
            text=True,  # Decode stdout/stderr as text
        )
        logger.info(
            f"{Fore.GREEN}SMS Alert Sent Successfully via Termux.{Style.RESET_ALL} Output: {result.stdout.strip() if result.stdout else '(No output)'}"
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
        logger.error(
            f"{Fore.RED}Termux SMS command failed with exit code {e.returncode}.{Style.RESET_ALL}"
        )
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Stderr: {e.stderr.strip() if e.stderr else '(No stderr)'}")
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
    max_retries: int | None = None,
    initial_delay: float | None = None,
    handled_exceptions=(
        ccxt.RateLimitExceeded,
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.RequestTimeout,
    ),
    error_message_prefix: str = "API Call Failed",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory to retry API calls with configurable settings.

    Handles specified CCXT exceptions with exponential backoff for RateLimitExceeded.
    Uses settings from the passed 'config' object as defaults if not provided directly
    to the decorator.

    Usage:
        @retry_api_call() # Uses defaults from config
        def my_api_func(exchange, symbol, config): ...

        @retry_api_call(max_retries=5, initial_delay=1.5) # Overrides defaults
        def another_api_func(exchange, params, config): ...

    Args:
        max_retries (Optional[int]): Max number of retries. Uses config.RETRY_COUNT if None.
        initial_delay (Optional[float]): Base delay in seconds. Uses config.RETRY_DELAY_SECONDS if None.
        handled_exceptions (tuple): Tuple of exception types to handle and retry.
        error_message_prefix (str): Prefix for log messages on failure.

    Returns:
        Callable: The actual decorator.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Find the config object passed to the decorated function
            config = kwargs.get("config") or next(
                (
                    arg
                    for arg in args
                    if hasattr(arg, "RETRY_COUNT")
                    and hasattr(arg, "RETRY_DELAY_SECONDS")
                ),
                None,
            )
            if not config:
                logger.error(
                    f"{Fore.RED}No config object with RETRY_COUNT/RETRY_DELAY_SECONDS found for retry_api_call in {func.__name__}{Style.RESET_ALL}"
                )
                # Optionally raise or return a default value depending on function needs
                raise ValueError("Config object required for retry_api_call")

            # Determine effective retry parameters
            effective_max_retries = (
                max_retries
                if max_retries is not None
                else getattr(config, "RETRY_COUNT", 3)
            )
            effective_base_delay = (
                initial_delay
                if initial_delay is not None
                else getattr(config, "RETRY_DELAY_SECONDS", 1.0)
            )

            func_name = func.__name__  # For logging
            attempt = 0
            while (
                attempt <= effective_max_retries
            ):  # Allow up to max_retries total retries (max_retries + 1 attempts)
                try:
                    if attempt > 0:
                        logger.debug(
                            f"Retrying {func_name} (Attempt {attempt}/{effective_max_retries})"
                        )
                    return func(*args, **kwargs)
                except handled_exceptions as e:
                    attempt += 1
                    if attempt > effective_max_retries:
                        logger.error(
                            f"{Fore.RED}{error_message_prefix}: Max retries ({effective_max_retries}) reached for {func_name}. Last error: {type(e).__name__} - {e}{Style.RESET_ALL}"
                        )
                        send_sms_alert(
                            f"{error_message_prefix}: Max retries for {func_name} ({type(e).__name__})",
                            config,
                        )
                        raise e  # Re-raise the last caught exception

                    delay = effective_base_delay
                    if isinstance(e, ccxt.RateLimitExceeded):
                        # Exponential backoff for rate limits
                        delay *= 2 ** (
                            attempt - 1
                        )  # Start with base delay on first retry
                        logger.warning(
                            f"{Fore.YELLOW}Rate limit exceeded in {func_name}. Retry {attempt}/{effective_max_retries} after {delay:.2f}s: {e}{Style.RESET_ALL}"
                        )
                    elif isinstance(e, (ccxt.NetworkError, ccxt.RequestTimeout)):
                        logger.error(
                            f"{Fore.RED}Network/Timeout error in {func_name}. Retry {attempt}/{effective_max_retries} after {delay:.2f}s: {e}{Style.RESET_ALL}"
                        )
                    elif isinstance(e, ccxt.ExchangeNotAvailable):
                        logger.error(
                            f"{Fore.RED}Exchange not available in {func_name}. Retry {attempt}/{effective_max_retries} after {delay:.2f}s: {e}{Style.RESET_ALL}"
                        )
                    else:  # Should not happen if handled_exceptions is correct, but defensive
                        logger.warning(
                            f"{Fore.YELLOW}Handled exception {type(e).__name__} in {func_name}. Retry {attempt}/{effective_max_retries} after {delay:.2f}s: {e}{Style.RESET_ALL}"
                        )

                    time.sleep(delay)

                except Exception as e:
                    # Catch unexpected errors immediately, log, and re-raise (don't retry)
                    logger.critical(
                        f"{Back.RED}{Fore.WHITE}Unexpected critical error in {func_name}: {type(e).__name__} - {e}{Style.RESET_ALL}",
                        exc_info=True,
                    )
                    send_sms_alert(
                        f"CRITICAL Error in {func_name}: {type(e).__name__}", config
                    )
                    raise e  # Re-raise unexpected exceptions

            # Should only be reached if loop completes without success or raising exception
            # (e.g., if handled_exceptions occurs on the last attempt and isn't re-raised above)
            # Re-raise the last error encountered if loop finishes due to retries exhausted
            raise Exception(
                f"Failed to execute {func_name} after {effective_max_retries} retries. Last known error: {e}"
            )

        return wrapper

    return decorator


# --- Order Book Analysis ---


@retry_api_call()  # Use default retry settings from config
def analyze_order_book(
    exchange: ccxt.bybit,
    symbol: str,
    depth: int,
    fetch_limit: int,
    config: ConfigPlaceholder,
) -> dict[str, Decimal | None]:
    """Fetches and analyzes the L2 order book.

    Args:
        exchange: Initialized ccxt.bybit exchange instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        depth: Analysis depth (levels from best bid/ask).
        fetch_limit: How many levels to fetch from API (>= depth).
        config: Configuration object.

    Returns:
        A dictionary with analysis results or defaults to None values on failure.
        Keys: 'best_bid', 'best_ask', 'mid_price', 'spread', 'spread_pct',
              'bid_volume_depth', 'ask_volume_depth', 'bid_ask_ratio_depth'.
    """
    func_name = "analyze_order_book"
    logger.debug(
        f"[{func_name}] Analyzing OB for {symbol} (Fetch Limit: {fetch_limit}, Analysis Depth: {depth})"
    )

    # Default return structure
    analysis_result = {
        "best_bid": None,
        "best_ask": None,
        "mid_price": None,
        "spread": None,
        "spread_pct": None,
        "bid_volume_depth": None,
        "ask_volume_depth": None,
        "bid_ask_ratio_depth": None,
    }

    try:
        logger.debug(f"[{func_name}] Fetching order book data (limit={fetch_limit})...")
        # Ensure fetch_limit is at least depth
        effective_fetch_limit = max(depth, fetch_limit)
        order_book = exchange.fetch_order_book(symbol, limit=effective_fetch_limit)

        if (
            not isinstance(order_book, dict)
            or "bids" not in order_book
            or "asks" not in order_book
        ):
            raise ValueError("Invalid order book structure received.")

        bids_raw = order_book["bids"]
        asks_raw = order_book["asks"]

        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            raise ValueError("Order book 'bids' or 'asks' data is not a list.")

        if not bids_raw or not asks_raw:
            logger.warning(
                f"[{func_name}] Order book for {symbol} is empty (no bids or asks)."
            )
            return analysis_result  # Return default empty result

        # Convert and validate fetched levels (up to fetch_limit)
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

        if not bids or not asks:
            logger.warning(
                f"[{func_name}] Order book for {symbol} has empty validated bids or asks."
            )
            return analysis_result

        # Perform analysis using the validated bids/asks
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        analysis_result["best_bid"] = best_bid
        analysis_result["best_ask"] = best_ask

        if best_bid >= best_ask:
            logger.error(
                f"{Fore.RED}[{func_name}] Order book crossed: Bid ({best_bid}) >= Ask ({best_ask}) for {symbol}.{Style.RESET_ALL}"
            )
            # Return partial results or empty? Returning partial for now.
            return analysis_result

        analysis_result["mid_price"] = (best_bid + best_ask) / Decimal("2")
        analysis_result["spread"] = best_ask - best_bid
        analysis_result["spread_pct"] = (
            (analysis_result["spread"] / best_bid) * Decimal("100")
            if best_bid > 0
            else Decimal("0")
        )

        # Calculate cumulative volume within the specified *analysis* depth
        bid_vol_depth = sum(
            b[1] for b in bids[:depth] if b[1] is not None
        )  # Ensure amount is not None
        ask_vol_depth = sum(
            a[1] for a in asks[:depth] if a[1] is not None
        )  # Ensure amount is not None
        analysis_result["bid_volume_depth"] = bid_vol_depth
        analysis_result["ask_volume_depth"] = ask_vol_depth
        analysis_result["bid_ask_ratio_depth"] = (
            (bid_vol_depth / ask_vol_depth)
            if ask_vol_depth and ask_vol_depth > 0
            else None
        )

        logger.debug(
            f"[{func_name}] OB Analysis OK: Spread={analysis_result['spread_pct']:.4f}%, "
            f"Ratio(d{depth})={analysis_result['bid_ask_ratio_depth']:.2f if analysis_result['bid_ask_ratio_depth'] is not None else 'N/A'}"
        )

        return analysis_result

    except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, TypeError) as e:
        # Handled exceptions will be retried by the decorator
        logger.warning(
            f"{Fore.YELLOW}[{func_name}] Error fetching/analyzing order book for {symbol}: {e}{Style.RESET_ALL}"
        )
        raise  # Re-raise for decorator
    except Exception as e:
        # Unexpected errors are logged critically and not retried
        logger.critical(
            f"{Back.RED}{Fore.WHITE}[{func_name}] Unexpected error analyzing order book for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            f"CRITICAL: OB Analysis failed for {symbol}", config
        )  # Use send_sms_alert here
        return analysis_result  # Return default structure on critical failure
