#!/usr/bin/env python

"""Bybit V5 CCXT Helper Functions (v3.5 - Enhanced & Fixed)

A robust, modular, and enhanced collection of helper functions for interacting with
the Bybit V5 API (Unified Trading Account) using the CCXT library. This version
fixes critical initialization errors (AttributeError: 'closed'), improves batch order
handling and error reporting, enhances market loading robustness, adds dependency checks,
and improves overall logging and error handling.

Key Features:
- Fully asynchronous operations (`async`/`await`).
- Logically grouped functions (can be split into submodules).
- Enhanced type safety with TypedDict, Enums, and precise hints.
- Performance optimizations via integrated MarketCache.
- Centralized asynchronous error handling and retry logic via decorator.
- Structured logging with conditional color support.
- Implemented features: Batch orders, comprehensive order types,
  conditional orders (basic Stop), common fetch/cancel operations.
- Increased robustness and handling of Bybit V5 specifics (category, filters, UTA).
- Self-contained: Includes necessary utility functions and decorators.
- Improved WebSocket handling stubs (full implementation not shown here).

Version: 3.5
"""

# Standard Library Imports
import asyncio
import json
import logging
import os
import random
import sys
import time
from collections.abc import Callable, Coroutine, Sequence
from decimal import Decimal, InvalidOperation, getcontext
from enum import Enum
from typing import (
    Any,
    Literal,
    TypedDict,
    TypeVar,
)

# Third-party Libraries
# Attempt to import CCXT and handle potential ImportError
try:
    import ccxt.async_support as ccxt
    from ccxt.base.errors import (
        ArgumentsRequired,
        AuthenticationError,
        BadSymbol,
        ExchangeError,
        ExchangeNotAvailable,
        InsufficientFunds,
        InvalidOrder,
        NetworkError,
        NotSupported,
        OrderImmediatelyFillable,
        OrderNotFound,
        RateLimitExceeded,
        RequestTimeout,
    )

    # Check CCXT version
    if hasattr(ccxt, "__version__"):
        try:
            ccxt_version = tuple(map(int, ccxt.__version__.split(".")))
            if ccxt_version < (4, 1, 0):
                print(f"Warning: CCXT version {ccxt.__version__} is outdated. Recommend version 4.1.0 or higher.")
        except Exception:
            print(f"Warning: Could not parse CCXT version: {ccxt.__version__}")
    else:
        print("Warning: Could not determine CCXT version.")

except ImportError:
    print("FATAL ERROR: CCXT library not found.")
    print("Please install it: pip install ccxt>=4.1.0")
    sys.exit(1)

# Attempt to import pandas and handle potential ImportError (Optional)
try:
    import pandas as pd
except ImportError:
    print("Info: pandas library not found. OHLCV functions will return lists, not DataFrames.")
    print("Install for DataFrame support: pip install pandas>=2.0.0")
    pd = None  # Set pandas to None if not installed

# Attempt to import colorama and handle potential ImportError (Optional)
try:
    from colorama import Back, Fore, Style, init

    # Initialize colorama
    if os.name == "nt":
        init(autoreset=True)  # Autoreset is convenient on Windows
    else:
        init()  # Standard init for other platforms
except ImportError:
    print("Info: colorama not found. Logs will be uncolored.")

    # Define dummy color objects if colorama is not available
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()

# Attempt to import websockets and handle potential ImportError (Optional)
# Note: Full WebSocket implementation is not included in this enhanced script,
# but the imports are kept for compatibility if added later.
try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed,
        ConnectionClosedError,
        ConnectionClosedOK,
        InvalidURI,
        ProtocolError,
        WebSocketException,
    )
    from websockets.legacy.client import (
        WebSocketClientProtocol,  # May change path in future websockets versions
    )
except ImportError:
    print("Info: websockets library not found. WebSocket features disabled.")
    websockets = None

    class DummyWebSocketException(Exception):
        pass

    WebSocketClientProtocol = Any  # type: ignore
    # Define dummy exceptions if websockets is not available
    WebSocketException = ConnectionClosed = ConnectionClosedOK = ConnectionClosedError = InvalidURI = ProtocolError = (
        DummyWebSocketException
    )


# --- Configuration & Constants ---

getcontext().prec = 28  # Set precision for Decimal operations


class Config(TypedDict, total=False):
    """Strongly typed configuration dictionary."""

    EXCHANGE_ID: Literal["bybit"]
    API_KEY: str | None  # Made optional to allow public-only mode
    API_SECRET: str | None  # Made optional
    TESTNET_MODE: bool
    SYMBOL: str  # Default symbol for context
    USDT_SYMBOL: str  # Typically 'USDT'
    DEFAULT_MARGIN_MODE: Literal["isolated", "cross"]
    DEFAULT_RECV_WINDOW: int
    DEFAULT_SLIPPAGE_PCT: Decimal
    POSITION_QTY_EPSILON: Decimal  # Small value for float comparisons
    SHALLOW_OB_FETCH_DEPTH: int  # For quick spread checks
    ORDER_BOOK_FETCH_LIMIT: int  # For full order book fetch
    EXPECTED_MARKET_TYPE: Literal["swap", "spot", "option", "future"]  # Default context
    EXPECTED_MARKET_LOGIC: Literal["linear", "inverse"]  # Default context
    RETRY_COUNT: int  # For retry decorator
    RETRY_DELAY_SECONDS: float  # Initial delay for retry
    WS_CONNECT_TIMEOUT: float  # WebSocket connection timeout
    WS_PING_INTERVAL: float | None  # WebSocket ping interval
    ENABLE_SMS_ALERTS: bool
    # Placeholder SMS config (ensure these are handled if ENABLE_SMS_ALERTS is True)
    # TWILIO_ACCOUNT_SID: Optional[str]
    # TWILIO_AUTH_TOKEN: Optional[str]
    # TWILIO_FROM_NUMBER: Optional[str]
    # ALERT_TO_NUMBER: Optional[str]
    BROKER_ID: str | None  # Optional Broker/Referral ID
    VERSION: str  # Helper script version


# --- Enums ---
# Using standard Enums for better code clarity and safety
class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Category(str, Enum):
    LINEAR = "linear"
    INVERSE = "inverse"
    SPOT = "spot"
    OPTION = "option"


class OrderFilter(str, Enum):
    """V5 Order Filter Types."""

    ORDER = "Order"
    STOP_ORDER = "StopOrder"
    TPSL_ORDER = "tpslOrder"
    # Add others if needed, e.g., "OcoOrder", "OtoOrder"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Til Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    POST_ONLY = "PostOnly"


class TriggerDirection(int, Enum):
    """Direction for conditional order triggers."""

    RISE = 1  # Trigger when price rises to triggerPrice
    FALL = 2  # Trigger when price falls to triggerPrice


class PositionIdx(int, Enum):
    """Position index for Hedge Mode."""

    ONE_WAY = 0
    BUY_SIDE = 1  # Hedge Mode Buy
    SELL_SIDE = 2  # Hedge Mode Sell


class TriggerBy(str, Enum):
    """Price type for conditional order triggers."""

    LAST = "LastPrice"
    MARK = "MarkPrice"
    INDEX = "IndexPrice"


class OrderType(str, Enum):
    MARKET = "Market"
    LIMIT = "Limit"


# --- Logger Setup ---
# Basic logger setup, can be replaced by a more sophisticated setup (like the Neon Logger mentioned in the logs)
logger = logging.getLogger(__name__)
if not logger.hasHandlers():  # Avoid adding handlers multiple times
    logger.setLevel(logging.INFO)  # Default level
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] {%(name)s:%(lineno)d} - %(message)s",  # Adjusted format slightly
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # To prevent duplicate logging if root logger is also configured:
    # logger.propagate = False


# --- Market Cache ---
class MarketCache:
    """Caches market data fetched from the exchange to reduce API calls."""

    def __init__(self):
        self._markets: dict[str, dict[str, Any]] = {}
        self._categories: dict[str, Category | None] = {}  # Cache derived category
        self._lock = asyncio.Lock()  # Protect concurrent access

    async def load_markets(self, exchange: ccxt.bybit, reload: bool = False) -> bool:
        """Loads or reloads all markets into the cache asynchronously and safely."""
        async with self._lock:
            if not self._markets or reload:
                action = "Reloading" if self._markets else "Loading"
                logger.info(f"{Fore.BLUE}[MarketCache] {action} markets from {exchange.id}...{Style.RESET_ALL}")
                try:
                    # Explicitly reload market data from the exchange
                    all_markets = await exchange.load_markets(reload=True)  # Force reload from API

                    if not all_markets:
                        # This case indicates a problem, maybe empty response or parsing issue in CCXT
                        logger.critical(
                            f"{Back.RED}[MarketCache] FATAL: Failed to load markets - received empty or invalid data from CCXT.{Style.RESET_ALL}"
                        )
                        self._markets = {}
                        self._categories = {}
                        return False

                    self._markets = all_markets
                    self._categories.clear()  # Clear derived category cache
                    logger.info(
                        f"{Fore.GREEN}[MarketCache] Successfully loaded {len(self._markets)} markets.{Style.RESET_ALL}"
                    )
                    return True

                # Handle specific CCXT exceptions for network/availability issues
                except (NetworkError, ExchangeNotAvailable, RequestTimeout) as e:
                    logger.error(
                        f"{Fore.RED}[MarketCache] Network/Availability error loading markets: {type(e).__name__}: {e}{Style.RESET_ALL}"
                    )
                    # Don't clear markets here, might be temporary issue, allow retries at higher level
                    return False
                # Handle general exchange errors during market loading
                except ExchangeError as e:
                    logger.error(
                        f"{Fore.RED}[MarketCache] Exchange error loading markets: {type(e).__name__}: {e}{Style.RESET_ALL}",
                        exc_info=False,
                    )
                    # Don't clear markets, could be permissions or temporary API issue
                    return False
                # Catch any other unexpected exceptions during market loading
                except Exception as e:
                    logger.critical(
                        f"{Back.RED}[MarketCache] CRITICAL UNEXPECTED error loading markets: {type(e).__name__}: {e}{Style.RESET_ALL}",
                        exc_info=True,
                    )
                    # Clear markets as state is unknown
                    self._markets = {}
                    self._categories = {}
                    return False
            else:
                # Markets already loaded and reload not requested
                logger.debug("[MarketCache] Markets already loaded, skipping reload.")
                return True

    def get_market(self, symbol: str) -> dict[str, Any] | None:
        """Retrieves market data for a symbol from the cache."""
        if not self._markets:
            logger.warning("[MarketCache] Market cache is empty. Call load_markets first.")
            return None
        market_data = self._markets.get(symbol)
        if not market_data:
            logger.warning(f"[MarketCache] Market data for '{symbol}' not found in cache.")  # Changed to warning
        return market_data

    def get_category(self, symbol: str) -> Category | None:
        """Retrieves the V5 category for a symbol, using cached result if available."""
        if not self._markets:
            logger.warning("[MarketCache] Market cache is empty. Cannot determine category.")
            return None

        # Check cache first
        if symbol in self._categories:
            return self._categories[symbol]

        # If not cached, determine from market data
        market = self.get_market(symbol)
        category: Category | None = None
        if market:
            category_str = _get_v5_category(market)  # Use internal helper
            if category_str:
                try:
                    category = Category(category_str)
                except ValueError:
                    logger.error(f"[MarketCache] Invalid category value '{category_str}' derived for '{symbol}'.")
                    category = None  # Mark as invalid/undetermined
            else:
                logger.warning(f"[MarketCache] Could not derive category for symbol '{symbol}'.")

        # Cache the result (even if None) to avoid re-computation
        self._categories[symbol] = category
        return category

    def get_all_symbols(self) -> list[str]:
        """Returns a list of all symbols currently loaded in the cache."""
        return list(self._markets.keys())


# Instantiate the cache globally or pass it around as needed
market_cache = MarketCache()

# --- Utility Functions ---


def safe_decimal_conversion(value: Any, default: Decimal | None = None) -> Decimal | None:
    """Safely converts a value to a Decimal, handling None, empty strings, NaN, Infinity."""
    if value is None or value == "":
        return default
    try:
        # Convert to string first to handle floats more reliably
        d = Decimal(str(value))
        # Check for NaN (Not a Number) and Infinity
        if d.is_nan() or d.is_infinite():
            logger.warning(f"[safe_decimal] Input '{value}' resulted in NaN or Infinity, returning default.")
            return default
        return d
    except (ValueError, TypeError, InvalidOperation):
        # Catch potential errors during conversion
        logger.warning(
            f"[safe_decimal] Could not convert '{value}' (type: {type(value).__name__}) to Decimal, returning default."
        )
        return default


def format_price(exchange: ccxt.bybit, symbol: str, price: Decimal | float | str | None) -> str | None:
    """Formats a price according to market precision using CCXT, with improved fallback."""
    if price is None:
        return None
    market = market_cache.get_market(symbol)  # Use cached market data
    price_decimal = safe_decimal_conversion(price)  # Use safe conversion

    if price_decimal is None:
        logger.error(f"[format_price] Invalid price value '{price}' for {symbol} after conversion.")
        return None  # Cannot format invalid decimal

    if not market:
        logger.warning(
            f"[format_price] Market data for {symbol} unavailable. Returning raw Decimal string: {price_decimal}"
        )
        return str(price_decimal)  # Fallback to raw string if no market data

    try:
        # Use CCXT's built-in precision formatting
        return exchange.price_to_precision(symbol, float(price_decimal))
    except BadSymbol:
        # This might happen if markets become stale after initial load
        logger.error(f"[format_price] BadSymbol error for {symbol}. Markets might be stale. Using fallback formatting.")
    except NotSupported:
        logger.warning(f"[format_price] CCXT price_to_precision not supported for {symbol}. Using fallback.")
    except Exception as e:
        # Catch other potential CCXT errors
        logger.error(f"[format_price] CCXT error formatting price {price_decimal} for {symbol}: {e}", exc_info=False)

    # --- Fallback Formatting (if CCXT method fails or market is stale) ---
    try:
        # Attempt to get precision from the cached market data
        price_precision_digits = market.get("precision", {}).get("price")
        if price_precision_digits is not None:
            # Use Decimal quantization for accurate rounding based on digits after decimal point
            # Assumes precision is number of decimal places
            precision_decimal = Decimal("1e-" + str(int(price_precision_digits)))
            formatted_price = price_decimal.quantize(precision_decimal)
            return str(formatted_price)
        else:
            # If precision info is missing, return the unconverted decimal as string
            logger.warning(
                f"[format_price] Price precision ('precision.price') not found for {symbol} in market data. Returning raw Decimal string."
            )
            return str(price_decimal)
    except (ValueError, TypeError, KeyError, InvalidOperation) as format_err:
        # Catch errors during fallback formatting itself
        logger.error(
            f"[format_price] Fallback formatting failed for {price_decimal}: {format_err}. Returning raw Decimal string."
        )
        return str(price_decimal)  # Last resort: return raw decimal string


def format_amount(exchange: ccxt.bybit, symbol: str, amount: Decimal | float | str | None) -> str | None:
    """Formats an amount according to market precision using CCXT, with improved fallback."""
    if amount is None:
        return None
    market = market_cache.get_market(symbol)
    amount_decimal = safe_decimal_conversion(amount)

    if amount_decimal is None:
        logger.error(f"[format_amount] Invalid amount value '{amount}' for {symbol} after conversion.")
        return None

    if not market:
        logger.warning(
            f"[format_amount] Market data for {symbol} unavailable. Returning raw Decimal string: {amount_decimal}"
        )
        return str(amount_decimal)

    try:
        # Use CCXT's built-in precision formatting
        return exchange.amount_to_precision(symbol, float(amount_decimal))
    except BadSymbol:
        logger.error(
            f"[format_amount] BadSymbol error for {symbol}. Markets might be stale. Using fallback formatting."
        )
    except NotSupported:
        logger.warning(f"[format_amount] CCXT amount_to_precision not supported for {symbol}. Using fallback.")
    except Exception as e:
        logger.error(f"[format_amount] CCXT error formatting amount {amount_decimal} for {symbol}: {e}", exc_info=False)

    # --- Fallback Formatting ---
    try:
        amount_precision_digits = market.get("precision", {}).get("amount")
        if amount_precision_digits is not None:
            precision_decimal = Decimal("1e-" + str(int(amount_precision_digits)))
            formatted_amount = amount_decimal.quantize(precision_decimal)
            return str(formatted_amount)
        else:
            logger.warning(
                f"[format_amount] Amount precision ('precision.amount') not found for {symbol} in market data. Returning raw Decimal string."
            )
            return str(amount_decimal)
    except (ValueError, TypeError, KeyError, InvalidOperation) as format_err:
        logger.error(
            f"[format_amount] Fallback formatting failed for {amount_decimal}: {format_err}. Returning raw Decimal string."
        )
        return str(amount_decimal)


def format_order_id(order_id: str | None) -> str:
    """Returns a truncated version of the order ID for cleaner logging."""
    if not order_id:
        return "N/A"
    # Show first few and last few characters for uniqueness
    if len(order_id) > 12:
        return f"{order_id[:4]}...{order_id[-4:]}"
    return order_id


def send_sms_alert(message: str, config: Config | None = None) -> None:
    """Placeholder for sending SMS alerts. Logs warning if triggered."""
    # Check if config exists and SMS is enabled
    if not config or not config.get("ENABLE_SMS_ALERTS", False):
        # Log info message if SMS is disabled but alert was called
        logger.info(f"[SMS Alert Disabled] >> {message}")
        return

    # Log a warning indicating an alert would be sent
    logger.warning(f"{Back.YELLOW}{Fore.BLACK}[SMS Alert Triggered]{Style.RESET_ALL} >> {message}")

    # ---=== Placeholder for Actual SMS Integration ===---
    # Example using Twilio (requires 'twilio' library: pip install twilio)
    # account_sid = config.get("TWILIO_ACCOUNT_SID")
    # auth_token = config.get("TWILIO_AUTH_TOKEN")
    # from_num = config.get("TWILIO_FROM_NUMBER")
    # to_num = config.get("ALERT_TO_NUMBER")
    #
    # if not all([account_sid, auth_token, from_num, to_num]):
    #     logger.error("[SMS Alert] Twilio configuration keys missing. Cannot send SMS.")
    #     return
    #
    # try:
    #     from twilio.rest import Client
    #     client = Client(account_sid, auth_token)
    #     sms = client.messages.create(
    #         body=f"[TradingBot] {message}",
    #         from_=from_num,
    #         to=to_num
    #     )
    #     logger.info(f"[SMS Alert] Successfully sent. SID: {sms.sid}")
    # except ImportError:
    #     logger.error("[SMS Alert] Twilio library not found. Cannot send SMS. (pip install twilio)")
    # except Exception as e:
    #     logger.error(f"[SMS Alert] Failed to send SMS: {e}", exc_info=False)
    # ---=== End Placeholder ===---


def _get_v5_category(market: dict[str, Any]) -> str | None:
    """Internal helper to determine Bybit V5 category from CCXT market info. Prioritizes V5 fields."""
    if not market:
        return None
    symbol = market.get("symbol", "N/A")  # For logging context

    # 1. Check V5 specific 'info' fields first (most reliable)
    info = market.get("info", {})
    v5_category = info.get("category")  # Bybit V5 API response field
    if v5_category and isinstance(v5_category, str):
        try:
            # Validate against our Enum
            cat_enum = Category(v5_category.lower())
            logger.debug(
                f"[_get_v5_category] Using category '{cat_enum.value}' from market['info']['category'] for {symbol}"
            )
            return cat_enum.value
        except ValueError:
            # Log if the value from API is not in our Enum
            logger.warning(
                f"[_get_v5_category] Unknown category '{v5_category}' found in market['info'] for {symbol}. Ignoring."
            )

    # 2. Check explicit CCXT flags (derived by CCXT)
    if market.get("spot", False):
        return Category.SPOT.value
    if market.get("option", False):
        return Category.OPTION.value  # Usually USDC settled
    if market.get("linear", False):
        return Category.LINEAR.value  # USDT or USDC settled contracts
    if market.get("inverse", False):
        return Category.INVERSE.value  # Coin settled contracts

    # 3. Infer from 'type' and other info (less reliable fallback)
    market_type = market.get("type")  # CCXT's general type ('spot', 'swap', 'future', 'option')
    settle_coin = market.get("settle", "").upper()  # e.g., 'USDT', 'USDC', 'BTC', 'ETH'
    base_coin = market.get("base", "").upper()
    quote_coin = market.get("quote", "").upper()
    contract_type_info = str(info.get("contractType", "")).lower()  # E.g., 'LinearPerpetual', 'InversePerpetual'

    logger.debug(
        f"[_get_v5_category] Inferring category for {symbol}: type={market_type}, settle={settle_coin}, contractType='{contract_type_info}'"
    )

    if market_type == "spot":
        return Category.SPOT.value
    if market_type == "option":
        # V5 options are typically USDC settled
        if settle_coin == "USDC" or info.get("settleCoin") == "USDC":
            return Category.OPTION.value
        else:
            logger.warning(
                f"[_get_v5_category] Found option type for {symbol} but settle is '{settle_coin}' (expected USDC). Category unclear."
            )
            return None
    if market_type in ["swap", "future"]:
        # Check contractType field if available
        if contract_type_info == "linear" or "linear" in contract_type_info:
            return Category.LINEAR.value
        if contract_type_info == "inverse" or "inverse" in contract_type_info:
            return Category.INVERSE.value
        # Fallback: Check settle currency
        if settle_coin in ["USDT", "USDC"] or info.get("settleCoin") in ["USDT", "USDC"]:
            return Category.LINEAR.value
        # Fallback: Check if settle is the base currency (characteristic of inverse)
        if settle_coin == base_coin and settle_coin:
            return Category.INVERSE.value
        # Fallback: Check if quote contains USD (likely linear)
        if "USD" in quote_coin:
            return Category.LINEAR.value  # USDT or USDC
        # If still unsure, make a best guess (Linear is more common now)
        logger.warning(
            f"[_get_v5_category] Could not reliably determine contract category for {symbol} (type: {market_type}, settle: {settle_coin}). Assuming LINEAR."
        )
        return Category.LINEAR.value

    # If type is unknown or doesn't fit known patterns
    logger.warning(f"[_get_v5_category] Could not determine category for {symbol} with market type '{market_type}'.")
    return None


# --- Asynchronous Retry Decorator ---
T = TypeVar("T")  # Generic type variable for return value
FuncT = Callable[..., Coroutine[Any, Any, T]]  # Type hint for the decorated async function


def retry_api_call(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,  # Adds randomness to delay: delay * (1 +/- jitter)
    retry_on_exceptions: Sequence[type[Exception]] = (
        NetworkError,
        RateLimitExceeded,
        ExchangeNotAvailable,
        RequestTimeout,
    ),  # Exceptions that trigger a retry
    log_level: int = logging.WARNING,  # Level for retry attempt logs
    fail_log_level: int = logging.ERROR,  # Level when max retries are reached
):
    """Asynchronous decorator for retrying API calls with exponential backoff and jitter."""

    def decorator(func: FuncT) -> FuncT:
        async def async_wrapper(*args, **kwargs) -> T:
            # Allow overriding retries per-call via kwargs
            effective_max_retries = kwargs.pop("retries", max_retries)
            current_delay = initial_delay
            last_exception: Exception | None = None

            # Loop from 0 to max_retries (inclusive), meaning max_retries + 1 attempts total
            for attempt in range(effective_max_retries + 1):
                try:
                    # Attempt to call the original async function
                    return await func(*args, **kwargs)
                except retry_on_exceptions as e:
                    last_exception = e
                    # If this was the last attempt, log failure and re-raise
                    if attempt == effective_max_retries:
                        logger.log(
                            fail_log_level,
                            f"{Fore.RED}[{func.__name__}] Max retries ({effective_max_retries}) reached. Last error: {type(e).__name__}: {e}{Style.RESET_ALL}",
                        )
                        raise  # Re-raise the last caught exception
                    else:
                        # Calculate wait time with backoff and jitter
                        actual_jitter = random.uniform(-jitter, jitter)
                        wait_time = max(
                            0.1, current_delay + (current_delay * actual_jitter)
                        )  # Ensure delay is positive
                        # Log the retry attempt
                        logger.log(
                            log_level,
                            f"{Fore.YELLOW}[{func.__name__}] Attempt {attempt + 1}/{effective_max_retries + 1} failed: {type(e).__name__}. Retrying in {wait_time:.2f}s...{Style.RESET_ALL}",
                        )
                        # Wait before the next attempt
                        await asyncio.sleep(wait_time)
                        # Increase delay for the next potential retry
                        current_delay *= backoff_factor
                except Exception as e:
                    # Catch any other unexpected exceptions not in retry_on_exceptions
                    logger.error(
                        f"{Fore.RED}[{func.__name__}] Unhandled exception during attempt {attempt + 1}: {type(e).__name__}: {e}{Style.RESET_ALL}",
                        exc_info=True,
                    )
                    raise  # Re-raise immediately, don't retry unhandled exceptions

            # This part should theoretically not be reached if max_retries >= 0
            if last_exception:
                raise last_exception  # Should have been raised in the loop
            # Log a critical error if we somehow exit the loop without returning or raising
            logger.critical(f"[{func.__name__}] Retry logic completed unexpectedly without return or exception.")
            # The function expects type T, but we have nothing; raising is safer.
            raise RuntimeError(f"[{func.__name__}] Retry logic failed unexpectedly.")

        # Preserve original function's name and docstring for introspection
        async_wrapper.__name__ = func.__name__
        async_wrapper.__doc__ = func.__doc__
        return async_wrapper

    return decorator


# --- Exchange Initialization & Configuration ---
@retry_api_call(
    max_retries=2,  # Fewer retries for initialization
    initial_delay=3.0,  # Longer initial delay
    retry_on_exceptions=(NetworkError, ExchangeNotAvailable, RequestTimeout),  # Only retry network issues on init
    fail_log_level=logging.CRITICAL,  # Failure here is critical
)
async def initialize_bybit(config: Config) -> ccxt.bybit | None:
    """Initializes Bybit CCXT V5 instance, loads markets, checks auth, sets initial config."""
    func_name = "initialize_bybit"
    mode = "Testnet" if config.get("TESTNET_MODE", False) else "Mainnet"
    logger.info(f"{Fore.BLUE}[{func_name}] Initializing Bybit V5 ({mode})...{Style.RESET_ALL}")
    exchange: ccxt.bybit | None = None  # Initialize as None

    try:
        # Check if API keys are provided for private access
        has_keys = bool(config.get("API_KEY") and config.get("API_SECRET"))
        if not has_keys:
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] API Key/Secret missing. PUBLIC endpoints mode only.{Style.RESET_ALL}"
            )

        # Construct Broker ID if provided or use default based on version
        broker_id = config.get("BROKER_ID")
        if not broker_id and config.get("VERSION"):
            broker_id = f"PB_Pyrmethus{config['VERSION']}"  # Example format

        # CCXT Exchange Options
        exchange_options: dict[str, Any] = {
            "apiKey": config.get("API_KEY"),
            "secret": config.get("API_SECRET"),
            "enableRateLimit": True,  # Enable CCXT's built-in rate limiter
            "options": {
                "defaultType": config.get("EXPECTED_MARKET_TYPE", "swap"),  # Default market type for calls
                "adjustForTimeDifference": True,  # Attempt to sync clock with server
                "recvWindow": config.get("DEFAULT_RECV_WINDOW", 5000),  # API request validity window
                # 'verbose': True, # Uncomment for extreme CCXT request/response logging
            },
        }
        if broker_id:
            exchange_options["options"]["brokerId"] = broker_id
            logger.debug(f"[{func_name}] Using Broker ID: {broker_id}")

        logger.debug(f"[{func_name}] Instantiating CCXT Bybit with options: {exchange_options['options']}")
        exchange = ccxt.bybit(exchange_options)

        # Set sandbox mode if configured
        if config.get("TESTNET_MODE", False):
            exchange.set_sandbox_mode(True)
        logger.info(f"[{func_name}] {mode} mode active. API Endpoint Base: {exchange.urls['api']}")

        # --- Load Markets (Crucial Step) ---
        logger.info(f"[{func_name}] Loading markets via MarketCache (force reload)...")
        # Use the MarketCache instance to load markets
        load_success = await market_cache.load_markets(exchange, reload=True)

        if not load_success:
            # If market loading fails critically, abort initialization
            logger.critical(
                f"{Back.RED}[{func_name}] CRITICAL: Failed to load markets. Aborting initialization.{Style.RESET_ALL}"
            )
            # Attempt to close the partially created exchange instance
            if exchange and hasattr(exchange, "close"):
                logger.info(f"[{func_name}] Attempting to close partially initialized exchange instance...")
                try:
                    await exchange.close()
                except Exception as close_err:
                    logger.error(
                        f"[{func_name}] Error closing exchange during cleanup after market load failure: {close_err}"
                    )
            return None  # Return None to indicate failure

        # --- Validate Default Symbol ---
        default_symbol = config.get("SYMBOL")
        if not default_symbol:
            log_msg = "CRITICAL: Default 'SYMBOL' not defined in configuration."
            logger.critical(f"{Back.RED}[{func_name}] {log_msg}{Style.RESET_ALL}")
            if exchange and hasattr(exchange, "close"):
                await exchange.close()  # Cleanup
            return None
        if not market_cache.get_market(default_symbol):
            log_msg = f"CRITICAL: Market data for default symbol '{default_symbol}' NOT FOUND after loading markets."
            logger.critical(f"{Back.RED}[{func_name}] {log_msg}{Style.RESET_ALL}")
            if exchange and hasattr(exchange, "close"):
                await exchange.close()  # Cleanup
            return None
        logger.debug(f"[{func_name}] Default symbol '{default_symbol}' found in market cache.")

        # --- Authentication Check (if keys provided) ---
        if has_keys:
            logger.info(f"[{func_name}] Performing authentication check (fetching UNIFIED balance)...")
            try:
                # Fetching balance is a common way to verify API key validity and permissions
                # Using UNIFIED account type for V5
                await exchange.fetch_balance(params={"accountType": "UNIFIED"})
                logger.info(f"[{func_name}] Authentication check successful (Unified Balance fetched).")
            except AuthenticationError as auth_err:
                # Critical failure if authentication fails
                logger.critical(
                    f"{Back.RED}[{func_name}] CRITICAL: Authentication FAILED: {auth_err}. Check API Key/Secret/Permissions.{Style.RESET_ALL}"
                )
                send_sms_alert("[BybitHelper] CRITICAL: Bybit Authentication Failed!", config)
                if exchange and hasattr(exchange, "close"):
                    await exchange.close()  # Cleanup
                return None
            except (NetworkError, RequestTimeout, ExchangeNotAvailable) as net_err:
                # Critical if network fails during auth check (retries already handled by decorator)
                logger.critical(
                    f"{Back.RED}[{func_name}] CRITICAL Network Error during authentication check: {net_err}.{Style.RESET_ALL}"
                )
                if exchange and hasattr(exchange, "close"):
                    await exchange.close()  # Cleanup
                return None
            except ExchangeError as bal_err:
                # Warning for other exchange errors during balance fetch (e.g., temporary issue)
                logger.warning(
                    f"{Fore.YELLOW}[{func_name}] Warning during auth check (fetch_balance): {type(bal_err).__name__}: {bal_err}. Check API status or permissions.{Style.RESET_ALL}"
                )
                # Continue initialization, but be aware of potential issues
        else:
            logger.info(f"[{func_name}] Skipping authentication check (no API keys provided).")

        # --- Initial Configuration (Leverage/Margin - Optional) ---
        category = market_cache.get_category(default_symbol) if default_symbol else None
        # Only attempt leverage setting if keys are present and symbol is Linear/Inverse
        if has_keys and category in [Category.LINEAR, Category.INVERSE]:
            logger.info(f"[{func_name}] Attempting initial margin/leverage config for {default_symbol}...")
            try:
                default_margin_mode = config.get("DEFAULT_MARGIN_MODE", "isolated")
                if default_margin_mode == "isolated":
                    # Set initial leverage (implicitly sets isolated mode for the symbol in V5 UTA)
                    initial_leverage = 10  # Example default, consider adding to Config
                    logger.info(
                        f"[{func_name}] Setting default leverage to {initial_leverage}x for {default_symbol} (implies ISOLATED mode)."
                    )
                    # Call set_leverage helper function
                    await set_leverage(exchange, default_symbol, initial_leverage, config)  # Already has retries
                else:  # 'cross'
                    # For V5 UTA, 'cross' is account-wide (REGULAR_MARGIN or PORTFOLIO_MARGIN)
                    logger.info(
                        f"[{func_name}] Configured default margin mode is CROSS (account-level for UTA). Verifying account mode..."
                    )
                    acc_info = await fetch_account_info_bybit_v5(exchange, config)  # Already has retries
                    # V5 UTA Modes: REGULAR_MARGIN (Cross), PORTFOLIO_MARGIN (Cross), ISOLATED_MARGIN (Isolated)
                    if acc_info and acc_info.get("marginMode") == "ISOLATED_MARGIN":
                        logger.warning(
                            f"{Fore.YELLOW}[{func_name}] Account margin mode is ISOLATED_MARGIN, but config DEFAULT_MARGIN_MODE is 'cross'. Potential mismatch?{Style.RESET_ALL}"
                        )
                    elif acc_info:
                        logger.info(
                            f"[{func_name}] Account margin mode confirmed as '{acc_info.get('marginMode')}' (non-isolated, consistent with 'cross' config)."
                        )
                    else:
                        logger.warning(
                            f"[{func_name}] Could not verify account margin mode via fetch_account_info_bybit_v5."
                        )

            except Exception as config_err:
                # Log warning if initial config fails, but don't abort initialization
                logger.warning(
                    f"{Fore.YELLOW}[{func_name}] Could not apply initial margin/leverage config for {default_symbol}: {type(config_err).__name__}. Error: {config_err}{Style.RESET_ALL}"
                )
        elif not category or category not in [Category.LINEAR, Category.INVERSE]:
            logger.info(
                f"[{func_name}] Skipping initial margin/leverage setup (default symbol '{default_symbol}' is {category}, not Linear/Inverse)."
            )
        else:  # No keys
            logger.info(f"[{func_name}] Skipping initial margin/leverage setup (no API keys).")

        # --- Initialization Success ---
        logger.info(f"{Fore.GREEN}[{func_name}] Bybit V5 exchange initialized successfully.{Style.RESET_ALL}")
        return exchange

    # --- Exception Handling for Initialization Block ---
    except AuthenticationError as e:
        logger.critical(f"{Back.RED}[{func_name}] CRITICAL Authentication Error during setup: {e}.{Style.RESET_ALL}")
        send_sms_alert("[BybitHelper] CRITICAL: Bybit Authentication Failed!", config)
    except (NetworkError, ExchangeNotAvailable, RequestTimeout) as e:
        # This might catch errors if retries fail within this function's retry decorator
        logger.critical(
            f"{Back.RED}[{func_name}] CRITICAL Network Error during initialization (after retries): {e}.{Style.RESET_ALL}"
        )
    except ExchangeError as e:
        logger.critical(
            f"{Back.RED}[{func_name}] CRITICAL Exchange Error during initialization: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=False,
        )
        send_sms_alert(f"[BybitHelper] CRITICAL: Init ExchangeError: {type(e).__name__}", config)
    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.critical(
            f"{Back.RED}[{func_name}] CRITICAL Unexpected Error during initialization: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(f"[BybitHelper] CRITICAL: Init Unexpected Error: {type(e).__name__}", config)

    # --- Cleanup on Failure ---
    # If any exception occurred above, try to close the exchange instance if it was created
    if exchange and hasattr(exchange, "close"):
        try:
            logger.info(f"[{func_name}] Closing exchange instance due to initialization failure.")
            await exchange.close()
        except Exception as close_err:
            # Log error during cleanup but proceed to return None
            logger.error(f"[{func_name}] Error closing exchange instance during cleanup: {close_err}")

    return None  # Return None indicating initialization failed


# --- Account Functions ---


@retry_api_call()
async def fetch_account_info_bybit_v5(exchange: ccxt.bybit, config: Config) -> dict | None:
    """Fetches detailed account information for the V5 Unified Trading Account.

    Args:
        exchange: Initialized ccxt.bybit instance.
        config: Configuration object (used for logging/context).

    Returns:
        Dictionary containing account info (margin mode, upgrade status, etc.) or None on failure.
        Structure example: {'unifiedMarginStatus': 1, 'marginMode': 'REGULAR_MARGIN', ...}
    """
    func_name = "fetch_account_info_bybit_v5"
    log_prefix = f"[{func_name}]"
    logger.debug(f"{log_prefix} Fetching V5 account info...")

    # Check if exchange object is valid
    if not exchange or not hasattr(exchange, "private_get_v5_account_info"):
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid exchange object or method 'private_get_v5_account_info' not available.{Style.RESET_ALL}"
        )
        return None

    try:
        # Use the specific V5 implicit endpoint call via CCXT
        response = await exchange.private_get_v5_account_info()
        logger.debug(f"{log_prefix} Raw response: {response}")

        # Validate the response structure
        if (
            response
            and isinstance(response, dict)
            and response.get("retCode") == 0
            and isinstance(response.get("result"), dict)
        ):
            account_info = response["result"]
            margin_mode = account_info.get("marginMode", "N/A")
            status = account_info.get("unifiedMarginStatus", "N/A")  # 1: Regular UTA, 2: Pro UTA
            dcp_status = account_info.get("dcpStatus", "N/A")  # Disconnected Program status

            logger.info(
                f"{Fore.GREEN}{log_prefix} Success. Margin Mode: {margin_mode}, UTA Status: {status}, DCP Status: {dcp_status}{Style.RESET_ALL}"
            )
            return account_info
        else:
            # Log error details if response format is unexpected or indicates failure
            ret_code = response.get("retCode", "N/A") if isinstance(response, dict) else "N/A"
            ret_msg = (
                response.get("retMsg", "Unknown error") if isinstance(response, dict) else "Invalid response format"
            )
            logger.error(
                f"{Fore.RED}{log_prefix} Failed to fetch account info. Code: {ret_code}, Msg: {ret_msg}{Style.RESET_ALL}"
            )
            return None

    except AuthenticationError as e:
        logger.error(f"{Fore.RED}{log_prefix} Authentication error: {e}{Style.RESET_ALL}")
        return None
    except (NetworkError, ExchangeNotAvailable, RequestTimeout) as e:
        # These are handled by the retry decorator, log warning and re-raise
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Network/Availability error: {type(e).__name__}. Retry handled by decorator.{Style.RESET_ALL}"
        )
        raise  # Re-raise for decorator
    except ExchangeError as e:
        # Handle other exchange-specific errors
        logger.error(f"{Fore.RED}{log_prefix} Exchange error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False)
        return None
    except Exception as e:
        # Catch any unexpected errors
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=True
        )
        return None


@retry_api_call()
async def set_leverage(exchange: ccxt.bybit, symbol: str, leverage: int, config: Config) -> bool:
    """Sets leverage for a symbol (Linear/Inverse), implicitly setting ISOLATED mode for that symbol in V5 UTA."""
    func_name = "set_leverage"
    log_prefix = f"[{func_name} ({symbol} -> {leverage}x)]"

    # Validate leverage input
    if leverage <= 0:
        logger.error(
            f"{Fore.RED}{log_prefix} Leverage must be a positive integer. Received: {leverage}.{Style.RESET_ALL}"
        )
        return False

    # Check market category from cache
    category = market_cache.get_category(symbol)
    if not category or category not in [Category.LINEAR, Category.INVERSE]:
        logger.error(
            f"{Fore.RED}{log_prefix} Leverage setting only applicable for LINEAR/INVERSE categories. Symbol '{symbol}' is type: {category}.{Style.RESET_ALL}"
        )
        return False

    # Get market data for validation
    market = market_cache.get_market(symbol)
    if not market:
        logger.error(
            f"{Fore.RED}{log_prefix} Market data not found for {symbol}. Cannot validate leverage limits.{Style.RESET_ALL}"
        )
        # Proceed cautiously without limit check, or return False? Let's proceed with warning.
        logger.warning(f"{log_prefix} Proceeding without leverage limit validation.")
    else:
        # Validate against market leverage limits if available
        try:
            limits_leverage = market.get("limits", {}).get("leverage", {})
            max_lev_raw = limits_leverage.get("max")
            min_lev_raw = limits_leverage.get("min")
            # Use safe conversion for limits
            max_lev = safe_decimal_conversion(max_lev_raw, default=None)
            min_lev = safe_decimal_conversion(min_lev_raw, default=Decimal(1))  # Default min to 1 if missing

            if max_lev is not None and min_lev is not None:
                if not (min_lev <= Decimal(leverage) <= max_lev):
                    logger.error(
                        f"{Fore.RED}{log_prefix} Requested leverage {leverage}x is outside the allowed range [{min_lev}x - {max_lev}x] for {symbol}.{Style.RESET_ALL}"
                    )
                    return False
                else:
                    logger.debug(
                        f"{log_prefix} Leverage {leverage}x is within market limits [{min_lev}x - {max_lev}x]."
                    )
            else:
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} Could not parse leverage limits (min={min_lev_raw}, max={max_lev_raw}). Proceeding without range check.{Style.RESET_ALL}"
                )
        except Exception as e:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Error validating leverage limits: {e}. Proceeding without range check.{Style.RESET_ALL}"
            )

    # Prepare parameters for Bybit V5 setLeverage endpoint
    # Note: Setting leverage per symbol in UTA automatically sets that symbol to ISOLATED margin mode.
    params = {
        "category": category.value,
        "buyLeverage": str(leverage),  # Bybit expects string values for leverage
        "sellLeverage": str(leverage),  # Set same for buy and sell
    }
    logger.info(
        f"{Fore.CYAN}{log_prefix} Sending request with params: {params}... (This implies ISOLATED mode for {symbol}){Style.RESET_ALL}"
    )

    try:
        # Use CCXT's set_leverage method, passing V5 params
        response = await exchange.set_leverage(leverage, symbol, params=params)
        # CCXT's set_leverage might return None or {} on success for Bybit
        logger.debug(f"{log_prefix} Raw response from exchange.set_leverage: {response}")
        # We assume success if no exception is raised, but Bybit might have specific success/failure codes inside errors.
        logger.info(
            f"{Fore.GREEN}{log_prefix} Request successful (Leverage set to {leverage}x, mode is ISOLATED for {symbol}).{Style.RESET_ALL}"
        )
        return True

    except ExchangeError as e:
        error_str = str(e).lower()
        # Check for common Bybit error codes indicating success or known issues
        # Bybit error codes: 110043 (Leverage not modified), 34036 (Not modified), 110021 (Hedge mode/positionIdx issue)
        if "leverage not modified" in error_str or "110043" in str(e) or "34036" in str(e):
            # If leverage is already set to the desired value, treat as success
            logger.info(
                f"{Fore.YELLOW}{log_prefix} Leverage already set to {leverage}x (or not modified). Considered success.{Style.RESET_ALL}"
            )
            return True
        elif "position idx" in error_str or "110021" in str(e):
            # This error might occur in Hedge Mode if positionIdx isn't handled correctly elsewhere
            logger.error(
                f"{Fore.RED}{log_prefix} Failed: {e}. Potential Hedge Mode issue? Check positionIdx settings.{Style.RESET_ALL}"
            )
            return False
        else:
            # Log other unexpected exchange errors
            logger.error(
                f"{Fore.RED}{log_prefix} ExchangeError setting leverage: {type(e).__name__}: {e}{Style.RESET_ALL}",
                exc_info=False,
            )
            return False
    except (NetworkError, ExchangeNotAvailable, RequestTimeout) as e:
        # Handled by retry decorator
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Network/Availability error: {type(e).__name__}. Retry handled.{Style.RESET_ALL}"
        )
        raise  # Re-raise for decorator
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error setting leverage: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return False


@retry_api_call()
async def fetch_usdt_balance(exchange: ccxt.bybit, config: Config) -> tuple[Decimal | None, Decimal | None]:
    """Fetches USDT total equity and available balance from Bybit V5 UNIFIED account."""
    func_name = "fetch_usdt_balance"
    log_prefix = f"[{func_name}]"
    # Get the specific symbol string used for USDT (usually 'USDT')
    usdt_symbol = config.get("USDT_SYMBOL", "USDT")

    logger.debug(f"{log_prefix} Fetching balance for UNIFIED account...")
    try:
        # Fetch balance data using CCXT, specifying the V5 account type
        balance_data = await exchange.fetch_balance(params={"accountType": "UNIFIED"})
        logger.debug(f"{log_prefix} Raw balance data received: {balance_data}")

        total_equity: Decimal | None = None
        available_balance: Decimal | None = None

        # --- Primary Method: Parse V5 'info' structure ---
        # This is generally more reliable for V5 specific details
        info_result = balance_data.get("info", {}).get("result", {})
        if info_result and isinstance(info_result.get("list"), list):
            unified_account_info = next(
                (acc for acc in info_result["list"] if acc.get("accountType") == "UNIFIED"), None
            )
            if unified_account_info and isinstance(unified_account_info, dict):
                # Total equity for the UNIFIED account
                total_equity = safe_decimal_conversion(unified_account_info.get("totalEquity"))
                logger.debug(f"{log_prefix} Equity from info.result.list[UNIFIED].totalEquity: {total_equity}")

                # Find USDT details within the 'coin' list of the UNIFIED account
                coin_list = unified_account_info.get("coin", [])
                if isinstance(coin_list, list):
                    usdt_coin_info = next((coin for coin in coin_list if coin.get("coin") == usdt_symbol), None)
                    if usdt_coin_info and isinstance(usdt_coin_info, dict):
                        # V5 uses 'availableToWithdraw' or 'availableBalance' for free balance
                        available_str = usdt_coin_info.get("availableToWithdraw") or usdt_coin_info.get(
                            "availableBalance"
                        )
                        available_balance = safe_decimal_conversion(available_str)
                        logger.debug(
                            f"{log_prefix} Available balance from info...coin['{usdt_symbol}']: {available_balance} (Raw: '{available_str}')"
                        )
                    else:
                        logger.warning(
                            f"{log_prefix} '{usdt_symbol}' coin data not found within UNIFIED account coin list."
                        )
                else:
                    logger.warning(f"{log_prefix} UNIFIED account 'coin' list is not a list or missing.")
            else:
                logger.warning(f"{log_prefix} UNIFIED account details not found in info.result.list.")
        else:
            logger.warning(f"{log_prefix} info.result.list is missing or not a list in balance response.")

        # --- Fallback Method: Use top-level CCXT structure ---
        # This relies on CCXT's parsing, which might be less precise for V5 details
        if total_equity is None:
            # CCXT often puts total equity per asset in 'total'
            total_equity_ccxt = balance_data.get("total", {}).get(usdt_symbol)
            if total_equity_ccxt is not None:
                total_equity = safe_decimal_conversion(total_equity_ccxt)
                logger.debug(f"{log_prefix} Equity from fallback CCXT top-level 'total.{usdt_symbol}': {total_equity}")

        if available_balance is None:
            # CCXT often puts available balance per asset in 'free'
            available_balance_ccxt = balance_data.get("free", {}).get(usdt_symbol)
            if available_balance_ccxt is not None:
                available_balance = safe_decimal_conversion(available_balance_ccxt)
                logger.debug(
                    f"{log_prefix} Available from fallback CCXT top-level 'free.{usdt_symbol}': {available_balance}"
                )

        # --- Final Processing and Return ---
        if total_equity is None:
            logger.warning(
                f"{log_prefix} Could not determine total USDT equity from V5 info or CCXT structure. Defaulting to 0."
            )
            final_equity = Decimal("0.0")
        else:
            # Ensure equity is not negative
            final_equity = max(Decimal("0.0"), total_equity)

        if available_balance is None:
            logger.warning(
                f"{log_prefix} Could not determine available USDT balance from V5 info or CCXT structure. Defaulting to 0."
            )
            final_available = Decimal("0.0")
        else:
            # Ensure available balance is not negative
            final_available = max(Decimal("0.0"), available_balance)

        logger.info(
            f"{Fore.GREEN}{log_prefix} Success - Total Equity: {final_equity:.4f}, Available: {final_available:.4f} {usdt_symbol}{Style.RESET_ALL}"
        )
        return final_equity, final_available

    except AuthenticationError as e:
        logger.error(f"{Fore.RED}{log_prefix} Authentication error: {e}{Style.RESET_ALL}")
        return None, None
    except (NetworkError, ExchangeNotAvailable, RequestTimeout) as e:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Network/Availability error: {type(e).__name__}. Retry handled.{Style.RESET_ALL}"
        )
        raise  # Re-raise for decorator
    except ExchangeError as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Exchange error fetching balance: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=False,
        )
        return None, None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error fetching balance: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None, None


# --- Market Data Functions ---


# Note: fetch_ohlcv_paginated remains largely the same as it already had robust pagination and retry logic per chunk.
# Added minor logging improvements and ensures config is passed.
@retry_api_call()  # Outer retry for the initial setup phase
async def fetch_ohlcv_paginated(
    exchange: ccxt.bybit,
    symbol: str,
    timeframe: str,
    since: int | None = None,
    limit: int | None = None,
    config: Config = None,  # Made config required
) -> pd.DataFrame | list[list] | None:
    """Fetches OHLCV data, handling pagination and retries internally per chunk."""
    func_name = "fetch_ohlcv_paginated"
    log_prefix = f"[{func_name} ({symbol}, {timeframe})]"

    # --- Pre-checks ---
    if not config:
        logger.error(f"{Fore.RED}{log_prefix} Configuration object is required.{Style.RESET_ALL}")
        return None
    if not exchange or not hasattr(exchange, "fetch_ohlcv"):
        logger.error(f"{Fore.RED}{log_prefix} Invalid exchange object or fetch_ohlcv method missing.{Style.RESET_ALL}")
        return None

    market = market_cache.get_market(symbol)
    category = market_cache.get_category(symbol)
    if not market or not category:
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid market/category for {symbol}. Cannot determine fetch parameters.{Style.RESET_ALL}"
        )
        return None

    # Determine fetch parameters based on category and CCXT capabilities
    try:
        # Bybit V5 max limit per request is 1000 candles
        fetch_limit_per_req = 1000
        timeframe_duration_ms = exchange.parse_timeframe(timeframe) * 1000
        if timeframe_duration_ms <= 0:
            raise ValueError("Invalid timeframe duration")
    except (ValueError, KeyError) as e:
        logger.error(f"{Fore.RED}{log_prefix} Invalid timeframe '{timeframe}': {e}.{Style.RESET_ALL}")
        return None

    # --- Pagination Logic ---
    all_candles: list[list] = []
    current_since = since
    # Safety break to prevent infinite loops
    max_loops = 200  # Limit total number of API calls
    loops = 0
    # Use config for retry settings per chunk fetch
    retries_per_chunk = config.get("RETRY_COUNT", 3)
    base_retry_delay = config.get("RETRY_DELAY_SECONDS", 1.0)
    backoff_factor = 2.0  # Standard exponential backoff
    # Small delay between successful chunk fetches to respect rate limits
    delay_between_chunks = (
        exchange.rateLimit / 1000 if exchange.enableRateLimit and exchange.rateLimit > 0 else 0.2
    ) + 0.05  # Add small buffer

    logger.info(
        f"{Fore.BLUE}{log_prefix} Starting fetch. Target limit: {limit or 'All'}, Candles/Call: {fetch_limit_per_req}, Category: {category.value}...{Style.RESET_ALL}"
    )
    params = {"category": category.value}  # Essential param for V5

    try:
        while loops < max_loops:
            loops += 1
            # Check if desired total limit is reached
            if limit is not None and len(all_candles) >= limit:
                logger.info(f"{log_prefix} Reached desired total limit of {limit} candles.")
                break

            # Determine limit for the current API call
            current_fetch_limit = fetch_limit_per_req
            if limit is not None:
                remaining = limit - len(all_candles)
                if remaining <= 0:
                    break  # Should have been caught above, but safety check
                current_fetch_limit = min(fetch_limit_per_req, remaining)

            logger.debug(f"{log_prefix} Loop {loops}, Fetching since={current_since} (Limit: {current_fetch_limit})...")

            # --- Inner retry logic for fetching one chunk ---
            candles_chunk: list[list] | None = None
            last_fetch_error: Exception | None = None
            for attempt in range(retries_per_chunk + 1):
                try:
                    # Fetch one chunk of OHLCV data
                    candles_chunk = await exchange.fetch_ohlcv(
                        symbol, timeframe, since=current_since, limit=current_fetch_limit, params=params
                    )
                    last_fetch_error = None  # Reset error on success
                    break  # Exit retry loop on success
                except (NetworkError, RequestTimeout, ExchangeNotAvailable, RateLimitExceeded) as e:
                    last_fetch_error = e
                    if attempt == retries_per_chunk:
                        logger.error(
                            f"{Fore.RED}{log_prefix} Chunk fetch failed after {retries_per_chunk + 1} attempts. Last Error: {type(e).__name__}: {e}{Style.RESET_ALL}"
                        )
                        break  # Exit retry loop after max retries
                    else:
                        # Calculate wait time with backoff and jitter
                        wait_time = base_retry_delay * (backoff_factor**attempt) + random.uniform(
                            0, base_retry_delay * 0.1
                        )
                        logger.warning(
                            f"{Fore.YELLOW}{log_prefix} Chunk attempt {attempt + 1} failed: {type(e).__name__}. Retrying in {wait_time:.2f}s...{Style.RESET_ALL}"
                        )
                        await asyncio.sleep(wait_time)
                except ExchangeError as e:
                    # Non-retryable exchange error for this chunk
                    last_fetch_error = e
                    logger.error(
                        f"{Fore.RED}{log_prefix} ExchangeError fetching chunk: {type(e).__name__}: {e}. Aborting chunk fetch.{Style.RESET_ALL}"
                    )
                    break  # Exit retry loop
                except Exception as e:
                    # Unexpected error during chunk fetch
                    last_fetch_error = e
                    logger.error(
                        f"{Fore.RED}{log_prefix} Unexpected error fetching chunk: {type(e).__name__}: {e}{Style.RESET_ALL}",
                        exc_info=True,
                    )
                    break  # Exit retry loop

            # If chunk fetch failed after retries, abort the entire pagination process
            if last_fetch_error:
                logger.error(
                    f"{Fore.RED}{log_prefix} Aborting pagination due to persistent chunk fetch failure.{Style.RESET_ALL}"
                )
                break  # Exit the main pagination loop

            # If no candles are returned, we've likely reached the end of available data
            if not candles_chunk:
                logger.info(f"{log_prefix} No more candles returned by API. Fetch complete.")
                break  # Exit the main pagination loop

            # --- Process valid chunk ---
            # Filter potential duplicates (sometimes exchanges return overlapping candles)
            if all_candles and candles_chunk and candles_chunk[0][0] <= all_candles[-1][0]:
                initial_chunk_len = len(candles_chunk)
                # Keep only candles with timestamp strictly greater than the last stored candle
                candles_chunk = [c for c in candles_chunk if c[0] > all_candles[-1][0]]
                if len(candles_chunk) < initial_chunk_len:
                    logger.debug(
                        f"{log_prefix} Removed {initial_chunk_len - len(candles_chunk)} duplicate/overlapping candle(s) from chunk."
                    )
                if not candles_chunk:
                    logger.debug(f"{log_prefix} All candles in the new chunk were duplicates. Stopping.")
                    break  # Exit main loop if only duplicates received

            # If a total limit is set, trim the chunk if necessary
            if limit is not None:
                needed = limit - len(all_candles)
                candles_chunk = candles_chunk[:needed]

            # If after filtering/trimming the chunk is empty, stop
            if not candles_chunk:
                break

            # Add the processed chunk to the main list
            all_candles.extend(candles_chunk)

            # Log progress
            last_timestamp = candles_chunk[-1][0]
            first_timestamp = candles_chunk[0][0]
            log_range = f"Range Ts: {first_timestamp} to {last_timestamp}"
            # Try to format timestamps if pandas is available
            if pd:
                try:
                    dt_fmt = "%Y-%m-%d %H:%M:%S %Z"
                    first_dt = pd.to_datetime(first_timestamp, unit="ms", utc=True).strftime(dt_fmt)
                    last_dt = pd.to_datetime(last_timestamp, unit="ms", utc=True).strftime(dt_fmt)
                    log_range = f"Range Dt: {first_dt} to {last_dt}"
                except Exception:
                    pass  # Ignore formatting errors
            logger.info(
                f"{log_prefix} Fetched {len(candles_chunk)} candles. {log_range}. Total collected: {len(all_candles)}"
            )

            # --- Prepare for next iteration ---
            # Set 'since' for the next request to the timestamp of the last candle + 1ms
            current_since = last_timestamp + 1
            # Pause briefly between successful fetches
            await asyncio.sleep(delay_between_chunks)

            # Check if the exchange returned fewer candles than requested (often indicates end of data)
            # Only break if we didn't set a specific 'limit' argument ourselves
            if limit is None and len(candles_chunk) < current_fetch_limit:
                logger.info(
                    f"{log_prefix} Received less than requested limit ({len(candles_chunk)} < {current_fetch_limit}). Assuming end of available data."
                )
                break  # Exit the main pagination loop
        # --- End of pagination loop ('while loops < max_loops:') ---

        if loops >= max_loops:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Reached maximum loop limit ({max_loops}). Fetch may be incomplete.{Style.RESET_ALL}"
            )

        logger.info(f"{log_prefix} Finished fetching. Total raw candles collected: {len(all_candles)}")
        if not all_candles:
            logger.warning(f"{log_prefix} No candles found for the specified criteria.")
            # Return empty structure matching expected type
            return pd.DataFrame() if pd else []

        # --- Process into DataFrame or List ---
        if pd:
            # Use pandas if available for structured data
            try:
                df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                # Convert timestamp to datetime index (UTC)
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("datetime", inplace=True)
                # Ensure numeric types for OHLCV columns
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")  # Coerce errors to NaN
                # Remove potential duplicate timestamps (keeping first occurrence)
                initial_len = len(df)
                df = df[~df.index.duplicated(keep="first")]
                if len(df) < initial_len:
                    logger.debug(
                        f"{log_prefix} Removed {initial_len - len(df)} duplicate timestamps during DataFrame processing."
                    )
                # Sort by datetime index
                df.sort_index(inplace=True)
                # Check for NaNs introduced by conversion errors
                nan_counts = df.isnull().sum()
                if nan_counts.sum() > 0:
                    logger.warning(
                        f"{log_prefix} NaN values found after numeric conversion: {nan_counts[nan_counts > 0].to_dict()}"
                    )
                logger.info(
                    f"{Fore.GREEN}{log_prefix} Processed {len(df)} unique candles into DataFrame.{Style.RESET_ALL}"
                )
                return df
            except Exception as df_err:
                logger.error(f"{log_prefix} Error processing data into pandas DataFrame: {df_err}", exc_info=True)
                # Fallback to returning the raw list if DataFrame processing fails
                all_candles.sort(key=lambda x: x[0])  # Ensure sorted list
                logger.warning(f"{log_prefix} Falling back to returning sorted list of candles.")
                return all_candles
        else:
            # Return raw list if pandas is not available
            all_candles.sort(key=lambda x: x[0])  # Ensure sorted list
            logger.info(
                f"{Fore.GREEN}{log_prefix} Processed {len(all_candles)} unique candles (returning as List).{Style.RESET_ALL}"
            )
            return all_candles

    # --- Outer Exception Handling (Catches errors not handled by inner loops/retries) ---
    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
        # Should ideally be caught by the decorator, but catch here as safety net
        logger.error(
            f"{Fore.RED}{log_prefix} Unhandled API error during pagination: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=False,
        )
    except ExchangeError as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unhandled Exchange error during pagination: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=False,
        )
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error during pagination: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )

    # Attempt to return partial data if some candles were collected before error
    if all_candles:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Returning partial data ({len(all_candles)} candles) due to error during fetch.{Style.RESET_ALL}"
        )
        if pd:
            try:  # Try processing partial data into DataFrame
                df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("datetime", inplace=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df[~df.index.duplicated(keep="first")]
                df.sort_index(inplace=True)
                return df
            except Exception as final_proc_err:
                logger.error(
                    f"{Fore.RED}{log_prefix} Error processing partial DataFrame on error exit: {final_proc_err}{Style.RESET_ALL}"
                )
                all_candles.sort(key=lambda x: x[0])
                return all_candles  # Fallback list
        else:
            all_candles.sort(key=lambda x: x[0])
            return all_candles  # Return partial list
    else:
        # Return None if no data collected and error occurred
        return None


@retry_api_call()
async def fetch_ticker_validated(exchange: ccxt.bybit, symbol: str, config: Config) -> dict | None:
    """Fetches ticker, validates timestamp and essential keys."""
    func_name = "fetch_ticker_validated"
    log_prefix = f"[{func_name} ({symbol})]"
    logger.debug(f"{log_prefix} Fetching ticker...")

    category = market_cache.get_category(symbol)
    if not category:
        logger.error(
            f"{Fore.RED}{log_prefix} Cannot determine category for {symbol}. Cannot fetch ticker.{Style.RESET_ALL}"
        )
        return None

    params = {"category": category.value}  # V5 requires category
    try:
        ticker = await exchange.fetch_ticker(symbol, params=params)

        # Basic validation of the returned ticker structure
        if not ticker or not isinstance(ticker, dict):
            logger.error(f"{Fore.RED}{log_prefix} Received empty or invalid ticker response.{Style.RESET_ALL}")
            return None

        # Check for essential keys expected in a CCXT ticker
        required_keys = ["symbol", "last", "bid", "ask", "timestamp", "datetime"]
        missing_keys = [k for k in required_keys if k not in ticker or ticker[k] is None]
        if missing_keys:
            logger.error(
                f"{Fore.RED}{log_prefix} Ticker response missing essential keys: {missing_keys}. Data: {str(ticker)[:200]}...{Style.RESET_ALL}"
            )
            return None  # Fail if core data is missing

        # --- Timestamp Validation ---
        ticker_time_ms = ticker.get("timestamp")
        current_time_ms = int(time.time() * 1000)
        # Define acceptable age range in seconds (allow slightly in future for clock skew)
        max_age_seconds = 90
        min_age_seconds = -10  # Allow ~10s future timestamp
        max_diff_ms = max_age_seconds * 1000
        min_diff_ms = min_age_seconds * 1000  # Negative value

        log_timestamp_msg = f"Timestamp: {ticker.get('datetime', 'N/A')}"  # Default log message
        is_timestamp_valid = False

        if ticker_time_ms is None:
            log_timestamp_msg = f"{Fore.YELLOW}Timestamp: Missing{Style.RESET_ALL}"
        elif not isinstance(ticker_time_ms, int):
            log_timestamp_msg = (
                f"{Fore.YELLOW}Timestamp: Invalid Type ({type(ticker_time_ms).__name__}){Style.RESET_ALL}"
            )
        else:
            time_diff_ms = current_time_ms - ticker_time_ms
            age_s = time_diff_ms / 1000.0
            dt_str = ticker.get("datetime", f"ts({ticker_time_ms})")  # Use CCXT datetime if available

            # Check if age is outside the acceptable range
            if time_diff_ms > max_diff_ms or time_diff_ms < min_diff_ms:
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} Ticker timestamp ({dt_str}) seems stale or invalid. Age: {age_s:.1f}s (Allowed Range: {min_age_seconds}s to {max_age_seconds}s).{Style.RESET_ALL}"
                )
                log_timestamp_msg = f"{Fore.YELLOW}Timestamp: Stale/Invalid (Age: {age_s:.1f}s){Style.RESET_ALL}"
                # Decide whether to fail or just warn based on staleness tolerance
                # For now, let's fail if timestamp is present but invalid
                # return None # Uncomment to enforce strict timestamp validation
            else:
                is_timestamp_valid = True  # Timestamp is within acceptable range
                log_timestamp_msg = f"Timestamp: OK (Age: {age_s:.1f}s)"

        # Fail if timestamp was present but deemed invalid by age check
        if not is_timestamp_valid and isinstance(ticker_time_ms, int):
            logger.error(
                f"{Fore.RED}{log_prefix} Ticker timestamp validation failed. {log_timestamp_msg}{Style.RESET_ALL}"
            )
            return None

        # Log success with key ticker info
        logger.info(
            f"{Fore.GREEN}{log_prefix} Fetched OK: Last={ticker.get('last')}, Bid={ticker.get('bid')}, Ask={ticker.get('ask')} | {log_timestamp_msg}{Style.RESET_ALL}"
        )
        return ticker

    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API error: {type(e).__name__}. Retry handled.{Style.RESET_ALL}")
        raise  # Re-raise for decorator
    except AuthenticationError as e:
        # Should not happen for public endpoint, but catch just in case
        logger.error(f"{Fore.RED}{log_prefix} Authentication error unexpectedly occurred: {e}{Style.RESET_ALL}")
        return None
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Exchange error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False)
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=True
        )
        return None


@retry_api_call()
async def fetch_funding_rate(
    exchange: ccxt.bybit, symbol: str, config: Config, fetch_next: bool = False
) -> Decimal | None:
    """Fetches current (last settled) or predicted next funding rate for a perpetual swap."""
    func_name = "fetch_funding_rate"
    rate_type_desc = "Next Predicted" if fetch_next else "Last Settled"
    log_prefix = f"[{func_name} ({symbol} - {rate_type_desc})]"

    market = market_cache.get_market(symbol)
    if not market or not market.get("swap", False):
        logger.error(
            f"{Fore.RED}{log_prefix} Symbol '{symbol}' is not identified as a swap/perpetual. Market data: {market}{Style.RESET_ALL}"
        )
        return None
    category = market_cache.get_category(symbol)
    if category not in [Category.LINEAR, Category.INVERSE]:
        logger.error(
            f"{Fore.RED}{log_prefix} Funding rates require LINEAR or INVERSE category. Found: {category} for {symbol}.{Style.RESET_ALL}"
        )
        return None

    params = {"category": category.value, "symbol": symbol}  # Required V5 params
    logger.debug(f"{log_prefix} Fetching with params: {params}")

    try:
        if fetch_next:
            # Fetching the *next* funding rate requires fetching the ticker, as it's included there
            logger.debug(f"{log_prefix} Fetching ticker to get next funding rate...")
            ticker = await fetch_ticker_validated(exchange, symbol, config)  # Use validated fetch
            if not ticker:
                logger.error(
                    f"{Fore.RED}{log_prefix} Failed to fetch validated ticker needed for next funding rate.{Style.RESET_ALL}"
                )
                return None

            # Extract funding rate info from the 'info' field of the ticker
            ticker_info = ticker.get("info", {})
            next_rate_str = ticker_info.get("fundingRate")  # V5 field name
            next_time_ms = ticker_info.get("nextFundingTime")  # V5 field name

            if next_rate_str is not None:
                rate_decimal = safe_decimal_conversion(next_rate_str)
                if rate_decimal is None:
                    logger.error(
                        f"{Fore.RED}{log_prefix} Could not parse 'fundingRate' ('{next_rate_str}') from ticker info.{Style.RESET_ALL}"
                    )
                    return None

                # Format next funding time for logging if possible
                next_dt_str = "N/A"
                if next_time_ms:
                    try:
                        ts = int(next_time_ms)
                        if pd:  # Use pandas for nice formatting if available
                            next_dt_str = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce").strftime(
                                "%Y-%m-%d %H:%M:%S %Z"
                            )
                        else:  # Basic formatting otherwise
                            next_dt_str = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.gmtime(ts / 1000))
                    except (ValueError, TypeError):
                        next_dt_str = str(next_time_ms)  # Fallback to raw string

                logger.info(
                    f"{Fore.GREEN}{log_prefix} Success - Next Rate: {rate_decimal:.8f} (Expected At: {next_dt_str}){Style.RESET_ALL}"
                )
                return rate_decimal
            else:
                logger.error(
                    f"{Fore.RED}{log_prefix} 'fundingRate' field not found in ticker info. Ticker Info: {str(ticker_info)[:200]}...{Style.RESET_ALL}"
                )
                return None
        else:
            # Fetching the *last settled* funding rate requires fetch_funding_history
            logger.debug(f"{log_prefix} Fetching funding history (limit=1) for last settled rate...")
            # Limit=1 gets the most recent settled rate
            history = await exchange.fetch_funding_history(symbol=symbol, limit=1, params=params)

            if history and isinstance(history, list) and len(history) > 0:
                last_interval = history[0]  # Get the most recent entry
                # V5 funding rate is often nested inside 'info' in CCXT's parsed structure
                rate_str = last_interval.get("info", {}).get("fundingRate")
                timestamp_ms = last_interval.get("timestamp")
                dt_str = last_interval.get("datetime")  # CCXT provides parsed datetime

                if rate_str is not None:
                    rate_decimal = safe_decimal_conversion(rate_str)
                    if rate_decimal is None:
                        logger.error(
                            f"{Fore.RED}{log_prefix} Could not parse 'fundingRate' ('{rate_str}') from funding history info.{Style.RESET_ALL}"
                        )
                        return None
                    logger.info(
                        f"{Fore.GREEN}{log_prefix} Success - Last Settled Rate: {rate_decimal:.8f} (Settled Time: {dt_str or timestamp_ms}){Style.RESET_ALL}"
                    )
                    return rate_decimal
                else:
                    logger.error(
                        f"{Fore.RED}{log_prefix} 'fundingRate' field not found in funding history info. History[0]: {str(last_interval)[:200]}...{Style.RESET_ALL}"
                    )
                    return None
            else:
                logger.error(
                    f"{Fore.RED}{log_prefix} Failed to fetch funding history or history is empty.{Style.RESET_ALL}"
                )
                return None

    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API error: {type(e).__name__}. Retry handled.{Style.RESET_ALL}")
        raise  # Re-raise for decorator
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Exchange error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False)
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=True
        )
        return None


@retry_api_call()
async def fetch_l2_order_book_validated(exchange: ccxt.bybit, symbol: str, limit: int, config: Config) -> dict | None:
    """Fetches L2 order book and performs basic validation."""
    func_name = "fetch_l2_order_book_validated"
    log_prefix = f"[{func_name} ({symbol}, limit={limit})]"
    logger.debug(f"{log_prefix} Fetching L2 order book...")

    category = market_cache.get_category(symbol)
    if not category:
        logger.error(
            f"{Fore.RED}{log_prefix} Cannot determine category for {symbol}. Cannot fetch order book.{Style.RESET_ALL}"
        )
        return None

    # Check if limit is valid/optimal for the category (informational)
    # Bybit V5 L2 Order Book Limits: Linear/Inverse: [1, 50, 200, 500], Spot: [1, 50, 200], Option: [25, 100, 200]
    valid_limits = {
        Category.SPOT: [1, 50, 200],
        Category.LINEAR: [1, 50, 200, 500],
        Category.INVERSE: [1, 50, 200, 500],
        Category.OPTION: [25, 100, 200],
    }
    category_valid_limits = valid_limits.get(category)
    if category_valid_limits and limit not in category_valid_limits:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Requested limit {limit} is not standard for {category.value}. Valid: {category_valid_limits}. Proceeding, but API might adjust or reject.{Style.RESET_ALL}"
        )

    params = {"category": category.value}  # V5 requires category
    try:
        # Fetch L2 order book using CCXT
        order_book = await exchange.fetch_l2_order_book(symbol, limit=limit, params=params)

        # --- Validation ---
        if not order_book or not isinstance(order_book, dict):
            logger.error(f"{Fore.RED}{log_prefix} Received empty or invalid order book response.{Style.RESET_ALL}")
            return None
        # Check essential keys
        if not isinstance(order_book.get("bids"), list) or not isinstance(order_book.get("asks"), list):
            logger.error(
                f"{Fore.RED}{log_prefix} Order book 'bids' or 'asks' key is missing or not a list.{Style.RESET_ALL}"
            )
            return None
        # Check if both sides are empty (can happen in thin markets)
        if not order_book.get("bids") and not order_book.get("asks"):
            logger.warning(f"{Fore.YELLOW}{log_prefix} Order book has empty bids AND asks.{Style.RESET_ALL}")
        # Check for timestamp (important for data freshness)
        if not order_book.get("timestamp") or not order_book.get("datetime"):
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Order book is missing timestamp/datetime information.{Style.RESET_ALL}"
            )

        # Check for crossed spread (top bid >= top ask)
        top_bid_p, top_ask_p = None, None
        if order_book.get("bids"):
            top_bid_p = safe_decimal_conversion(order_book["bids"][0][0])
        if order_book.get("asks"):
            top_ask_p = safe_decimal_conversion(order_book["asks"][0][0])

        if top_bid_p is not None and top_ask_p is not None and top_bid_p >= top_ask_p:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Order book spread is crossed or zero: Top Bid={top_bid_p}, Top Ask={top_ask_p}.{Style.RESET_ALL}"
            )

        # Log success summary
        top_bid_log = order_book["bids"][0][0] if order_book.get("bids") else "N/A"
        top_ask_log = order_book["asks"][0][0] if order_book.get("asks") else "N/A"
        dt_log = order_book.get("datetime", "N/A")
        logger.info(
            f"{Fore.GREEN}{log_prefix} Fetched OK at {dt_log}. Top Bid: {top_bid_log}, Top Ask: {top_ask_log}{Style.RESET_ALL}"
        )
        return order_book

    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API error: {type(e).__name__}. Retry handled.{Style.RESET_ALL}")
        raise  # Re-raise for decorator
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Exchange error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False)
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=True
        )
        return None


@retry_api_call()
async def fetch_recent_trades(exchange: ccxt.bybit, symbol: str, limit: int, config: Config) -> list[dict]:
    """Fetches recent public market trades, applying category-specific limits."""
    func_name = "fetch_recent_trades"
    log_prefix = f"[{func_name} ({symbol}, limit={limit})]"
    logger.debug(f"{log_prefix} Fetching recent trades...")

    category = market_cache.get_category(symbol)
    if not category:
        logger.error(
            f"{Fore.RED}{log_prefix} Cannot determine category for {symbol}. Cannot fetch trades.{Style.RESET_ALL}"
        )
        return []

    # Bybit V5 Trade History Limits: Linear/Inverse: 1000, Spot: 60, Option: 100
    limit_map = {Category.SPOT: 60, Category.LINEAR: 1000, Category.INVERSE: 1000, Category.OPTION: 100}
    max_limit = limit_map.get(category)
    effective_limit = limit

    # Adjust limit if it exceeds the category maximum
    if max_limit is not None:
        if limit > max_limit:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Requested limit {limit} exceeds maximum {max_limit} for {category.value}. Clamping limit to {max_limit}.{Style.RESET_ALL}"
            )
            effective_limit = max_limit
    else:
        # Should not happen if category is valid, but handle defensively
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Unknown maximum trade limit for category {category.value}. Using requested limit {limit}.{Style.RESET_ALL}"
        )

    params = {"category": category.value, "limit": effective_limit}  # V5 requires category, pass limit too
    try:
        # Fetch trades using CCXT method
        trades = await exchange.fetch_trades(symbol, limit=effective_limit, params=params)

        if trades is None:  # CCXT might return None on error or empty
            logger.warning(f"{log_prefix} Received None from fetch_trades.")
            return []

        logger.info(
            f"{Fore.GREEN}{log_prefix} Fetched {len(trades)} recent trades (Limit requested: {effective_limit}).{Style.RESET_ALL}"
        )
        return trades  # Return the list of trades

    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API error: {type(e).__name__}. Retry handled.{Style.RESET_ALL}")
        raise  # Re-raise for decorator
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Exchange error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False)
        return []  # Return empty list on error
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=True
        )
        return []  # Return empty list on error


# --- Order Management Functions ---


@retry_api_call(
    max_retries=1, retry_on_exceptions=(NetworkError, RequestTimeout, RateLimitExceeded)
)  # Limit retries for placement
async def place_market_order_slippage_check(
    exchange: ccxt.bybit,
    symbol: str,
    side: Side,
    amount: Decimal,
    config: Config,
    max_slippage_pct: Decimal | None = None,  # Override default slippage check
    is_reduce_only: bool = False,
    time_in_force: TimeInForce = TimeInForce.IOC,  # IOC is good default for market
    client_order_id: str | None = None,  # Custom ID for tracking
    position_idx: PositionIdx | None = None,  # For Hedge Mode
) -> dict | None:
    """Places a market order with optional pre-execution spread check."""
    func_name = "place_market_order"
    action = "ReduceOnly" if is_reduce_only else "Open/Increase"
    qty_epsilon = config.get("POSITION_QTY_EPSILON", Decimal("1E-8"))
    log_prefix = f"[{func_name} ({symbol}, {side.value}, Amt:{amount}, {action})]"

    # --- Input Validation ---
    if amount <= qty_epsilon:
        logger.error(f"{Fore.RED}{log_prefix} Invalid order amount: {amount}. Must be positive.{Style.RESET_ALL}")
        return None
    # Market orders typically use IOC or FOK. GTC doesn't make sense.
    if time_in_force not in [TimeInForce.IOC, TimeInForce.FOK]:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} TimeInForce '{time_in_force.value}' used for Market order. IOC or FOK is recommended. Using specified TIF.{Style.RESET_ALL}"
        )

    category = market_cache.get_category(symbol)
    market = market_cache.get_market(symbol)
    if not category or not market:
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid category/market for {symbol}. Cannot place order.{Style.RESET_ALL}"
        )
        return None

    # Simple hedge mode logging based on position_idx
    if position_idx is not None and position_idx != PositionIdx.ONE_WAY:
        logger.debug(f"{log_prefix} positionIdx={position_idx.value} provided, assuming Hedge Mode context.")

    # Format amount according to market precision
    formatted_amount_str = format_amount(exchange, symbol, amount)
    if formatted_amount_str is None:
        logger.error(f"{Fore.RED}{log_prefix} Failed to format amount {amount} for precision.{Style.RESET_ALL}")
        return None
    try:
        # Convert formatted string amount back to float for CCXT call
        formatted_amount_float = float(formatted_amount_str)
    except ValueError:
        logger.error(
            f"{Fore.RED}{log_prefix} Formatted amount '{formatted_amount_str}' is not a valid float.{Style.RESET_ALL}"
        )
        return None

    # Determine max slippage percentage for check
    effective_max_slippage = max_slippage_pct if max_slippage_pct is not None else config.get("DEFAULT_SLIPPAGE_PCT")
    spread_check_enabled = effective_max_slippage is not None and effective_max_slippage >= Decimal(0)

    log_msg = f"{Fore.BLUE}{log_prefix} Placing order. Amount: {formatted_amount_str}, TIF: {time_in_force.value}"
    if spread_check_enabled:
        log_msg += f", Spread Check Max: {effective_max_slippage:.4%}"
    else:
        log_msg += ", Spread Check: Disabled"
    logger.info(log_msg + Style.RESET_ALL)

    # --- Spread Check (Optional) ---
    if spread_check_enabled:
        try:
            # Fetch shallow order book for current bid/ask
            ob_depth = config.get("SHALLOW_OB_FETCH_DEPTH", 5)  # Use small depth for speed
            ob_shallow = await fetch_l2_order_book_validated(exchange, symbol, ob_depth, config)  # Use validated fetch

            if ob_shallow and ob_shallow.get("bids") and ob_shallow.get("asks"):
                best_bid = safe_decimal_conversion(ob_shallow["bids"][0][0])
                best_ask = safe_decimal_conversion(ob_shallow["asks"][0][0])

                if best_bid and best_ask and best_bid > 0:
                    # Calculate spread percentage: (Ask - Bid) / Bid
                    spread_pct = (best_ask - best_bid) / best_bid
                    logger.debug(f"{log_prefix} Spread Check: Bid={best_bid}, Ask={best_ask}, Spread={spread_pct:.4%}")
                    # Compare with allowed slippage
                    if spread_pct > effective_max_slippage:
                        logger.error(
                            f"{Back.RED}{log_prefix} ABORTED: Current spread {spread_pct:.4%} exceeds maximum allowed {effective_max_slippage:.4%}.{Style.RESET_ALL}"
                        )
                        send_sms_alert(
                            f"[{symbol}] MKT Order ABORT ({side.value}): Spread {spread_pct:.4%} > Max {effective_max_slippage:.4%}",
                            config,
                        )
                        return None  # Abort placement
                else:
                    logger.warning(
                        f"{Fore.YELLOW}{log_prefix} Could not get valid best bid/ask from shallow OB. Skipping spread check.{Style.RESET_ALL}"
                    )
            else:
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} Could not fetch shallow order book. Skipping spread check.{Style.RESET_ALL}"
                )
        except Exception as ob_err:
            # Log error during check but proceed with placement (fail-open)
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Error during spread check: {ob_err}. Proceeding with order placement.{Style.RESET_ALL}",
                exc_info=False,
            )

    # --- Prepare and Place Order ---
    params: dict[str, Any] = {
        "category": category.value,
        "reduceOnly": is_reduce_only,
        "timeInForce": time_in_force.value,
        # V5 Specific Params (Map from CCXT standard args where possible)
        # 'marketUnit': 'baseCoin' or 'quoteCoin' - specify if amount is base or quote qty (default usually base)
    }
    # Add Client Order ID if provided (sanitize for Bybit rules)
    if client_order_id:
        # Bybit V5 orderLinkId requirements: letters, numbers, -, _ ; max length 36
        clean_cid = "".join(filter(lambda c: c.isalnum() or c in ["-", "_"], client_order_id))[:36]
        if len(clean_cid) != len(client_order_id):
            logger.warning(f"{log_prefix} Client Order ID sanitized from '{client_order_id}' to '{clean_cid}'")
        params["orderLinkId"] = clean_cid
    # Add Position Index if provided
    if position_idx is not None:
        params["positionIdx"] = position_idx.value

    try:
        logger.info(f"{log_prefix} Sending create_market_order request with params: {params}...")
        # Use CCXT's standard method for creating market orders
        order = await exchange.create_market_order(
            symbol=symbol,
            side=side.value,
            amount=formatted_amount_float,  # Pass the float amount
            params=params,  # Pass V5 specific parameters here
        )

        # --- Process Response ---
        if not order or not isinstance(order, dict):
            logger.error(
                f"{Fore.RED}{log_prefix} FAILED - Received invalid order response from create_market_order.{Style.RESET_ALL}"
            )
            return None

        order_id = order.get("id")
        status = order.get("status", "unknown")  # e.g., 'closed' (filled), 'canceled' (if IOC failed)
        filled_amount = safe_decimal_conversion(order.get("filled", "0"))
        avg_price = safe_decimal_conversion(order.get("average"))  # Avg fill price

        # Log based on status
        log_color = Fore.GREEN if status in ["closed", "filled"] else Fore.YELLOW if status == "open" else Fore.RED
        logger.info(
            f"{log_color}{log_prefix} Order Result - ID: ...{format_order_id(order_id)}, Status: {status}, Filled Qty: {format_amount(exchange, symbol, filled_amount)}, Avg Price: {format_price(exchange, symbol, avg_price)}{Style.RESET_ALL}"
        )

        # Check for partial fills with IOC/FOK
        if time_in_force in [TimeInForce.IOC, TimeInForce.FOK]:
            # Use epsilon comparison for filled amount vs requested amount
            requested_amount_dec = safe_decimal_conversion(formatted_amount_str)  # Convert formatted str back
            if (
                filled_amount is not None
                and requested_amount_dec is not None
                and filled_amount < requested_amount_dec * (Decimal(1) - qty_epsilon)
            ):
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} Order {order_id} ({time_in_force.value}) was partially filled ({filled_amount} / {requested_amount_dec}). Status: {status}.{Style.RESET_ALL}"
                )
            elif filled_amount is None and requested_amount_dec is not None and requested_amount_dec > qty_epsilon:
                # This case might indicate an issue or immediate cancellation
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} Order {order_id} ({time_in_force.value}) reported NO fill amount, but requested > 0. Status: {status}.{Style.RESET_ALL}"
                )

        # Return the parsed order dictionary from CCXT
        return order

    # --- Error Handling for Order Placement ---
    except InsufficientFunds as e:
        logger.error(f"{Back.RED}{log_prefix} FAILED - Insufficient Funds: {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{symbol}] Order Fail ({side.value} MKT): Insufficient Funds", config)
        return None
    except InvalidOrder as e:  # Covers various rejection reasons (size, price, etc.)
        logger.error(f"{Back.RED}{log_prefix} FAILED - Invalid Order / Rejected by Exchange: {e}{Style.RESET_ALL}")
        return None
    except ExchangeError as e:  # Catch other specific exchange errors
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - Exchange Error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False
        )
        return None
    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
        # Let retry decorator handle these by re-raising
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - API Communication Error: {type(e).__name__}: {e}{Style.RESET_ALL}"
        )
        raise e
    except Exception as e:  # Catch any other unexpected errors
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - Unexpected Error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=True
        )
        return None


@retry_api_call(max_retries=1, retry_on_exceptions=(NetworkError, RequestTimeout, RateLimitExceeded))
async def place_limit_order_tif(
    exchange: ccxt.bybit,
    symbol: str,
    side: Side,
    amount: Decimal,
    price: Decimal,
    config: Config,
    time_in_force: TimeInForce = TimeInForce.GTC,  # GTC is common default for limit
    is_reduce_only: bool = False,
    is_post_only: bool = False,  # If true, ensures order is maker-only
    client_order_id: str | None = None,
    position_idx: PositionIdx | None = None,
) -> dict | None:
    """Places a limit order with specified Time-In-Force and optional post-only."""
    func_name = "place_limit_order"
    action = "ReduceOnly" if is_reduce_only else "Open/Increase"
    qty_epsilon = config.get("POSITION_QTY_EPSILON", Decimal("1E-8"))

    # Determine effective TIF based on post_only flag
    effective_tif = TimeInForce.POST_ONLY if is_post_only else time_in_force
    tif_str = effective_tif.value
    post_only_str = " (PostOnly)" if is_post_only else ""
    log_prefix = (
        f"[{func_name} ({symbol}, {side.value}, Amt:{amount} @ Px:{price}, {action}, TIF:{tif_str}{post_only_str})]"
    )

    # --- Input Validation ---
    if amount <= qty_epsilon or price <= Decimal("0"):
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid order amount ({amount}) or price ({price}). Both must be positive.{Style.RESET_ALL}"
        )
        return None
    # Warn if PostOnly is combined with incompatible TIF (though Bybit might handle it)
    if is_post_only and time_in_force not in [TimeInForce.GTC, TimeInForce.POST_ONLY]:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Using PostOnly flag with TIF '{time_in_force.value}'. Effective TIF will be PostOnly.{Style.RESET_ALL}"
        )

    category = market_cache.get_category(symbol)
    market = market_cache.get_market(symbol)
    if not category or not market:
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid category/market for {symbol}. Cannot place order.{Style.RESET_ALL}"
        )
        return None

    # Format amount and price according to market precision
    formatted_amount_str = format_amount(exchange, symbol, amount)
    formatted_price_str = format_price(exchange, symbol, price)
    if formatted_amount_str is None or formatted_price_str is None:
        logger.error(
            f"{Fore.RED}{log_prefix} Failed to format amount ({amount}) or price ({price}) for precision.{Style.RESET_ALL}"
        )
        return None
    try:
        # Convert formatted strings back to floats for CCXT call
        formatted_amount_float = float(formatted_amount_str)
        formatted_price_float = float(formatted_price_str)
    except ValueError:
        logger.error(
            f"{Fore.RED}{log_prefix} Formatted amount '{formatted_amount_str}' or price '{formatted_price_str}' is not a valid float.{Style.RESET_ALL}"
        )
        return None

    logger.info(f"{Fore.BLUE}{log_prefix} Placing order...{Style.RESET_ALL}")

    # --- Prepare Parameters ---
    params: dict[str, Any] = {
        "category": category.value,
        "reduceOnly": is_reduce_only,
        "timeInForce": effective_tif.value,
        # Note: CCXT's create_limit_order handles postOnly flag internally for Bybit
        # It maps the postOnly=True argument to the correct Bybit parameter if needed.
    }
    # Add Client Order ID if provided
    if client_order_id:
        clean_cid = "".join(filter(lambda c: c.isalnum() or c in ["-", "_"], client_order_id))[:36]
        if len(clean_cid) != len(client_order_id):
            logger.warning(f"{log_prefix} Client Order ID sanitized: '{clean_cid}'")
        params["orderLinkId"] = clean_cid
    # Add Position Index if provided
    if position_idx is not None:
        params["positionIdx"] = position_idx.value

    try:
        logger.info(f"{log_prefix} Sending create_limit_order request with params: {params}...")
        # Use CCXT's standard method for creating limit orders
        order = await exchange.create_limit_order(
            symbol=symbol,
            side=side.value,
            amount=formatted_amount_float,
            price=formatted_price_float,
            params=params,
            # Pass postOnly flag directly to CCXT method if needed (check CCXT Bybit implementation)
            # Alternatively, rely on TIF="PostOnly" in params for V5
        )

        # --- Process Response ---
        if not order or not isinstance(order, dict):
            logger.error(
                f"{Fore.RED}{log_prefix} FAILED - Received invalid order response from create_limit_order.{Style.RESET_ALL}"
            )
            return None

        order_id = order.get("id")
        status = order.get("status", "unknown")  # e.g., 'open', 'closed', 'canceled'
        order_price = safe_decimal_conversion(order.get("price"))
        order_amount = safe_decimal_conversion(order.get("amount"))

        # Log based on status (Green for open/accepted, Yellow for others like triggered/new)
        log_color = Fore.GREEN if status == "open" else Fore.YELLOW if status in ["triggered", "new"] else Fore.RED
        logger.info(
            f"{log_color}{log_prefix} Order Result - ID: ...{format_order_id(order_id)}, Status: {status}, Price: {format_price(exchange, symbol, order_price)}, Amount: {format_amount(exchange, symbol, order_amount)}{Style.RESET_ALL}"
        )
        return order

    # --- Error Handling ---
    except OrderImmediatelyFillable as e:
        # This specific error occurs when a PostOnly order would execute immediately as taker
        if is_post_only or effective_tif == TimeInForce.POST_ONLY:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} PostOnly order FAILED as expected (would execute immediately): {e}{Style.RESET_ALL}"
            )
            return None  # Return None as the order was rejected as intended by PostOnly
        else:
            # Should not happen for non-PostOnly orders, indicates unexpected state
            logger.error(
                f"{Back.RED}{log_prefix} FAILED - Unexpected OrderImmediatelyFillable for non-PostOnly order: {e}{Style.RESET_ALL}"
            )
            return None
    except InsufficientFunds as e:
        logger.error(f"{Back.RED}{log_prefix} FAILED - Insufficient Funds: {e}{Style.RESET_ALL}")
        return None
    except InvalidOrder as e:
        logger.error(f"{Back.RED}{log_prefix} FAILED - Invalid Order / Rejected by Exchange: {e}{Style.RESET_ALL}")
        return None
    except ExchangeError as e:
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - Exchange Error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False
        )
        return None
    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - API Communication Error: {type(e).__name__}: {e}{Style.RESET_ALL}"
        )
        raise e  # Re-raise for retry decorator
    except Exception as e:
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - Unexpected Error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=True
        )
        return None


@retry_api_call(max_retries=1, retry_on_exceptions=(NetworkError, RequestTimeout, RateLimitExceeded))
async def place_batch_orders(
    exchange: ccxt.bybit,
    orders: list[dict[str, Any]],  # List of order request dictionaries
    config: Config,
    category_override: Category | None = None,  # Force a category for the batch
) -> tuple[list[dict | None], list[dict | None]]:
    """Places multiple orders in a single V5 batch request. Handles validation and response parsing.

    Args:
        exchange: Initialized ccxt.bybit instance.
        orders: A list of dictionaries, each representing an order request.
                Required keys per order: 'symbol', 'side' (Side enum or 'buy'/'sell'),
                                         'type' ('Limit' or 'Market'), 'amount'.
                Optional keys: 'price' (for Limit), 'clientOrderId', 'reduceOnly',
                               'timeInForce', 'positionIdx', trigger params, etc.
        config: Configuration object.
        category_override: If specified, forces all orders in the batch to belong
                           to this category. If None, category is determined from
                           the first valid order and all others must match.

    Returns:
        A tuple containing two lists of the same length as the input `orders`:
        1. success_orders: List where each element is a parsed CCXT order dict
                           on success, or None on failure/validation error.
        2. error_details: List where each element is None on success, or a
                          dictionary {'code': ..., 'msg': ...} on failure.
    """
    func_name = "place_batch_orders"
    num_orders = len(orders)
    qty_epsilon = config.get("POSITION_QTY_EPSILON", Decimal("1E-8"))
    log_prefix = f"[{func_name} ({num_orders} orders)]"
    logger.info(f"{Fore.BLUE}{log_prefix} Preparing batch order request...{Style.RESET_ALL}")

    # Initialize result lists with Nones
    final_success_orders: list[dict | None] = [None] * num_orders
    final_error_details: list[dict | None] = [None] * num_orders

    if not orders:
        logger.warning(f"{Fore.YELLOW}{log_prefix} No orders provided in the batch list.{Style.RESET_ALL}")
        return final_success_orders, final_error_details  # Return empty results

    # --- Prepare and Validate Individual Orders for V5 format ---
    batch_requests_v5: list[dict | None] = [None] * num_orders  # Stores V5 formatted requests
    category_to_use: str | None = category_override.value if category_override else None
    determined_category_enum: Category | None = category_override
    batch_limit = 10  # Default conservative limit, will be updated based on category

    # --- Phase 1: Pre-validation and Formatting ---
    abort_batch = False
    for i, order_req in enumerate(orders):
        error_detail: dict | None = None
        symbol = order_req.get("symbol")
        side_raw = order_req.get("side")
        order_type_raw = order_req.get("type")  # Expect 'Limit' or 'Market'
        amount_raw = order_req.get("amount")

        # Basic Input Validation
        if not all([symbol, side_raw, order_type_raw, amount_raw]):
            error_detail = {"code": -101, "msg": "Missing required fields (symbol, side, type, amount)."}
        elif not isinstance(symbol, str):
            error_detail = {"code": -101, "msg": "Symbol must be a string."}
        elif not isinstance(side_raw, (Side, str)):
            error_detail = {"code": -101, "msg": "Side must be Side enum or 'buy'/'sell' string."}
        elif not isinstance(order_type_raw, str) or order_type_raw.lower() not in ["limit", "market"]:
            error_detail = {"code": -101, "msg": "Type must be 'Limit' or 'Market'."}
        elif safe_decimal_conversion(amount_raw, Decimal("-1")) <= qty_epsilon:
            error_detail = {"code": -101, "msg": f"Amount '{amount_raw}' must be positive."}

        if error_detail:
            logger.error(f"{Fore.RED}{log_prefix} Order #{i + 1} Input Err: {error_detail['msg']}{Style.RESET_ALL}")
            final_error_details[i] = error_detail
            continue  # Skip to next order

        # Determine and Validate Category & Batch Limit
        current_category_enum = market_cache.get_category(symbol)
        if not current_category_enum:
            error_detail = {"code": -102, "msg": f"Cannot determine market category for symbol '{symbol}'."}
            logger.error(f"{Fore.RED}{log_prefix} Order #{i + 1} Category Err: {error_detail['msg']}{Style.RESET_ALL}")
            final_error_details[i] = error_detail
            continue

        current_category_str = current_category_enum.value
        # V5 Batch Limits: Linear/Inverse/Spot: 10, Option: 5
        cat_limit_map = {Category.LINEAR: 10, Category.INVERSE: 10, Category.SPOT: 10, Category.OPTION: 5}

        if category_override:
            # Check if order matches the forced override category
            if current_category_str != category_override.value:
                error_detail = {
                    "code": -103,
                    "msg": f"Symbol '{symbol}' category '{current_category_str}' does not match batch override category '{category_override.value}'.",
                }
                logger.error(
                    f"{Fore.RED}{log_prefix} Order #{i + 1} Mismatch Err: {error_detail['msg']}{Style.RESET_ALL}"
                )
                final_error_details[i] = error_detail
                continue
            effective_category = category_override.value
            if i == 0:  # Set limit based on override on first item
                batch_limit = cat_limit_map.get(category_override, 10)  # Default 10 if somehow override is invalid enum
                logger.info(f"{log_prefix} Using overridden batch category: {effective_category}, Limit: {batch_limit}")
                if num_orders > batch_limit:
                    error_msg = (
                        f"Batch size {num_orders} exceeds limit {batch_limit} for category '{effective_category}'."
                    )
                    logger.error(f"{Back.RED}{log_prefix} {error_msg} Aborting entire batch.{Style.RESET_ALL}")
                    # Mark all subsequent orders as failed due to batch limit
                    for j in range(i, num_orders):
                        if final_error_details[j] is None:
                            final_error_details[j] = {"code": -100, "msg": error_msg}
                    abort_batch = True
                    break  # Stop processing further orders
        else:
            # Dynamically determine category from first valid order
            if category_to_use is None:  # First valid order determines batch category
                category_to_use = current_category_str
                determined_category_enum = current_category_enum
                batch_limit = cat_limit_map.get(determined_category_enum, 10)
                logger.info(
                    f"{log_prefix} Determined batch category from first order: {category_to_use}, Limit: {batch_limit}"
                )
                if num_orders > batch_limit:
                    error_msg = f"Batch size {num_orders} exceeds limit {batch_limit} for determined category '{category_to_use}'."
                    logger.error(f"{Back.RED}{log_prefix} {error_msg} Aborting entire batch.{Style.RESET_ALL}")
                    final_error_details[i] = {"code": -100, "msg": error_msg}  # Mark current order
                    for j in range(i + 1, num_orders):  # Mark subsequent orders
                        if final_error_details[j] is None:
                            final_error_details[j] = {"code": -100, "msg": error_msg}
                    abort_batch = True
                    break  # Stop processing
            # Check consistency for subsequent orders
            elif current_category_str != category_to_use:
                error_detail = {
                    "code": -104,
                    "msg": f"Symbol '{symbol}' category '{current_category_str}' does not match batch category '{category_to_use}'. Mixing categories not allowed.",
                }
                logger.error(f"{Fore.RED}{log_prefix} Order #{i + 1} Mix Err: {error_detail['msg']}{Style.RESET_ALL}")
                final_error_details[i] = error_detail
                continue
            effective_category = category_to_use  # Category is consistent

        # Format Amount/Price based on market precision
        market = market_cache.get_market(symbol)  # Re-fetch just in case cache updated? Unlikely needed.
        if not market:  # Should not happen if category was found, but check defensively
            error_detail = {
                "code": -105,
                "msg": f"Market data unexpectedly missing for '{symbol}' after category check.",
            }
            logger.error(f"{Fore.RED}{log_prefix} Order #{i + 1} Data Err: {error_detail['msg']}{Style.RESET_ALL}")
            final_error_details[i] = error_detail
            continue
        amount_str = format_amount(exchange, symbol, amount_raw)
        if amount_str is None:
            error_detail = {"code": -106, "msg": f"Invalid amount format or precision error for amount '{amount_raw}'."}
            logger.error(f"{Fore.RED}{log_prefix} Order #{i + 1} Format Err: {error_detail['msg']}{Style.RESET_ALL}")
            final_error_details[i] = error_detail
            continue

        price_str: str | None = None
        if order_type_raw.lower() == "limit":
            price_raw = order_req.get("price")
            if price_raw is None or safe_decimal_conversion(price_raw, Decimal("-1")) <= Decimal(0):
                error_detail = {"code": -107, "msg": "Limit order requires a valid positive 'price'."}
                logger.error(f"{Fore.RED}{log_prefix} Order #{i + 1} Price Err: {error_detail['msg']}{Style.RESET_ALL}")
                final_error_details[i] = error_detail
                continue
            price_str = format_price(exchange, symbol, price_raw)
            if price_str is None:
                error_detail = {
                    "code": -108,
                    "msg": f"Invalid price format or precision error for price '{price_raw}'.",
                }
                logger.error(
                    f"{Fore.RED}{log_prefix} Order #{i + 1} Format Err: {error_detail['msg']}{Style.RESET_ALL}"
                )
                final_error_details[i] = error_detail
                continue

        # Normalize Side & Type for V5 JSON payload
        side_val = side_raw.value if isinstance(side_raw, Side) else str(side_raw).lower()
        if side_val not in ["buy", "sell"]:  # Should have been caught earlier, but double check
            error_detail = {"code": -109, "msg": f"Invalid side value '{side_raw}'."}
            logger.error(f"{Fore.RED}{log_prefix} Order #{i + 1} Side Err: {error_detail['msg']}{Style.RESET_ALL}")
            final_error_details[i] = error_detail
            continue
        # V5 uses "Buy" / "Sell"
        bybit_v5_side = side_val.capitalize()
        # V5 uses "Limit" / "Market"
        bybit_v5_type = order_type_raw.capitalize()

        # Build the V5 request dictionary for this specific order
        v5_req: dict[str, Any] = {
            "symbol": symbol,
            "side": bybit_v5_side,
            "orderType": bybit_v5_type,
            "qty": amount_str,
            # --- Map common optional parameters ---
            # Client Order ID (already validated length/chars implicitly by filter below)
            "orderLinkId": order_req.get("clientOrderId") if order_req.get("clientOrderId") else None,
            # Reduce Only
            "reduceOnly": order_req.get("reduceOnly") if order_req.get("reduceOnly") is not None else None,
            # Time In Force
            "timeInForce": order_req.get("timeInForce").value
            if isinstance(order_req.get("timeInForce"), TimeInForce)
            else order_req.get("timeInForce"),
            # Position Index (convert enum to value)
            "positionIdx": order_req.get("positionIdx").value
            if isinstance(order_req.get("positionIdx"), PositionIdx)
            else order_req.get("positionIdx"),
            # Trigger parameters (pass directly if provided)
            "triggerPrice": format_price(exchange, symbol, order_req.get("triggerPrice"))
            if order_req.get("triggerPrice")
            else None,
            "triggerBy": order_req.get("triggerBy").value
            if isinstance(order_req.get("triggerBy"), TriggerBy)
            else order_req.get("triggerBy"),
            "triggerDirection": order_req.get("triggerDirection").value
            if isinstance(order_req.get("triggerDirection"), TriggerDirection)
            else order_req.get("triggerDirection"),
            # TP/SL parameters (pass directly) - Ensure correct V5 names ('stopLoss', 'takeProfit', 'tpTriggerBy', 'slTriggerBy', 'tpslMode', etc.)
            "takeProfit": format_price(exchange, symbol, order_req.get("takeProfit"))
            if order_req.get("takeProfit")
            else None,
            "stopLoss": format_price(exchange, symbol, order_req.get("stopLoss"))
            if order_req.get("stopLoss")
            else None,
            "tpTriggerBy": order_req.get("tpTriggerBy").value
            if isinstance(order_req.get("tpTriggerBy"), TriggerBy)
            else order_req.get("tpTriggerBy"),
            "slTriggerBy": order_req.get("slTriggerBy").value
            if isinstance(order_req.get("slTriggerBy"), TriggerBy)
            else order_req.get("slTriggerBy"),
            "tpslMode": order_req.get("tpslMode"),  # e.g., 'Full' or 'Partial'
            "tpOrderType": order_req.get("tpOrderType", OrderType.MARKET.value),  # Default TP type if not specified
            "slOrderType": order_req.get("slOrderType", OrderType.MARKET.value),  # Default SL type
            "tpLimitPrice": format_price(exchange, symbol, order_req.get("tpLimitPrice"))
            if order_req.get("tpLimitPrice")
            else None,
            "slLimitPrice": format_price(exchange, symbol, order_req.get("slLimitPrice"))
            if order_req.get("slLimitPrice")
            else None,
        }
        # Add price for Limit orders
        if price_str:
            v5_req["price"] = price_str

        # --- Sanitize and Clean V5 Request ---
        # Remove None values as Bybit API might not like nulls for optional fields
        v5_req_cleaned = {k: v for k, v in v5_req.items() if v is not None}

        # Sanitize clientOrderId again just before adding
        if "orderLinkId" in v5_req_cleaned:
            cid = str(v5_req_cleaned["orderLinkId"])
            clean_cid = "".join(filter(lambda c: c.isalnum() or c in ["-", "_"], cid))[:36]
            if len(clean_cid) != len(cid):
                logger.warning(f"{log_prefix} Order #{i + 1} orderLinkId sanitized: '{clean_cid}'")
            v5_req_cleaned["orderLinkId"] = clean_cid
            # If CID becomes empty after sanitizing, remove it
            if not clean_cid:
                del v5_req_cleaned["orderLinkId"]

        # Ensure boolean flags are actual booleans if present (might be strings from config)
        if "reduceOnly" in v5_req_cleaned:
            val = v5_req_cleaned["reduceOnly"]
            v5_req_cleaned["reduceOnly"] = (
                str(val).lower() in ["true", "1", "yes"] if isinstance(val, str) else bool(val)
            )

        # Store the validated and formatted request
        batch_requests_v5[i] = v5_req_cleaned
        logger.debug(f"{log_prefix} Prepared Order #{i + 1}: {v5_req_cleaned}")

    # Check if batch processing was aborted due to limit exceeded
    if abort_batch:
        return final_success_orders, final_error_details

    # --- Phase 2: Filter out invalid requests & Prepare for Sending ---
    # Create mapping between original index and the index in the list being sent
    original_index_to_sent_index: dict[int, int] = {}
    sent_index_to_original_index: dict[int, int] = {}
    valid_v5_reqs_to_send: list[dict] = []
    sent_idx = 0
    for original_idx, req in enumerate(batch_requests_v5):
        # Only include requests that were successfully prepared AND had no pre-validation errors
        if req is not None and final_error_details[original_idx] is None:
            valid_v5_reqs_to_send.append(req)
            original_index_to_sent_index[original_idx] = sent_idx
            sent_index_to_original_index[sent_idx] = original_idx
            sent_idx += 1

    # If no valid orders remain after filtering, return the errors found so far
    if not valid_v5_reqs_to_send:
        logger.error(f"{Fore.RED}{log_prefix} No valid orders remaining to send after pre-validation.{Style.RESET_ALL}")
        return final_success_orders, final_error_details

    # Final check on the category determined for the batch
    final_batch_category = category_to_use
    if not final_batch_category:  # Should not happen if valid_reqs exists and category logic is sound
        logger.critical(
            f"{Back.RED}{log_prefix} Internal Error: Final batch category could not be determined despite valid orders. Aborting.{Style.RESET_ALL}"
        )
        internal_error = {"code": -199, "msg": "Internal error: Final batch category missing"}
        # Mark all potentially valid orders as failed due to this internal error
        for i in range(num_orders):
            if final_error_details[i] is None and batch_requests_v5[i] is not None:
                final_error_details[i] = internal_error
        return final_success_orders, final_error_details

    # --- Phase 3: Execute Batch Request ---
    logger.info(
        f"{log_prefix} Sending batch create request for {len(valid_v5_reqs_to_send)} orders (Category: {final_batch_category})..."
    )
    # Prepare payload for the V5 batch endpoint
    params = {
        "category": final_batch_category,
        "request": valid_v5_reqs_to_send,  # The list of V5 order request dicts
    }

    # ******** This is the block where SyntaxError was likely missing 'except' *********
    try:
        # Make the API call using the implicit method
        response = await exchange.private_post_v5_order_create_batch(params)
        logger.debug(f"{log_prefix} Raw API response: {response}")

        # --- Phase 4: Process Batch Response ---
        if not response or not isinstance(response, dict):
            raise ExchangeError(f"{log_prefix} Invalid response received from batch order API: {response}")

        ret_code = response.get("retCode")
        ret_msg = response.get("retMsg", "N/A")
        result_data = response.get("result", {})
        # Bybit V5 response structure: result.list for successes, result.errInfo for errors
        success_raw = result_data.get("list", []) if isinstance(result_data, dict) else []
        errors_raw = result_data.get("errInfo", []) if isinstance(result_data, dict) else []

        if not isinstance(success_raw, list):
            logger.warning(f"{log_prefix} API response 'result.list' is not a list.")
            success_raw = []
        if not isinstance(errors_raw, list):
            logger.warning(f"{log_prefix} API response 'result.errInfo' is not a list.")
            errors_raw = []

        # Check overall request status code
        if ret_code == 0:
            logger.info(
                f"{Fore.GREEN}{log_prefix} Batch request processed by API. Success Reports: {len(success_raw)}, Failure Reports: {len(errors_raw)}{Style.RESET_ALL}"
            )

            # Keep track of original indices processed to avoid double-marking
            processed_original_indices = set()

            # Process errors reported by the API first
            for err_info in errors_raw:
                if not isinstance(err_info, dict):
                    continue  # Skip invalid error entries
                err_code = err_info.get("code", -1)
                err_msg = err_info.get("msg", "Unknown API error")
                # V5 'errInfo' usually contains the index in the *sent* request list
                err_req_idx = err_info.get("index")

                if (
                    err_req_idx is not None
                    and isinstance(err_req_idx, int)
                    and 0 <= err_req_idx < len(valid_v5_reqs_to_send)
                ):
                    # Map back to the original index in the input list
                    original_list_idx = sent_index_to_original_index.get(err_req_idx)
                    if original_list_idx is not None:
                        # Get details from the request that failed
                        failed_req = valid_v5_reqs_to_send[err_req_idx]
                        err_cid = failed_req.get("orderLinkId", "N/A")
                        req_symbol = failed_req.get("symbol", "N/A")
                        logger.error(
                            f"{Fore.RED}{log_prefix} Order #{original_list_idx + 1} ({req_symbol}, CID:{err_cid}) FAILED (API Reported). Code: {err_code}, Msg: {err_msg}{Style.RESET_ALL}"
                        )
                        # Store the error detail in the final results
                        if final_error_details[original_list_idx] is None:  # Avoid overwriting pre-validation errors
                            final_error_details[original_list_idx] = {"code": err_code, "msg": err_msg}
                        processed_original_indices.add(original_list_idx)
                    else:
                        # This indicates an internal mapping error
                        logger.error(
                            f"{log_prefix} Internal mapping error: Could not find original index for error request index {err_req_idx}."
                        )
                else:
                    # Error entry didn't have a valid index
                    logger.error(
                        f"{log_prefix} API error entry missing valid index: {err_info}. Cannot map error reliably."
                    )

            # Process successes reported by the API
            for order_data in success_raw:
                if not isinstance(order_data, dict):
                    continue  # Skip invalid success entries
                cid = order_data.get("orderLinkId")
                oid = order_data.get("orderId")
                symbol = order_data.get("symbol")  # Get symbol for market context
                original_list_idx = None
                found_match = False

                # Try to find the original request using the clientOrderId (most reliable)
                if cid:
                    for req_idx, sent_req in enumerate(valid_v5_reqs_to_send):
                        current_original_idx = sent_index_to_original_index.get(req_idx)
                        if current_original_idx is None:
                            continue  # Skip if mapping failed

                        # Check if CID matches and this index hasn't been processed as an error
                        if (
                            cid == sent_req.get("orderLinkId")
                            and current_original_idx not in processed_original_indices
                        ):
                            original_list_idx = current_original_idx
                            found_match = True
                            break  # Found match

                # If no CID match (or no CID provided), we might assume order preservation, but it's risky.
                # For now, primarily rely on CID. If no CID, mapping success is uncertain.
                if not found_match:
                    logger.warning(
                        f"{log_prefix} Success reported for OrderID {oid} (CID: {cid or 'None'}) but could not reliably map it back to an original request that wasn't already marked as failed."
                    )
                    continue  # Cannot process this success report reliably

                # If found and not already errored
                if original_list_idx is not None and final_error_details[original_list_idx] is None:
                    try:
                        # Parse the successful order data into CCXT format
                        market_context = market_cache.get_market(symbol) if symbol else None
                        if not market_context:
                            logger.warning(
                                f"{log_prefix} Market context missing for symbol {symbol} while parsing success order {oid}."
                            )
                            continue

                        parsed_order = exchange.parse_order(order_data, market_context)
                        final_success_orders[original_list_idx] = parsed_order
                        status = parsed_order.get("status", "?")
                        logger.info(
                            f"{Fore.GREEN}{log_prefix} Order #{original_list_idx + 1} ({symbol}, CID:{cid}) PLACED/Success. ID: ...{format_order_id(oid)}, Status: {status}{Style.RESET_ALL}"
                        )
                        processed_original_indices.add(original_list_idx)
                    except Exception as parse_err:
                        logger.error(
                            f"{log_prefix} Error parsing successful order data #{original_list_idx + 1} (ID:{oid}, CID:{cid}): {parse_err}"
                        )
                        # Mark as error because we couldn't process the success response
                        final_error_details[original_list_idx] = {
                            "code": -201,
                            "msg": f"Success reported by API but failed to parse response: {parse_err}",
                        }
                        processed_original_indices.add(original_list_idx)
                # else: Original index already marked as error or mapping failed.

            # Final check: Mark any sent orders that didn't get a success or error response as unknown
            for sent_req_idx, sent_req in enumerate(valid_v5_reqs_to_send):
                original_idx = sent_index_to_original_index.get(sent_req_idx)
                # If we have an original index, and it wasn't processed, and had no pre-validation error
                if (
                    original_idx is not None
                    and original_idx not in processed_original_indices
                    and final_error_details[original_idx] is None
                ):
                    req_symbol = sent_req.get("symbol", "N/A")
                    req_cid = sent_req.get("orderLinkId", "N/A")
                    logger.error(
                        f"{Fore.RED}{log_prefix} Order #{original_idx + 1} ({req_symbol}, CID:{req_cid}) - No response status received from API (Unknown outcome).{Style.RESET_ALL}"
                    )
                    final_error_details[original_idx] = {
                        "code": -202,
                        "msg": "No success or specific error message received from API for this order in the batch.",
                    }

        else:  # Batch request itself failed (retCode != 0)
            error_msg = f"Batch API request failed entirely. Code: {ret_code}, Msg: {ret_msg}"
            logger.error(f"{Back.RED}{log_prefix} {error_msg}{Style.RESET_ALL}")
            # Mark all orders that were *sent* as failed with this general error, unless they had a pre-validation error
            batch_api_error = {"code": ret_code, "msg": ret_msg}
            for sent_req_idx in range(len(valid_v5_reqs_to_send)):
                original_list_idx = sent_index_to_original_index.get(sent_req_idx)
                if original_list_idx is not None and final_error_details[original_list_idx] is None:
                    final_error_details[original_list_idx] = batch_api_error

        # Return the final success and error lists
        return final_success_orders, final_error_details

    # --- Exception Handling for the Batch API Call ---
    except InvalidOrder as e:  # Catch issues like parameter errors detected by CCXT or exchange for the *whole* batch
        logger.error(f"{Back.RED}{log_prefix} FAILED - Invalid Batch Order Parameters/Rejected: {e}{Style.RESET_ALL}")
        err_resp = {"code": -301, "msg": f"Invalid Batch Order Call: {e}"}
        # Mark all sent orders as failed
        for sent_req_idx in range(len(valid_v5_reqs_to_send)):
            original_list_idx = sent_index_to_original_index.get(sent_req_idx)
            if original_list_idx is not None and final_error_details[original_list_idx] is None:
                final_error_details[original_list_idx] = err_resp
        return final_success_orders, final_error_details
    except ExchangeError as e:  # Catch other specific exchange rejections for the batch call
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - Batch Exchange Error: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=False,
        )
        err_resp = {"code": -302, "msg": f"Batch Exchange Error: {e}"}
        for sent_req_idx in range(len(valid_v5_reqs_to_send)):
            original_list_idx = sent_index_to_original_index.get(sent_req_idx)
            if original_list_idx is not None and final_error_details[original_list_idx] is None:
                final_error_details[original_list_idx] = err_resp
        return final_success_orders, final_error_details
    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - Batch API Communication Error: {type(e).__name__}: {e}{Style.RESET_ALL}"
        )
        # Re-raise for the retry decorator to handle
        raise e
    except Exception as e:  # Catch any other unexpected errors during the batch call
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - Unexpected Batch Error: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        err_resp = {"code": -300, "msg": f"Unexpected Batch Error: {e}"}
        for sent_req_idx in range(len(valid_v5_reqs_to_send)):
            original_list_idx = sent_index_to_original_index.get(sent_req_idx)
            if original_list_idx is not None and final_error_details[original_list_idx] is None:
                final_error_details[original_list_idx] = err_resp
        return final_success_orders, final_error_details
    # ******** End of fixed try...except block *********


@retry_api_call()
async def fetch_position(exchange: ccxt.bybit, symbol: str, config: Config) -> dict | None:
    """Fetches position information for a specific symbol using V5 category context."""
    func_name = "fetch_position"
    log_prefix = f"[{func_name} ({symbol})]"
    logger.debug(f"{log_prefix} Fetching position...")

    category = market_cache.get_category(symbol)
    # Positions are typically relevant for derivatives (Linear/Inverse)
    # Spot might return something related to holdings, but standard position concept applies less. Options have positions too.
    # Let's allow fetching for all categories but log appropriately.
    if not category:
        logger.error(
            f"{Fore.RED}{log_prefix} Cannot determine category for {symbol}. Cannot fetch position.{Style.RESET_ALL}"
        )
        return None
    if category == Category.SPOT:
        logger.info(
            f"{log_prefix} Fetching 'position' for SPOT symbol {symbol}. This usually reflects holdings, not leveraged positions."
        )
        # Consider using fetch_balance for SPOT instead if holdings are needed.
        # For now, proceed with fetch_positions, but be aware of the context.

    params = {
        "category": category.value,
        "symbol": symbol,
    }  # V5 requires category, symbol is optional for filtering response
    try:
        # CCXT's fetch_positions can fetch multiple, but we filter by symbol using params
        # Bybit V5 endpoint: GET /v5/position/list
        # CCXT should map fetch_positions correctly with category + symbol param
        positions = await exchange.fetch_positions(symbols=[symbol], params=params)  # Request specific symbol
        logger.debug(f"{log_prefix} Raw positions response: {positions}")

        if positions and isinstance(positions, list):
            # fetch_positions returns a list, even if only one symbol is requested and found
            if len(positions) > 1:
                logger.warning(
                    f"{log_prefix} Received {len(positions)} positions when expecting 1 for {symbol}. Using the first entry."
                )
            position_data = positions[0]  # Assume the first one is the one we asked for

            # Validate essential keys from CCXT unified position structure
            # Core keys: symbol, side, contracts, entryPrice, markPrice, unrealizedPnl, leverage, marginType
            # Optional but useful: liquidationPrice, collateral, initialMargin, maintMargin
            required = [
                "symbol",
                "side",
                "contracts",
                "entryPrice",
                "markPrice",
                "unrealizedPnl",
                "leverage",
                "marginType",
            ]
            missing = [k for k in required if k not in position_data or position_data[k] is None]
            if missing:
                # Log missing keys but don't necessarily fail if basic info is present
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} Fetched position data for {symbol} missing expected keys: {missing}. Raw data: {str(position_data)[:200]}...{Style.RESET_ALL}"
                )

            pos_side = position_data.get("side")  # 'long', 'short', or None/empty string if flat
            contracts = safe_decimal_conversion(
                position_data.get("contracts"), Decimal(0)
            )  # Use safe conversion, default 0
            entry_price = safe_decimal_conversion(position_data.get("entryPrice"))
            mark_price = safe_decimal_conversion(position_data.get("markPrice"))
            unrealized_pnl = safe_decimal_conversion(position_data.get("unrealizedPnl"))
            leverage = safe_decimal_conversion(position_data.get("leverage"))
            margin_type = position_data.get("marginType")  # 'isolated' or 'cross'

            qty_epsilon = config.get("POSITION_QTY_EPSILON", Decimal("1E-8"))
            # Check if position size is significant
            if contracts is not None and abs(contracts) > qty_epsilon:
                log_color = Fore.GREEN if pos_side == "long" else Fore.RED if pos_side == "short" else Fore.YELLOW
                liq_price = safe_decimal_conversion(position_data.get("liquidationPrice"))
                logger.info(
                    f"{log_color}{log_prefix} Active Position: Side={pos_side}, Size={contracts}, Entry={entry_price}, Mark={mark_price}, uPNL={unrealized_pnl:.4f}, Lev={leverage}x, LiqPx={liq_price}, Mode={margin_type}{Style.RESET_ALL}"
                )
            else:
                # Log flat position
                logger.info(
                    f"{Fore.BLUE}{log_prefix} No active position found (position size is zero or negligible).{Style.RESET_ALL}"
                )
                # Ensure 'side' is None or consistent for flat positions if needed by calling code
                if position_data.get("side") is not None and abs(contracts) <= qty_epsilon:
                    position_data["side"] = None  # Standardize flat position side

            return position_data  # Return the CCXT unified position structure

        else:
            # This might happen if the API returns empty list for a valid symbol with no position, or if error occurred
            logger.info(
                f"{Fore.BLUE}{log_prefix} No position data returned by API for {symbol} (likely flat or fetch issue).{Style.RESET_ALL}"
            )
            # Return None if no position data is found
            return None

    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} API error: {type(e).__name__}. Retry handled.{Style.RESET_ALL}")
        raise  # Re-raise for decorator
    except AuthenticationError as e:
        logger.error(f"{Fore.RED}{log_prefix} Authentication error: {e}{Style.RESET_ALL}")
        return None
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Exchange error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False)
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=True
        )
        return None


@retry_api_call()
async def fetch_open_orders(
    exchange: ccxt.bybit,
    symbol: str | None = None,  # Optional: fetch only for a specific symbol
    order_filter: OrderFilter | None = OrderFilter.ORDER,  # Default to active limit/market orders ('Order')
    config: Config = None,  # Required for context/logging
) -> list[dict]:
    """Fetches open orders (active/conditional) for V5 UTA, allowing filtering.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: If specified, fetch orders only for this symbol.
        order_filter: V5 filter type (Order, StopOrder, tpslOrder, etc.).
                      If None, fetches all types (may require multiple calls depending on CCXT).
        config: Configuration object.

    Returns:
        A list of open orders matching the criteria, in CCXT unified format. Returns empty list on failure.
    """
    func_name = "fetch_open_orders"
    symbol_log = f" ({symbol})" if symbol else " (All Symbols)"
    filter_log = f" Filter: {order_filter.value}" if order_filter else " (All Filters)"
    log_prefix = f"[{func_name}{symbol_log}{filter_log}]"
    logger.debug(f"{log_prefix} Fetching open orders...")

    if not config:
        logger.error(f"{Fore.RED}{log_prefix} Config object is required.{Style.RESET_ALL}")
        return []
    if not exchange or not hasattr(exchange, "fetch_open_orders"):
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid exchange object or fetch_open_orders method missing.{Style.RESET_ALL}"
        )
        return []

    params: dict[str, Any] = {}
    category: Category | None = None

    # Determine category if fetching for a specific symbol
    if symbol:
        category = market_cache.get_category(symbol)
        if not category:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot determine category for symbol {symbol}. Cannot fetch symbol-specific orders.{Style.RESET_ALL}"
            )
            # Option 1: Fail - Return empty list
            return []
            # Option 2: Warn and fetch all categories (more complex, might return unrelated orders)
            # logger.warning(f"{log_prefix} Falling back to fetching open orders for all categories.")
            # symbol = None # Clear symbol if fetching all
            # category = None
        else:
            params["category"] = category.value  # Set category for the specific symbol
            logger.debug(f"{log_prefix} Determined category {category.value} for symbol {symbol}.")

    # Add V5 specific filter if provided
    if order_filter:
        params["orderFilter"] = order_filter.value

    all_open_orders: list[dict] = []

    # If fetching for a specific symbol (and category determined), make one call
    if symbol and category:
        try:
            logger.debug(f"{log_prefix} Querying category: {category.value} with params: {params}")
            orders = await exchange.fetch_open_orders(symbol=symbol, params=params)
            if orders:
                all_open_orders.extend(orders)
        except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} API error fetching for {symbol}: {type(e).__name__}. Retry handled.{Style.RESET_ALL}"
            )
            raise  # Let outer retry handle
        except AuthenticationError as e:
            logger.error(f"{Fore.RED}{log_prefix} Authentication error fetching orders: {e}{Style.RESET_ALL}")
            return []  # Fail fast on auth error
        except ExchangeError as e:
            logger.error(
                f"{Fore.RED}{log_prefix} Exchange error fetching orders for {symbol}: {type(e).__name__}: {e}{Style.RESET_ALL}",
                exc_info=False,
            )
            # Return empty on error for this specific symbol fetch
            return []
        except Exception as e:
            logger.error(
                f"{Fore.RED}{log_prefix} Unexpected error fetching orders for {symbol}: {type(e).__name__}: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            return []
    else:
        # Fetching for all symbols - requires iterating through relevant categories
        # Determine categories to fetch (e.g., Linear, Inverse, Spot, Option)
        categories_to_fetch: list[Category] = [
            Category.LINEAR,
            Category.INVERSE,
            Category.SPOT,
            Category.OPTION,
        ]  # Adjust as needed
        logger.info(f"{log_prefix} Fetching orders across categories: {[c.value for c in categories_to_fetch]}")

        for cat in categories_to_fetch:
            params["category"] = cat.value
            # We don't pass 'symbol' here as we want all symbols for this category
            try:
                logger.debug(f"{log_prefix} Querying category: {cat.value} with params: {params}")
                # Call fetch_open_orders without symbol, but with category param
                orders = await exchange.fetch_open_orders(symbol=None, params=params)
                if orders:
                    logger.debug(f"{log_prefix} Found {len(orders)} open orders in category {cat.value}.")
                    all_open_orders.extend(orders)
                # Short delay between category fetches if querying multiple
                await asyncio.sleep(0.1)  # Small delay to avoid hitting rate limits aggressively

            except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} API error fetching for category {cat.value}: {type(e).__name__}. Retry handled.{Style.RESET_ALL}"
                )
                # We might want to raise here to let the outer retry handle the entire operation
                # Or continue and potentially get partial results? Let's raise for simplicity.
                raise
            except AuthenticationError as e:
                logger.error(
                    f"{Fore.RED}{log_prefix} Authentication error fetching orders for {cat.value}: {e}{Style.RESET_ALL}"
                )
                return []  # Fail fast on auth error
            except ExchangeError as e:
                logger.error(
                    f"{Fore.RED}{log_prefix} Exchange error fetching orders for {cat.value}: {type(e).__name__}: {e}{Style.RESET_ALL}",
                    exc_info=False,
                )
                # Continue to the next category? Or return partial results? Let's continue.
            except Exception as e:
                logger.error(
                    f"{Fore.RED}{log_prefix} Unexpected error fetching orders for {cat.value}: {type(e).__name__}: {e}{Style.RESET_ALL}",
                    exc_info=True,
                )
                # Continue

    # Log final count
    count = len(all_open_orders)
    log_color = Fore.GREEN if count == 0 else Fore.YELLOW
    logger.info(f"{log_color}{log_prefix} Found {count} total open orders matching criteria.{Style.RESET_ALL}")
    return all_open_orders


@retry_api_call(
    max_retries=1, retry_on_exceptions=(NetworkError, RequestTimeout, RateLimitExceeded)
)  # Limit retries for cancel
async def cancel_order(exchange: ccxt.bybit, order_id: str, symbol: str, config: Config) -> bool:
    """Cancels a specific order by ID and symbol for V5, using category context."""
    func_name = "cancel_order"
    log_prefix = f"[{func_name} (ID: ...{format_order_id(order_id)}, Sym: {symbol})]"
    logger.info(f"{Fore.BLUE}{log_prefix} Attempting cancellation...{Style.RESET_ALL}")

    if not order_id:
        logger.error(f"{Fore.RED}{log_prefix} Order ID is required.{Style.RESET_ALL}")
        return False

    category = market_cache.get_category(symbol)
    if not category:
        logger.error(
            f"{Fore.RED}{log_prefix} Cannot determine category for symbol {symbol}. Cancellation requires category.{Style.RESET_ALL}"
        )
        return False

    params = {"category": category.value}
    # Bybit V5 cancel endpoint: /v5/order/cancel
    # It accepts either 'orderId' or 'orderLinkId'.
    # CCXT's cancel_order method typically sends the 'id' argument as 'orderId'.
    # If you need to cancel using a clientOrderId, you might need to use cancel_order_by_client_id or pass 'orderLinkId' in params explicitly.
    # Assuming 'order_id' passed here is the exchange's order ID.

    try:
        logger.debug(f"{log_prefix} Sending cancel_order request with params: {params}")
        # Use CCXT standard method, passing category in params
        response = await exchange.cancel_order(id=order_id, symbol=symbol, params=params)

        # --- Analyze Response ---
        # CCXT cancel_order behavior varies. It might return None, order structure, or specific info.
        # Success is often indicated by lack of exception.
        # Let's check the raw response ('info') for Bybit V5 confirmation if possible.
        logger.debug(f"{log_prefix} Raw cancel response: {response}")
        v5_result = {}
        if isinstance(response, dict) and "info" in response and isinstance(response["info"], dict):
            v5_raw_response = response["info"]
            # Example V5 success: {'retCode': 0, 'retMsg': 'OK', 'result': {'orderId': '...', 'orderLinkId': '...'}, ...}
            if v5_raw_response.get("retCode") == 0 and isinstance(v5_raw_response.get("result"), dict):
                v5_result = v5_raw_response["result"]

        cancelled_id_from_response = v5_result.get("orderId")

        # Check if the response confirms the cancellation of the requested ID
        if cancelled_id_from_response == order_id:
            logger.info(
                f"{Fore.GREEN}{log_prefix} Successfully cancelled (confirmed by API response ID).{Style.RESET_ALL}"
            )
            return True
        else:
            # If ID doesn't match or response is unclear, assume success based on no exception (CCXT standard)
            logger.info(
                f"{Fore.GREEN}{log_prefix} Cancellation request sent successfully (no exception raised). API confirmation unclear or ID mismatch (Expected: {order_id}, Got: {cancelled_id_from_response}).{Style.RESET_ALL}"
            )
            return True  # Treat as success if no error

    except OrderNotFound as e:
        # Order already filled, cancelled, or never existed. Treat as successful cancellation outcome.
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Order not found (may already be filled/cancelled): {e}{Style.RESET_ALL}"
        )
        return True  # Considered success from the perspective of ensuring it's not open
    except InvalidOrder as e:
        # E.g., trying to cancel an order in a final state (rejected, filled) where cancel is invalid
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid order state for cancellation (already filled/rejected?): {e}{Style.RESET_ALL}"
        )
        # This is arguably a failure if the intent was to cancel an assumed-open order.
        return False
    except ExchangeError as e:
        # Catch other specific exchange errors during cancellation
        logger.error(
            f"{Fore.RED}{log_prefix} Exchange error during cancellation: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=False,
        )
        return False
    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
        # Let retry decorator handle these by re-raising
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - API Communication Error: {type(e).__name__}: {e}{Style.RESET_ALL}"
        )
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error during cancellation: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return False


@retry_api_call(max_retries=1, retry_on_exceptions=(NetworkError, RequestTimeout, RateLimitExceeded))
async def cancel_all_orders(
    exchange: ccxt.bybit,
    symbol: str | None = None,  # Optional: cancel only for this symbol
    category_to_cancel: Category | None = None,  # Optional: cancel only for this category
    order_filter: OrderFilter | None = None,  # Optional: V5 filter (Order, StopOrder, etc.)
    config: Config = None,  # Required for context
) -> bool:
    """Cancels all open orders, optionally filtered by symbol, category, or V5 filter.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: If provided, cancel orders only for this symbol (requires category).
        category_to_cancel: If provided, cancel orders only within this category.
        order_filter: V5 filter (Order, StopOrder, tpslOrder). If None, cancels matching types based on endpoint.
        config: Configuration object.

    Returns:
        True if all targeted cancellation requests were sent successfully without critical errors,
        False otherwise. Note: This doesn't guarantee individual orders were cancelled if they were already closed.
    """
    func_name = "cancel_all_orders"
    symbol_log = f" ({symbol})" if symbol else " (All Symbols)"
    cat_log = f" Category: {category_to_cancel.value}" if category_to_cancel else ""
    filter_log = f" Filter: {order_filter.value}" if order_filter else ""
    log_prefix = f"[{func_name}{symbol_log}{cat_log}{filter_log}]"
    logger.info(f"{Fore.BLUE}{log_prefix} Attempting cancel-all operation...{Style.RESET_ALL}")

    if not config:
        logger.error(f"{Fore.RED}{log_prefix} Config object is required.{Style.RESET_ALL}")
        return False
    if not exchange or not hasattr(exchange, "private_post_v5_order_cancel_all"):
        logger.error(f"{Fore.RED}{log_prefix} Invalid exchange object or cancel-all method missing.{Style.RESET_ALL}")
        return False

    # --- Determine Target Categories ---
    categories_to_target: list[Category] = []
    if category_to_cancel:
        # Target only the specified category
        categories_to_target.append(category_to_cancel)
        # If symbol is also provided, ensure it matches the category
        if symbol:
            symbol_category = market_cache.get_category(symbol)
            if symbol_category != category_to_cancel:
                logger.error(
                    f"{Fore.RED}{log_prefix} Mismatch: Symbol {symbol} (Category: {symbol_category}) does not belong to specified category_to_cancel ({category_to_cancel.value}). Aborting.{Style.RESET_ALL}"
                )
                return False
    elif symbol:
        # Target only the category of the specified symbol
        cat = market_cache.get_category(symbol)
        if cat:
            categories_to_target.append(cat)
            logger.info(f"{log_prefix} Targeting category {cat.value} based on symbol {symbol}.")
        else:
            logger.error(
                f"{Fore.RED}{log_prefix} Cannot determine category for symbol {symbol}. Cannot target cancellation. Aborting.{Style.RESET_ALL}"
            )
            return False
    else:
        # No symbol, no specific category -> target all relevant categories
        categories_to_target = [Category.LINEAR, Category.INVERSE, Category.SPOT, Category.OPTION]  # Adjust as needed
        logger.info(f"{log_prefix} Targeting all categories: {[c.value for c in categories_to_target]}")

    # --- Execute Cancellation per Category ---
    overall_success = True  # Tracks if any category failed critically
    any_action_taken = False  # Tracks if any orders were actually cancelled

    for cat in categories_to_target:
        params: dict[str, Any] = {"category": cat.value}
        # Add symbol to params only if cancelling for a specific symbol within its category
        if symbol and cat == market_cache.get_category(symbol):
            params["symbol"] = symbol
        elif symbol and cat != market_cache.get_category(symbol):
            # This case should be prevented by checks above, but safeguard
            logger.warning(f"{log_prefix} Skipping category {cat.value} as it doesn't match target symbol {symbol}.")
            continue

        # Add V5 filter if provided
        if order_filter:
            params["orderFilter"] = order_filter.value
        # Bybit V5 endpoint: POST /v5/order/cancel-all

        try:
            logger.debug(f"{log_prefix} Sending cancel-all for Category: {cat.value} with params: {params}")
            # Use the implicit API call via CCXT
            response = await exchange.private_post_v5_order_cancel_all(params)
            logger.debug(f"{log_prefix} Raw cancel-all response for {cat.value}: {response}")

            # --- Process Response for this Category ---
            if not isinstance(response, dict):
                logger.error(
                    f"{Fore.RED}{log_prefix} Invalid response for category {cat.value}. Response: {response}{Style.RESET_ALL}"
                )
                overall_success = False
                continue  # Mark failure and move to next category

            ret_code = response.get("retCode")
            ret_msg = response.get("retMsg", "N/A")
            # Result list contains IDs of successfully cancelled orders (or is empty)
            result_list = response.get("result", {}).get("list", []) if isinstance(response.get("result"), dict) else []

            if ret_code == 0:
                cancelled_count = len(result_list) if isinstance(result_list, list) else 0
                if cancelled_count > 0:
                    any_action_taken = True
                    logger.info(
                        f"{Fore.GREEN}{log_prefix} Successfully cancelled {cancelled_count} orders in category {cat.value}.{Style.RESET_ALL}"
                    )
                else:
                    # Success code 0 but empty list means no matching orders found
                    logger.info(
                        f"{Fore.BLUE}{log_prefix} No matching open orders found to cancel in category {cat.value} (API Success Code 0).{Style.RESET_ALL}"
                    )
                # Consider this category successful
            else:
                # Handle specific non-error codes if necessary
                # Example: Bybit might return a specific code if no orders existed (e.g., 170101 - No orders to cancel)
                if ret_code == 170101:
                    logger.info(
                        f"{Fore.BLUE}{log_prefix} No matching open orders found to cancel in category {cat.value} (API Code: {ret_code} - {ret_msg}).{Style.RESET_ALL}"
                    )
                else:
                    # Log actual errors
                    logger.error(
                        f"{Fore.RED}{log_prefix} Cancel-all failed for category {cat.value}. Code: {ret_code}, Msg: {ret_msg}{Style.RESET_ALL}"
                    )
                    overall_success = False  # Mark overall operation as failed

            # Optional delay if iterating multiple categories rapidly
            if len(categories_to_target) > 1:
                await asyncio.sleep(0.1)  # Small delay

        except ExchangeError as e:
            # Catch exchange errors specific to this category's call
            logger.error(
                f"{Fore.RED}{log_prefix} Exchange error cancelling for {cat.value}: {type(e).__name__}: {e}{Style.RESET_ALL}",
                exc_info=False,
            )
            overall_success = False
        except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
            # Let retry decorator handle these by re-raising for the *entire* function call
            logger.error(
                f"{Back.RED}{log_prefix} FAILED - API Communication Error cancelling for {cat.value}: {type(e).__name__}: {e}{Style.RESET_ALL}"
            )
            raise e
        except Exception as e:
            # Catch unexpected errors for this category
            logger.error(
                f"{Fore.RED}{log_prefix} Unexpected error cancelling for {cat.value}: {type(e).__name__}: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            overall_success = False

    # --- Final Logging based on Outcome ---
    if overall_success:
        if any_action_taken:
            logger.info(
                f"{Fore.GREEN}{log_prefix} All targeted cancellation requests completed successfully.{Style.RESET_ALL}"
            )
        else:
            logger.info(
                f"{Fore.BLUE}{log_prefix} Cancellation requests completed, but no matching open orders were found across targeted categories.{Style.RESET_ALL}"
            )
    else:
        logger.error(
            f"{Fore.RED}{log_prefix} One or more category cancellation requests failed. Check logs above.{Style.RESET_ALL}"
        )

    return overall_success


# --- Conditional Order Functions (Basic Examples) ---
# Note: Bybit V5 has complex conditional order capabilities (TP/SL attached to positions, OCO, etc.)
# This function provides a basic standalone stop order (market or limit trigger).


@retry_api_call(max_retries=1, retry_on_exceptions=(NetworkError, RequestTimeout, RateLimitExceeded))
async def place_stop_order(
    exchange: ccxt.bybit,
    symbol: str,
    side: Side,  # Side of the order to be placed when triggered (e.g., SELL for stop-loss on LONG)
    amount: Decimal,
    trigger_price: Decimal,
    config: Config,
    order_type: OrderType = OrderType.MARKET,  # Type of order placed when triggered (Market or Limit)
    limit_price: Decimal | None = None,  # Required if order_type is Limit (price for the triggered limit order)
    is_reduce_only: bool = True,  # Typically True for SL/TP to only close position
    trigger_by: TriggerBy = TriggerBy.MARK,  # Price type to monitor (Mark, Last, Index)
    trigger_direction: TriggerDirection | None = None,  # Optional: 1=Rise, 2=Fall. Bybit often infers.
    position_idx: PositionIdx | None = None,  # For Hedge Mode context
    client_order_id: str | None = None,
    # V5 specific TP/SL params might be needed for more advanced usage:
    # tpslMode: 'Full' or 'Partial' (for position TP/SL)
    # basePrice: Reference price for trigger direction inference (optional)
    # stopOrderType: 'Stop', 'TakeProfit', 'StopLoss', 'TrailingStop', 'PartialTakeProfit', 'PartialStopLoss'
) -> dict | None:
    """Places a basic conditional stop order (stop-loss or take-profit trigger).

    Note: This uses the generic `create_order` method with params for conditional orders.
          Bybit V5 has more specific TP/SL features often tied directly to positions
          or using dedicated parameters not fully covered by this basic function.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: The market symbol.
        side: Side of the order to be PLACED when triggered (e.g., SELL to close a LONG).
        amount: Amount of the order to be placed when triggered.
        trigger_price: The price at which the order should be triggered.
        config: Configuration object.
        order_type: Type of order (Market or Limit) to place once triggered.
        limit_price: If order_type is Limit, this is the limit price for that triggered order.
        is_reduce_only: If True, the triggered order can only reduce position size.
        trigger_by: Which price source (Mark, Last, Index) triggers the order.
        trigger_direction: 1 for trigger on rise, 2 for trigger on fall. Often inferred by Bybit.
        position_idx: Hedge mode position index (0, 1, or 2).
        client_order_id: Custom identifier for the conditional order.

    Returns:
        The parsed conditional order dictionary from CCXT if successfully placed, else None.
        The status will typically be 'open' or 'untriggered'.
    """
    func_name = "place_stop_order"
    # Determine action description based on context (requires knowledge of existing position)
    # This basic function doesn't know the context, so generic "Stop" is used.
    action_desc = "Stop"  # Could be 'StopLoss', 'TakeProfit', 'EntryTrigger' depending on use case
    log_prefix = f"[{func_name} ({symbol}, TrigSide:{side.value}, Amt:{amount}, TrigPx:{trigger_price}, Type:{order_type.value}, Action:{action_desc})]"

    # --- Basic Validations ---
    qty_epsilon = config.get("POSITION_QTY_EPSILON", Decimal("1E-8"))
    if amount <= qty_epsilon or trigger_price <= Decimal(0):
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid amount ({amount}) or trigger price ({trigger_price}). Must be positive.{Style.RESET_ALL}"
        )
        return None
    if order_type == OrderType.LIMIT and (limit_price is None or limit_price <= Decimal(0)):
        logger.error(
            f"{Fore.RED}{log_prefix} Limit stop order requires a valid positive 'limit_price'. Received: {limit_price}.{Style.RESET_ALL}"
        )
        return None
    if order_type == OrderType.MARKET and limit_price is not None:
        logger.warning(
            f"{Fore.YELLOW}{log_prefix} 'limit_price' was provided for a Market stop order. It will likely be ignored by the exchange.{Style.RESET_ALL}"
        )

    category = market_cache.get_category(symbol)
    market = market_cache.get_market(symbol)
    # Conditional orders primarily for derivatives, but Spot might support basic stops too.
    if not category or not market or category not in [Category.LINEAR, Category.INVERSE, Category.SPOT]:
        logger.error(
            f"{Fore.RED}{log_prefix} Invalid category/market ({category}) for stop order on {symbol}.{Style.RESET_ALL}"
        )
        return None

    # Format numbers according to market precision
    formatted_amount_str = format_amount(exchange, symbol, amount)
    formatted_trigger_price_str = format_price(exchange, symbol, trigger_price)
    formatted_limit_price_str = (
        format_price(exchange, symbol, limit_price) if order_type == OrderType.LIMIT and limit_price else None
    )

    if formatted_amount_str is None or formatted_trigger_price_str is None:
        logger.error(
            f"{Fore.RED}{log_prefix} Failed to format amount ({amount}) or trigger price ({trigger_price}).{Style.RESET_ALL}"
        )
        return None
    if order_type == OrderType.LIMIT and formatted_limit_price_str is None:
        logger.error(
            f"{Fore.RED}{log_prefix} Failed to format limit price ({limit_price}) for Limit stop order.{Style.RESET_ALL}"
        )
        return None

    logger.info(f"{Fore.BLUE}{log_prefix} Placing conditional order request...{Style.RESET_ALL}")

    # --- Prepare V5 Parameters for Conditional Order via create_order ---
    # CCXT often maps high-level types like 'stop' or 'stop_limit' to underlying params.
    # We pass specific V5 params explicitly for clarity and control.
    params: dict[str, Any] = {
        "category": category.value,
        "triggerPrice": formatted_trigger_price_str,
        "triggerBy": trigger_by.value,
        "reduceOnly": is_reduce_only,
        # V5 specific params for conditional orders:
        "orderType": order_type.value,  # This is the type ('Market' or 'Limit') of the order placed *after* trigger
        # 'stopOrderType': 'Stop', # Explicitly defining the conditional type (vs TP/SL attached to position)
        # Let CCXT handle mapping based on 'type' argument below if possible.
        # --- TP/SL Specific Params (Might be needed for certain stopOrderType values) ---
        # 'tpslMode': 'Full', # or 'Partial' - Usually for position TP/SL
        # 'slLimitPrice': formatted_limit_price_str if order_type == OrderType.LIMIT else None, # V5 name for limit price of SL order
        # 'tpLimitPrice': formatted_limit_price_str if order_type == OrderType.LIMIT else None, # V5 name for limit price of TP order
    }
    # Add triggerDirection if specified (Bybit often infers this based on trigger_price vs current price)
    if trigger_direction:
        params["triggerDirection"] = trigger_direction.value
    # Add limit price for the triggered Limit order (CCXT create_order 'price' arg usually maps here)
    # V5 uses 'price' field within the conditional order request for the limit price of the triggered order.
    # We will pass this via the main 'price' argument to create_order below.

    # Add Client Order ID if provided
    if client_order_id:
        clean_cid = "".join(filter(lambda c: c.isalnum() or c in ["-", "_"], client_order_id))[:36]
        if len(clean_cid) != len(client_order_id):
            logger.warning(f"{log_prefix} Client Order ID sanitized: '{clean_cid}'")
        params["orderLinkId"] = clean_cid
    # Add Position Index if provided
    if position_idx is not None:
        params["positionIdx"] = position_idx.value

    # Remove None values from params before sending
    params_cleaned = {k: v for k, v in params.items() if v is not None}

    try:
        logger.info(f"{log_prefix} Sending create_order request for conditional order with params: {params_cleaned}...")

        # --- Determine CCXT order type and price argument ---
        # CCXT uses specific types ('stop', 'stop_limit', 'take_profit', 'take_profit_limit')
        # We need to map our OrderType (Market/Limit) to CCXT's conditional types.
        ccxt_type = "stop" if order_type == OrderType.MARKET else "stop_limit"
        # Price argument for create_order: Use None for Market stops, limit_price for Limit stops.
        price_arg = (
            float(formatted_limit_price_str) if order_type == OrderType.LIMIT and formatted_limit_price_str else None
        )

        # Use the generic create_order method, passing V5 params
        # CCXT's Bybit implementation should handle mapping 'stop'/'stop_limit' with triggerPrice etc.
        order = await exchange.create_order(
            symbol=symbol,
            type=ccxt_type,  # Use CCXT's conditional type name
            side=side.value,  # Side of the order to be placed on trigger
            amount=float(formatted_amount_str),
            price=price_arg,  # Limit price for stop-limit, None for stop-market
            params=params_cleaned,  # Pass all specific V5 params here
        )

        # --- Process Response ---
        if not order or not isinstance(order, dict):
            logger.error(
                f"{Fore.RED}{log_prefix} FAILED - Received invalid order response from create_order.{Style.RESET_ALL}"
            )
            return None

        # Log Success (Conditional orders usually return ID immediately, status is 'untriggered' or similar)
        order_id = order.get("id")
        status = order.get("status", "unknown")  # Should be 'open' or specific conditional status like 'untriggered'
        # CCXT might place trigger price in 'stopPrice' or 'triggerPrice'
        returned_trigger_price_raw = order.get("triggerPrice") or order.get("stopPrice")
        returned_trigger_price = safe_decimal_conversion(returned_trigger_price_raw)
        returned_limit_price = safe_decimal_conversion(order.get("price"))  # Limit price of triggered order

        log_color = Fore.YELLOW  # Conditional orders are pending until triggered
        log_msg = f"{log_color}{log_prefix} SUCCESS (Conditional Order Placed) - ID: ...{format_order_id(order_id)}, Status: {status}, TriggerPx: {format_price(exchange, symbol, returned_trigger_price)}"
        if order_type == OrderType.LIMIT:
            log_msg += f", LimitPx: {format_price(exchange, symbol, returned_limit_price)}"
        logger.info(log_msg + Style.RESET_ALL)
        return order

    # --- Error Handling ---
    except InsufficientFunds as e:
        # Might occur if margin is needed even for conditional order placement
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - Insufficient Funds (check margin requirements?): {e}{Style.RESET_ALL}"
        )
        return None
    except InvalidOrder as e:  # Covers parameter errors, price/size constraints etc.
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - Invalid Order Parameters/Rejected by Exchange: {e}{Style.RESET_ALL}"
        )
        return None
    except ExchangeError as e:  # Catch other specific exchange errors
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - Exchange Error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False
        )
        return None
    except (NetworkError, ExchangeNotAvailable, RateLimitExceeded) as e:
        # Let retry decorator handle these by re-raising
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - API Communication Error: {type(e).__name__}: {e}{Style.RESET_ALL}"
        )
        raise e
    except Exception as e:  # Catch any other unexpected errors
        logger.error(
            f"{Back.RED}{log_prefix} FAILED - Unexpected Error: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=True
        )
        return None


# --- WebSocket Management Stubs ---
# Full implementation is complex and requires careful state management.
# These are basic placeholders demonstrating the structure.


async def connect_ws(exchange: ccxt.bybit, config: Config) -> WebSocketClientProtocol | None:
    """Placeholder: Establishes and authenticates a WebSocket connection."""
    func_name = "connect_ws"
    log_prefix = f"[{func_name}]"
    if not websockets:
        logger.error(
            f"{Fore.RED}{log_prefix} 'websockets' library not installed. Cannot establish WebSocket connection.{Style.RESET_ALL}"
        )
        return None

    # Determine WS URL based on public/private and mainnet/testnet
    is_testnet = config.get("TESTNET_MODE", False)
    has_keys = bool(config.get("API_KEY") and config.get("API_SECRET"))
    # Bybit V5 WS Endpoints:
    # Public Mainnet: wss://stream.bybit.com/v5/public/{category} (category=linear,inverse,spot,option)
    # Private Mainnet: wss://stream.bybit.com/v5/private
    # Public Testnet: wss://stream-testnet.bybit.com/v5/public/{category}
    # Private Testnet: wss://stream-testnet.bybit.com/v5/private
    base_url = "wss://stream-testnet.bybit.com/v5" if is_testnet else "wss://stream.bybit.com/v5"
    ws_url = f"{base_url}/private" if has_keys else f"{base_url}/public/linear"  # Default to public linear if no keys

    logger.info(f"{Fore.BLUE}{log_prefix} Attempting to connect to WebSocket: {ws_url}...{Style.RESET_ALL}")
    connect_timeout = config.get("WS_CONNECT_TIMEOUT", 10.0)  # Timeout for connection attempt

    try:
        # Establish connection
        # Add extra headers if needed (User-Agent etc.)
        # ping_interval=None disables automatic pings by websockets library if Bybit requires manual pings
        ws = await asyncio.wait_for(websockets.connect(ws_url, ping_interval=None), timeout=connect_timeout)
        logger.info(f"{Fore.GREEN}{log_prefix} WebSocket connection established to {ws_url}.{Style.RESET_ALL}")

        # Authenticate if private connection
        if has_keys:
            logger.info(f"{log_prefix} Authenticating private WebSocket connection...")
            auth_success = await ws_authenticate(ws, config)
            if not auth_success:
                logger.error(
                    f"{Fore.RED}{log_prefix} WebSocket authentication failed. Closing connection.{Style.RESET_ALL}"
                )
                await ws.close()
                return None
            logger.info(f"{Fore.GREEN}{log_prefix} WebSocket authentication successful.{Style.RESET_ALL}")

        # Start manual heartbeat task if needed (Bybit requires ping every 20s)
        ping_interval = config.get("WS_PING_INTERVAL", 20.0)
        if ping_interval:
            logger.info(f"{log_prefix} Starting WebSocket heartbeat task (Interval: {ping_interval}s).")
            asyncio.create_task(ws_heartbeat(ws, ping_interval, log_prefix))  # Run in background

        return ws  # Return the active WebSocket connection object

    except TimeoutError:
        logger.error(
            f"{Fore.RED}{log_prefix} WebSocket connection timed out ({connect_timeout}s) to {ws_url}.{Style.RESET_ALL}"
        )
        return None
    except (InvalidURI, WebSocketException, OSError) as e:  # Catch connection errors
        logger.error(f"{Fore.RED}{log_prefix} WebSocket connection failed: {type(e).__name__}: {e}{Style.RESET_ALL}")
        return None
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error during WebSocket connection: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


async def ws_authenticate(ws: WebSocketClientProtocol, config: Config) -> bool:
    """Placeholder: Sends authentication message for private WebSocket stream."""
    if not ws or not config.get("API_KEY") or not config.get("API_SECRET"):
        return False  # Cannot authenticate without keys

    # Bybit V5 WS Auth:
    # 1. Calculate expires timestamp (current time + validity window in ms)
    # 2. Create signature: hex(HMAC_SHA256(secret, "GET/realtime" + expires))
    # 3. Send auth message: {"op": "auth", "args": [api_key, expires, signature]}
    try:
        api_key = config["API_KEY"]
        api_secret = config["API_SECRET"]
        expires = int((time.time() + 10) * 1000)  # Expires 10 seconds from now (in ms)
        signature_payload = f"GET/realtime{expires}"

        # Use CCXT's utility for HMAC signing if available, otherwise implement manually
        if hasattr(exchange, "hmac"):  # Check if exchange instance has hmac method
            signature = exchange.hmac(exchange.encode(signature_payload), exchange.encode(api_secret), "sha256")
        else:  # Manual implementation (requires 'hashlib' and 'hmac')
            import hashlib
            import hmac

            signature = hmac.new(
                api_secret.encode("utf-8"), signature_payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()

        auth_msg = {"op": "auth", "args": [api_key, expires, signature]}
        await ws.send(json.dumps(auth_msg))
        logger.debug("[ws_authenticate] Auth message sent.")

        # Wait for auth response (important!)
        # Bybit sends: {"op":"auth","success":true/false,"ret_msg":"...", ...}
        try:
            response_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            response = json.loads(response_raw)
            logger.debug(f"[ws_authenticate] Auth response received: {response}")
            if response.get("op") == "auth" and response.get("success") is True:
                return True
            else:
                logger.error(f"[ws_authenticate] Auth failed. Response: {response}")
                return False
        except TimeoutError:
            logger.error("[ws_authenticate] Timeout waiting for authentication response.")
            return False
        except (json.JSONDecodeError, WebSocketException) as e:
            logger.error(f"[ws_authenticate] Error receiving/parsing auth response: {e}")
            return False

    except Exception as e:
        logger.error(f"[ws_authenticate] Error during authentication process: {e}", exc_info=True)
        return False


async def ws_heartbeat(ws: WebSocketClientProtocol, interval: float, log_prefix: str):
    """Placeholder: Sends periodic pings to keep WebSocket connection alive."""
    while ws and not ws.closed:
        try:
            ping_payload = json.dumps({"req_id": f"hb_{int(time.time())}", "op": "ping"})  # Add req_id for V5
            await ws.send(ping_payload)
            logger.debug(f"{log_prefix} Sent ping.")
            await asyncio.sleep(interval)
        except ConnectionClosed:
            logger.warning(f"{log_prefix} Heartbeat task: Connection closed. Stopping pings.")
            break
        except WebSocketException as e:
            logger.error(f"{log_prefix} Heartbeat task: WebSocket error sending ping: {e}. Stopping pings.")
            break
        except Exception as e:
            logger.error(f"{log_prefix} Heartbeat task: Unexpected error: {e}. Stopping pings.", exc_info=True)
            break
    logger.info(f"{log_prefix} Heartbeat task finished.")


async def ws_subscribe(ws: WebSocketClientProtocol, topics: list[str], config: Config) -> bool:
    """Placeholder: Subscribes to specified WebSocket topics."""
    func_name = "ws_subscribe"
    log_prefix = f"[{func_name}]"
    if not ws or not topics:
        logger.error(f"{log_prefix} WebSocket not connected or no topics specified.")
        return False
    if not websockets:
        return False  # Check library loaded

    logger.info(f"{Fore.BLUE}{log_prefix} Subscribing to topics: {topics}...{Style.RESET_ALL}")
    try:
        # Format subscription message according to Bybit V5 WS API specs
        # e.g., {"op": "subscribe", "args": ["kline.1m.BTCUSDT", "order"]}
        subscribe_message = {
            "req_id": f"sub_{int(time.time())}",  # Optional but good practice for V5
            "op": "subscribe",
            "args": topics,
        }
        await ws.send(json.dumps(subscribe_message))
        logger.debug(f"{log_prefix} Subscription message sent: {subscribe_message}")

        # Optionally, wait for subscription confirmation response from Bybit
        # Bybit sends: {"op":"subscribe","success":true/false,"ret_msg":"...", "conn_id":"...", "req_id":"..."}
        # This requires more complex logic in the message handler to correlate responses.
        # For simplicity here, we assume success if send doesn't fail.
        logger.info(f"{Fore.GREEN}{log_prefix} Subscription request sent for topics: {topics}.{Style.RESET_ALL}")
        return True

    except (WebSocketException, ConnectionClosed) as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Error sending subscription message: {type(e).__name__}: {e}{Style.RESET_ALL}"
        )
        return False
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected error during subscription: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return False


async def ws_message_handler(ws: WebSocketClientProtocol, message_queue: asyncio.Queue, config: Config):
    """Placeholder: Listens for messages, parses them, and puts relevant data on a queue."""
    func_name = "ws_message_handler"
    log_prefix = f"[{func_name}]"
    if not ws or not websockets:
        logger.error(f"{log_prefix} WebSocket not available. Cannot start message handler.")
        return
    if not message_queue:
        logger.error(f"{log_prefix} Message queue not provided. Cannot start message handler.")
        return

    logger.info(f"{Fore.BLUE}{log_prefix} Starting WebSocket message listener loop...{Style.RESET_ALL}")
    try:
        # Continuously listen for incoming messages
        async for message_raw in ws:
            try:
                # Attempt to parse the message as JSON
                data = json.loads(message_raw)
                # logger.debug(f"{log_prefix} Received WS message: {str(data)[:200]}") # Very verbose

                # --- Message Handling Logic ---
                # 1. Handle Pong responses to our Pings (confirm connection is alive)
                if isinstance(data, dict) and data.get("op") == "pong":
                    logger.debug(f"{log_prefix} Received pong: {data}")
                    continue  # Nothing more to do with pongs

                # 2. Handle Subscription confirmations (optional, for logging/state)
                if isinstance(data, dict) and data.get("op") == "subscribe":
                    req_id = data.get("req_id")
                    success = data.get("success")
                    ret_msg = data.get("ret_msg", "")
                    if success:
                        logger.info(
                            f"{Fore.GREEN}{log_prefix} Subscription successful (ReqID: {req_id}). Msg: {ret_msg}{Style.RESET_ALL}"
                        )
                    else:
                        logger.error(
                            f"{Fore.RED}{log_prefix} Subscription failed (ReqID: {req_id}). Msg: {ret_msg}{Style.RESET_ALL}"
                        )
                    continue

                # 3. Handle Authentication responses (should be handled during connect usually)
                if isinstance(data, dict) and data.get("op") == "auth":
                    logger.info(f"{log_prefix} Received unexpected auth response in main loop: {data}")
                    continue  # Already handled during connect

                # 4. Identify actual data messages (e.g., kline, orderbook, orders, positions)
                # Bybit V5 uses 'topic' field for data streams
                topic = data.get("topic") if isinstance(data, dict) else None
                if topic:
                    # logger.debug(f"{log_prefix} Received data for topic: {topic}")
                    # Put the relevant data onto the queue for processing elsewhere
                    await message_queue.put(data)
                # Handle heartbeat messages from server if any (Bybit V5 doesn't usually send unsolicited pings)
                # elif is_server_heartbeat(data): # Implement check if needed
                #    logger.debug(f"{log_prefix} Received server heartbeat.")
                else:
                    # Log unexpected message formats or types
                    logger.warning(
                        f"{log_prefix} Received unhandled WebSocket message format/type: {str(data)[:200]}..."
                    )

            except json.JSONDecodeError:
                logger.error(f"{log_prefix} Failed to decode JSON from WebSocket message: {message_raw}")
            except asyncio.QueueFull:
                logger.error(f"{log_prefix} Message queue is full! Data is being produced faster than consumed.")
                # Implement backpressure or increase queue size if this happens often
                await asyncio.sleep(0.1)  # Small delay before trying to put again
            except Exception as e:
                # Catch errors during processing of a single message
                logger.error(f"{log_prefix} Error processing WebSocket message: {type(e).__name__}: {e}", exc_info=True)

    # Handle connection closure or errors in the listener loop itself
    except (ConnectionClosed, WebSocketException) as e:
        logger.error(
            f"{Fore.RED}{log_prefix} WebSocket connection closed or error occurred: {type(e).__name__}: {e}. Listener loop stopped.{Style.RESET_ALL}"
        )
        # Signal that reconnection might be needed by putting a special marker in the queue or using another flag
        try:
            await message_queue.put({"type": "WS_CLOSED", "error": str(e)})
        except asyncio.QueueFull:
            logger.error(f"{log_prefix} Queue full while trying to signal WS closure.")
    except Exception as e:
        logger.error(
            f"{Fore.RED}{log_prefix} Unexpected critical error in WebSocket listener loop: {type(e).__name__}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        try:
            await message_queue.put({"type": "WS_ERROR", "error": str(e)})
        except asyncio.QueueFull:
            logger.error(f"{log_prefix} Queue full while trying to signal WS error.")
    finally:
        logger.info(f"{log_prefix} WebSocket message listener loop has finished.")


# --- Main Execution Guard (Example Usage) ---
async def example_usage():
    """Demonstrates basic usage of the enhanced helper functions."""
    # --- Load Configuration ---
    # In a real application, load this securely (e.g., .env file, config parser)
    cfg: Config = {
        "EXCHANGE_ID": "bybit",
        "API_KEY": os.environ.get("BYBIT_API_KEY"),
        "API_SECRET": os.environ.get("BYBIT_API_SECRET"),
        "TESTNET_MODE": True,  # <--- Set to False for Mainnet!
        "SYMBOL": "BTC/USDT:USDT",  # Example Linear Perpetual
        "USDT_SYMBOL": "USDT",
        "DEFAULT_MARGIN_MODE": "isolated",  # or "cross"
        "DEFAULT_RECV_WINDOW": 10000,  # Increased from default 5000ms
        "DEFAULT_SLIPPAGE_PCT": Decimal("0.005"),  # 0.5% allowed for market order spread check
        "POSITION_QTY_EPSILON": Decimal("1E-8"),  # For float comparisons
        "SHALLOW_OB_FETCH_DEPTH": 5,  # For quick spread checks
        "ORDER_BOOK_FETCH_LIMIT": 50,  # Default limit for fetch_l2_order_book
        "EXPECTED_MARKET_TYPE": "swap",  # Default context type
        "EXPECTED_MARKET_LOGIC": "linear",  # Default context logic
        "RETRY_COUNT": 3,  # Default retry attempts for API calls
        "RETRY_DELAY_SECONDS": 1.5,  # Initial delay for retries
        "ENABLE_SMS_ALERTS": False,  # Set to True and configure Twilio keys to enable
        "VERSION": "3.5",  # Match script version
        # "BROKER_ID": "YOUR_BROKER_ID", # Optional: Set your Bybit Broker/Referral ID
        "WS_CONNECT_TIMEOUT": 15.0,  # WebSocket connection timeout
        "WS_PING_INTERVAL": 20.0,  # Send ping every 20s for Bybit WS
    }

    # Basic check for API keys
    has_keys = bool(cfg["API_KEY"] and cfg["API_SECRET"])
    if not has_keys:
        print(
            f"{Fore.YELLOW}Warning: BYBIT_API_KEY or BYBIT_API_SECRET environment variables not set.{Style.RESET_ALL}"
        )
        print(f"{Fore.YELLOW}Proceeding in public-only mode. Private endpoints will be skipped.{Style.RESET_ALL}")
        # Decide if you want to exit or continue in public mode
        # return # Uncomment to exit if keys are mandatory

    # --- Initialize Exchange ---
    print("\n--- Initializing Bybit Exchange ---")
    exchange = await initialize_bybit(cfg)

    if not exchange:
        print(f"{Back.RED}FATAL: Failed to initialize Bybit exchange. Exiting example.{Style.RESET_ALL}")
        return  # Exit if initialization failed

    print(
        f"\n{Fore.GREEN}--- Exchange Initialized Successfully ({exchange.id}, Mode: {'Testnet' if cfg['TESTNET_MODE'] else 'Mainnet'}) ---{Style.RESET_ALL}"
    )

    # --- Example API Calls ---
    try:
        print(f"\n--- Fetching Ticker ({cfg['SYMBOL']}) ---")
        ticker = await fetch_ticker_validated(exchange, cfg["SYMBOL"], cfg)
        if ticker:
            print(
                f"  Symbol: {ticker.get('symbol')}, Last: {ticker.get('last')}, Bid: {ticker.get('bid')}, Ask: {ticker.get('ask')}"
            )
            # Store last price for placing test orders relative to it
            last_price = safe_decimal_conversion(ticker.get("last"))
        else:
            print(f"{Fore.RED}  Failed to fetch ticker.{Style.RESET_ALL}")
            last_price = None  # Cannot place relative orders

        print(f"\n--- Fetching OHLCV ({cfg['SYMBOL']}, 1h, limit=5) ---")
        ohlcv = await fetch_ohlcv_paginated(exchange, cfg["SYMBOL"], "1h", limit=5, config=cfg)
        if ohlcv is not None:
            print("  OHLCV Data Sample:")
            if pd and isinstance(ohlcv, pd.DataFrame):
                print(ohlcv.tail().to_string())  # Print last 5 rows if DataFrame
            elif isinstance(ohlcv, list):
                for candle in ohlcv[-5:]:
                    print(f"  - {candle}")  # Print last 5 if list
        else:
            print(f"{Fore.RED}  Failed to fetch OHLCV.{Style.RESET_ALL}")

        # --- Private Endpoint Examples (Require API Keys) ---
        if has_keys:
            print("\n--- Fetching USDT Balance (Unified Account) ---")
            equity, available = await fetch_usdt_balance(exchange, cfg)
            if equity is not None and available is not None:
                print(f"  Total Equity: {equity:.4f} {cfg['USDT_SYMBOL']}")
                print(f"  Available Balance: {available:.4f} {cfg['USDT_SYMBOL']}")
            else:
                print(f"{Fore.RED}  Failed to fetch balance.{Style.RESET_ALL}")

            print(f"\n--- Fetching Position ({cfg['SYMBOL']}) ---")
            position = await fetch_position(exchange, cfg["SYMBOL"], cfg)
            # Position dict can be complex, print key info
            if position:
                print(
                    f"  Symbol: {position.get('symbol')}, Side: {position.get('side', 'flat')}, Size: {position.get('contracts', 0)}, Entry: {position.get('entryPrice', 'N/A')}, LiqPx: {position.get('liquidationPrice', 'N/A')}"
                )
            else:
                # This could mean flat, or an error occurred during fetch
                print(f"{Fore.YELLOW}  No position data returned (likely flat or fetch error).{Style.RESET_ALL}")

            print(f"\n--- Fetching Open Orders ({cfg['SYMBOL']}, Filter: StopOrder) ---")
            # Example: Fetch only open conditional stop orders
            open_stop_orders = await fetch_open_orders(
                exchange, cfg["SYMBOL"], order_filter=OrderFilter.STOP_ORDER, config=cfg
            )
            print(f"  Found {len(open_stop_orders)} open STOP orders for {cfg['SYMBOL']}.")
            for order in open_stop_orders:
                print(
                    f"  - ID: ...{format_order_id(order['id'])}, Side: {order['side']}, TrigPx: {order.get('stopPrice') or order.get('triggerPrice', 'N/A')}, Amount: {order['amount']}"
                )

            # --- Example Order Placement (Use with extreme caution on Mainnet!) ---
            if (
                cfg["TESTNET_MODE"] and last_price
            ):  # Only run placement examples on Testnet AND if ticker fetch succeeded
                print("\n--- Placing Small Test Limit Order (Testnet Only) ---")
                try:
                    # Determine small amount based on market minimums (hardcoded for BTC example)
                    min_qty_btc = Decimal("0.001")
                    test_amount = min_qty_btc
                    # Place order significantly away from current price to avoid immediate fill
                    test_price_buy = last_price * Decimal("0.8")  # 20% below last
                    test_price_sell = last_price * Decimal("1.2")  # 20% above last

                    limit_order_buy = await place_limit_order_tif(
                        exchange,
                        cfg["SYMBOL"],
                        Side.BUY,
                        test_amount,
                        test_price_buy,
                        cfg,
                        time_in_force=TimeInForce.GTC,
                        client_order_id=f"test_buy_{int(time.time())}",
                    )
                    if limit_order_buy:
                        print(
                            f"  Placed Buy Limit Order ID: ...{format_order_id(limit_order_buy.get('id'))}, Status: {limit_order_buy.get('status')}"
                        )
                    else:
                        print(f"{Fore.RED}  Failed to place test buy limit order.{Style.RESET_ALL}")

                    await asyncio.sleep(1)  # Small delay

                    limit_order_sell = await place_limit_order_tif(
                        exchange,
                        cfg["SYMBOL"],
                        Side.SELL,
                        test_amount,
                        test_price_sell,
                        cfg,
                        time_in_force=TimeInForce.GTC,
                        client_order_id=f"test_sell_{int(time.time())}",
                    )
                    if limit_order_sell:
                        print(
                            f"  Placed Sell Limit Order ID: ...{format_order_id(limit_order_sell.get('id'))}, Status: {limit_order_sell.get('status')}"
                        )
                    else:
                        print(f"{Fore.RED}  Failed to place test sell limit order.{Style.RESET_ALL}")

                    await asyncio.sleep(2)  # Wait a bit before cancelling

                    print(f"\n--- Cancelling All Orders ({cfg['SYMBOL']}) from Test ---")
                    # Cancel all orders specifically for this symbol placed during the test
                    cancel_all_success = await cancel_all_orders(exchange, cfg["SYMBOL"], config=cfg)
                    print(f"  Cancel All for {cfg['SYMBOL']} Status: {cancel_all_success}")

                except Exception as e:
                    print(f"{Fore.RED}  Error during test order placement/cancellation: {e}{Style.RESET_ALL}")
                    # Attempt cleanup even on error
                    print(
                        f"{Fore.YELLOW}  Attempting cleanup: Cancelling all orders for {cfg['SYMBOL']} after error...{Style.RESET_ALL}"
                    )
                    await cancel_all_orders(exchange, cfg["SYMBOL"], config=cfg)

                # Example: Place Batch Orders (Testnet Only)
                print("\n--- Placing Batch Orders (Testnet Only) ---")
                # Requires last_price to be fetched successfully
                batch_order_requests = [
                    {  # Valid Buy Limit Below Market
                        "symbol": cfg["SYMBOL"],
                        "side": Side.BUY,
                        "type": "Limit",  # Use strings for type now
                        "amount": Decimal("0.001"),
                        "price": last_price * Decimal("0.9"),
                        "clientOrderId": f"batch_buy_{int(time.time())}",
                    },
                    {  # Valid Sell Limit Above Market
                        "symbol": cfg["SYMBOL"],
                        "side": Side.SELL,
                        "type": "Limit",
                        "amount": Decimal("0.001"),
                        "price": last_price * Decimal("1.1"),
                        "clientOrderId": f"batch_sell_{int(time.time())}",
                    },
                    {  # Invalid Order (e.g., amount too small - expect API error)
                        "symbol": cfg["SYMBOL"],
                        "side": Side.BUY,
                        "type": "Limit",
                        "amount": Decimal("0.0000001"),
                        "price": last_price * Decimal("0.8"),
                        "clientOrderId": f"batch_invalid_amt_{int(time.time())}",
                    },
                    {  # Valid Market Order (Will likely execute immediately on testnet)
                        "symbol": cfg["SYMBOL"],
                        "side": Side.BUY,
                        "type": "Market",
                        "amount": Decimal("0.001"),  # Amount for market order
                        "timeInForce": TimeInForce.IOC,  # Good practice for market
                        "clientOrderId": f"batch_market_{int(time.time())}",
                    },
                ]

                # Call the batch order function
                successes, errors = await place_batch_orders(exchange, batch_order_requests, cfg)

                print("  Batch Results:")
                for i, (s, e) in enumerate(zip(successes, errors, strict=False)):
                    orig_req = batch_order_requests[i]
                    cid = orig_req.get("clientOrderId", "N/A")
                    outcome = ""
                    if s:
                        outcome = f"{Fore.GREEN}SUCCESS: ID ...{format_order_id(s.get('id'))}, Status: {s.get('status')}{Style.RESET_ALL}"
                    elif e:
                        outcome = f"{Fore.RED}FAILED: Code={e.get('code')}, Msg='{e.get('msg')}'{Style.RESET_ALL}"
                    else:
                        outcome = f"{Fore.YELLOW}UNKNOWN STATUS (Should have success or error){Style.RESET_ALL}"
                    print(f"    Order #{i + 1} (CID:{cid}) -> {outcome}")

                # Clean up any potentially open orders from the batch test
                await asyncio.sleep(2)
                print(f"\n--- Cancelling All Orders ({cfg['SYMBOL']}) from Batch Test ---")
                await cancel_all_orders(exchange, cfg["SYMBOL"], config=cfg)

            elif not cfg["TESTNET_MODE"]:
                print(f"\n{Fore.YELLOW}--- Skipping Order Placement Examples (Not on Testnet) ---{Style.RESET_ALL}")
            else:  # Testnet but last_price failed
                print(
                    f"\n{Fore.YELLOW}--- Skipping Order Placement Examples (Failed to get ticker price) ---{Style.RESET_ALL}"
                )

        else:  # No API keys
            print(f"\n{Fore.YELLOW}--- Skipping Private Endpoint Examples (API Keys Not Provided) ---{Style.RESET_ALL}")

    except Exception as e:
        print(
            f"\n{Back.RED}--- An unexpected error occurred during example usage: {type(e).__name__}: {e} ---{Style.RESET_ALL}"
        )
        import traceback

        traceback.print_exc()

    finally:
        # --- Close Exchange Connection ---
        # Crucial step to release resources
        if exchange and hasattr(exchange, "close"):
            print("\n--- Closing Exchange Connection ---")
            try:
                await exchange.close()
                print(f"{Fore.GREEN}  Exchange connection closed successfully.{Style.RESET_ALL}")
            except Exception as close_err:
                print(f"{Fore.RED}  Error closing exchange connection: {close_err}{Style.RESET_ALL}")


# --- Main Entry Point ---
if __name__ == "__main__":
    # Configure root logger level if needed (e.g., for libraries)
    # logging.getLogger('ccxt').setLevel(logging.WARNING)
    # logging.getLogger('websockets').setLevel(logging.INFO)

    # Set overall logging level for this script's logger
    # logger.setLevel(logging.DEBUG) # Uncomment for more verbose debug output

    print("--- Running Bybit Helper Example Usage ---")
    try:
        # Use asyncio.run() for simple top-level execution
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("\nExecution interrupted by user (Ctrl+C).")
    except Exception as main_err:
        print(f"\n{Back.RED}--- Main execution failed: {type(main_err).__name__}: {main_err} ---{Style.RESET_ALL}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n--- Example Usage Finished ---")
