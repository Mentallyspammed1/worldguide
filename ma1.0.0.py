# ============ FILE: pyrmethus_market_analyzer_v3.1_async_stable.py ============
# ==============================================================================
# 🔥 Pyrmethus's Arcane Market Analyzer v3.1 ASYNC Stable Edition 🔥
# ==============================================================================
# Description:
#   Real-time cryptocurrency market analysis tool leveraging asynchronous
#   operations for enhanced performance and responsiveness. Utilizes ccxt.pro
#   for WebSocket connections (ticker, order book) and asyncio for concurrent
#   data fetching (balance, positions, OHLCV), calculations, and display updates.
#   Features interactive limit order placement based on live order book data.
#
# Features:
#   - Real-time Ticker & Order Book updates via WebSockets (ccxt.pro).
#   - Asynchronous REST API calls for Balance, Positions, OHLCV.
#   - Concurrent background tasks for data fetching and processing.
#   - Calculation of technical indicators (SMA, EMA, Momentum, StochRSI).
#   - Fibonacci Pivot Point calculation.
#   - Detailed order book analysis (VWAP, volume imbalance, large orders).
#   - Live position display with unrealized PnL.
#   - Interactive limit order placement using order book levels.
#   - Market order placement.
#   - Robust configuration loading from .env file.
#   - Colorized terminal output for better readability.
#   - Graceful handling of connection errors and rate limits with retries.
#   - Termux toast notifications for order confirmations/errors (if available).
#
# Requirements:
#   - Python 3.7+
#   - ccxt >= 4.0.0 (including ccxt.pro)
#   - python-dotenv
#   - colorama
#
# Usage:
#   1. Create a `.env` file with your Bybit API keys and configuration overrides.
#      (See CONFIG section for available keys).
#   2. Run the script: `python pyrmethus_market_analyzer_v3.1_async_stable.py`
#   3. Interact via the command prompt (b: buy, s: sell, r: refresh, x: exit).
#
# Disclaimer:
#   This script is for educational and informational purposes only. Trading
#   cryptocurrencies involves significant risk. Use this tool responsibly and
#   at your own risk. The author is not liable for any financial losses.
# ==============================================================================
# Version: 3.1 (Async Stable)
# Author: [Your Name/Alias Here - Optional]
# License: [Specify License - e.g., MIT - Optional]
# ==============================================================================

import asyncio
import decimal
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

import ccxt  # Standard ccxt needed for specific Exception types

# Third-party Libraries
import ccxt.pro as ccxtpro  # WebSocket and async REST support
from colorama import Back, Fore, Style, init
from dotenv import load_dotenv

# Initialize Colorama & Decimal Precision
init(autoreset=True)
decimal.getcontext().prec = 50  # Set high precision for calculations

# --- Constants ---
CONNECTING = "connecting"
OK = "ok"
ERROR = "error"
CONN_STATUS_KEYS = ["ws_ticker", "ws_ob", "rest"]

# Load environment variables from .env file
load_dotenv()


# ==============================================================================
# Configuration Loading (Robust Error Handling)
# ==============================================================================
def get_config_value(key: str, default: Any, cast_type: Callable = str) -> Any:
    """Retrieves a configuration value from environment variables.

    Args:
        key: The environment variable key.
        default: The default value to use if the key is not found or invalid.
        cast_type: The type to cast the retrieved value to (e.g., int, bool, decimal.Decimal).

    Returns:
        The configuration value, cast to the specified type, or the default value.
    """
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        if cast_type == bool:
            normalized_value = value.lower().strip()
            if normalized_value in ("true", "1", "yes", "y"):
                return True
            if normalized_value in ("false", "0", "no", "n"):
                return False
            raise ValueError(f"Invalid boolean string: '{value}'")
        elif cast_type == decimal.Decimal:
            return decimal.Decimal(value)
        else:
            return cast_type(value)
    except (ValueError, TypeError, decimal.InvalidOperation):
        return default


CONFIG = {
    # --- Core Exchange Settings ---
    "API_KEY": get_config_value("BYBIT_API_KEY", None),
    "API_SECRET": get_config_value("BYBIT_API_SECRET", None),
    "EXCHANGE_TYPE": get_config_value(
        "BYBIT_EXCHANGE_TYPE", "linear", str
    ),  # 'linear' or 'inverse' or 'spot' etc.
    "SYMBOL": get_config_value("BYBIT_SYMBOL", "BTC/USDT:USDT", str).upper(),
    "POSITION_IDX": get_config_value(
        "BYBIT_POSITION_IDX", 0, int
    ),  # 0: One-Way, 1: Hedge Buy, 2: Hedge Sell
    # --- Network & Timing ---
    "REFRESH_INTERVAL": get_config_value(
        "REFRESH_INTERVAL", 5, int
    ),  # Display refresh rate (seconds) if no input
    "CONNECT_TIMEOUT": get_config_value(
        "CONNECT_TIMEOUT", 35000, int
    ),  # ccxtpro connection timeout (ms)
    "RETRY_DELAY_NETWORK_ERROR": get_config_value(
        "RETRY_DELAY_NETWORK_ERROR", 10, int
    ),  # Base delay after network errors (s)
    "RETRY_DELAY_RATE_LIMIT": get_config_value(
        "RETRY_DELAY_RATE_LIMIT", 60, int
    ),  # Delay after rate limit errors (s)
    "BALANCE_POS_FETCH_INTERVAL": get_config_value(
        "BALANCE_POS_FETCH_INTERVAL", 45, int
    ),  # How often to fetch balance/positions (s)
    "OHLCV_FETCH_INTERVAL": get_config_value(
        "OHLCV_FETCH_INTERVAL", 300, int
    ),  # How often to fetch historical candles (s)
    # --- Order Book Configuration ---
    "ORDER_FETCH_LIMIT": get_config_value(
        "ORDER_FETCH_LIMIT", 50, int
    ),  # WebSocket order book depth subscription
    "MAX_ORDERBOOK_DEPTH_DISPLAY": get_config_value(
        "MAX_ORDERBOOK_DEPTH_DISPLAY", 30, int
    ),  # Max levels to display
    "VOLUME_THRESHOLDS": {
        "high": get_config_value(
            "VOLUME_THRESHOLD_HIGH", decimal.Decimal("10"), decimal.Decimal
        ),
        "medium": get_config_value(
            "VOLUME_THRESHOLD_MEDIUM", decimal.Decimal("2"), decimal.Decimal
        ),
    },
    # --- Indicator Configuration ---
    "INDICATOR_TIMEFRAME": get_config_value("INDICATOR_TIMEFRAME", "15m", str),
    "SMA_PERIOD": get_config_value("SMA_PERIOD", 9, int),
    "SMA2_PERIOD": get_config_value("SMA2_PERIOD", 20, int),
    "EMA1_PERIOD": get_config_value("EMA1_PERIOD", 12, int),
    "EMA2_PERIOD": get_config_value("EMA2_PERIOD", 34, int),
    "MOMENTUM_PERIOD": get_config_value("MOMENTUM_PERIOD", 10, int),
    "RSI_PERIOD": get_config_value("RSI_PERIOD", 14, int),
    "STOCH_K_PERIOD": get_config_value("STOCH_K_PERIOD", 14, int),
    "STOCH_D_PERIOD": get_config_value("STOCH_D_PERIOD", 3, int),
    "STOCH_RSI_OVERSOLD": get_config_value(
        "STOCH_RSI_OVERSOLD", decimal.Decimal("20"), decimal.Decimal
    ),
    "STOCH_RSI_OVERBOUGHT": get_config_value(
        "STOCH_RSI_OVERBOUGHT", decimal.Decimal("80"), decimal.Decimal
    ),
    "PIVOT_TIMEFRAME": get_config_value("PIVOT_TIMEFRAME", "1d", str),
    "MIN_OHLCV_CANDLES": max(  # Auto-calculate minimum required candles for indicators
        get_config_value("SMA_PERIOD", 9, int),
        get_config_value("SMA2_PERIOD", 20, int),
        get_config_value("EMA1_PERIOD", 12, int),
        get_config_value("EMA2_PERIOD", 34, int),
        get_config_value("MOMENTUM_PERIOD", 10, int)
        + 1,  # Momentum needs period + 1 prior closes
        # StochRSI needs RSI period + K period + D period - 1 (roughly, slight overestimation is safe)
        get_config_value("RSI_PERIOD", 14, int)
        + get_config_value("STOCH_K_PERIOD", 14, int)
        + get_config_value("STOCH_D_PERIOD", 3, int),
    )
    + 5,  # Add buffer for stability
    # --- Display Formatting ---
    "PNL_PRECISION": get_config_value("PNL_PRECISION", 2, int),
    "MIN_PRICE_DISPLAY_PRECISION": get_config_value(
        "MIN_PRICE_DISPLAY_PRECISION", 3, int
    ),  # Ensure price shown has at least this many decimals
    "STOCH_RSI_DISPLAY_PRECISION": get_config_value(
        "STOCH_RSI_DISPLAY_PRECISION", 2, int
    ),
    "VOLUME_DISPLAY_PRECISION": get_config_value("VOLUME_DISPLAY_PRECISION", 2, int),
    "BALANCE_DISPLAY_PRECISION": get_config_value("BALANCE_DISPLAY_PRECISION", 2, int),
    # --- Trading Configuration ---
    "FETCH_BALANCE_ASSET": get_config_value(
        "FETCH_BALANCE_ASSET", "USDT", str
    ).upper(),  # Asset to display balance for
    "DEFAULT_ORDER_TYPE": get_config_value(
        "DEFAULT_ORDER_TYPE", "limit", str
    ).lower(),  # 'limit' or 'market'
    "LIMIT_ORDER_SELECTION_TYPE": get_config_value(
        "LIMIT_ORDER_SELECTION_TYPE", "interactive", str
    ).lower(),  # 'interactive' or 'manual'
    # --- Debugging ---
    "VERBOSE_DEBUG": get_config_value("VERBOSE_DEBUG", False, bool),
}

# Fibonacci Ratios (Constant)
FIB_RATIOS = {
    "r3": decimal.Decimal("1.000"),
    "r2": decimal.Decimal("0.618"),
    "r1": decimal.Decimal("0.382"),
    "s1": decimal.Decimal("0.382"),
    "s2": decimal.Decimal("0.618"),
    "s3": decimal.Decimal("1.000"),
}

# ==============================================================================
# Shared State & Global Exchange Instance
# ==============================================================================
# This dictionary holds the latest data received from various sources.
# It's updated by the background tasks and read by the display function.
latest_data: dict[str, Any] = {
    "ticker": None,  # Latest ticker info from WebSocket
    "orderbook": None,  # Latest analyzed order book from WebSocket
    "balance": None,  # Latest fetched account balance for FETCH_BALANCE_ASSET
    "positions": [],  # List of open positions for the SYMBOL
    "indicator_ohlcv": None,  # List of OHLCV candles for indicators
    "pivot_ohlcv": None,  # List of OHLCV candles for pivot points
    "indicators": {},  # Calculated indicator values {name: {value: V, error: E}}
    "pivots": None,  # Calculated pivot point levels
    "market_info": None,  # Market details (precision, limits) fetched once
    "last_update_times": {},  # Timestamps of last successful updates for data types
    "connection_status": dict.fromkeys(
        CONN_STATUS_KEYS, "init"
    ),  # Status of connections ('init', 'connecting', 'ok', 'error')
}
exchange: ccxtpro.Exchange | None = (
    None  # Global exchange instance (initialized in main_async)
)


# ==============================================================================
# Utility Functions
# ==============================================================================
def print_color(
    text: str,
    color: str = Fore.WHITE,
    style: str = Style.NORMAL,
    end: str = "\n",
    **kwargs: Any,
) -> None:
    """Prints colorized text using Colorama."""


def verbose_print(text: str, color: str = Fore.CYAN, style: str = Style.DIM) -> None:
    """Prints debug messages only if VERBOSE_DEBUG is enabled in CONFIG."""
    if CONFIG.get("VERBOSE_DEBUG", False):
        print_color(f"# DEBUG: {text}", color=color, style=style)


def termux_toast(message: str, duration: str = "short") -> None:
    """Displays a Termux toast notification if termux-toast command is available.
    Sanitizes the message to prevent command injection vulnerabilities.

    Args:
        message: The message string to display.
        duration: Toast duration ('short' or 'long').
    """
    try:
        # Basic sanitization: allow alphanumeric and common punctuation. Limit length.
        safe_message = "".join(
            c
            for c in str(message)[:150]
            if c.isalnum() or c in " .,!?-:+%$/=()[]{}<>@*_"
        )
        # Use list form of command to avoid shell interpretation issues
        command = ["termux-toast", "-d", duration, safe_message]
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            verbose_print(
                f"Toast command failed (Code {result.returncode}): {result.stderr.strip()}",
                Fore.YELLOW,
            )
    except FileNotFoundError:
        pass  # termux-toast not installed, ignore silently
    except subprocess.TimeoutExpired:
        verbose_print("Toast command timed out.", Fore.YELLOW)
    except Exception as e:
        verbose_print(f"Toast error: {e}", Fore.YELLOW)


def format_decimal(
    value: decimal.Decimal | str | int | float | None,
    native_precision: int,
    min_display_precision: int | None = None,
) -> str:
    """Formats a numeric value as a string using Decimal for accurate rounding.

    Args:
        value: The numeric value (can be None, Decimal, str, int, float).
        native_precision: The number of decimal places inherent to the data (e.g., market price precision).
        min_display_precision: The minimum number of decimal places to show, padding with zeros if needed.
                                If None, uses native_precision.

    Returns:
        A formatted string representation of the number, or "N/A" if input is None.
        Returns original string representation on formatting error.
    """
    if value is None:
        return "N/A"
    try:
        # Convert input to Decimal for reliable arithmetic
        d_value = (
            decimal.Decimal(str(value))
            if not isinstance(value, decimal.Decimal)
            else value
        )

        # Determine the target precision for display
        display_precision = max(int(native_precision), 0)
        if min_display_precision is not None:
            display_precision = max(
                display_precision, max(int(min_display_precision), 0)
            )

        # Create a quantizer for rounding
        quantizer = decimal.Decimal("1") / (decimal.Decimal("10") ** display_precision)
        rounded_value = d_value.quantize(quantizer, rounding=decimal.ROUND_HALF_UP)

        # Format as fixed-point string and ensure correct decimal places
        formatted_str = f"{rounded_value:f}"
        if "." in formatted_str:
            integer_part, decimal_part = formatted_str.split(".")
            # Trim or pad decimal part to the exact display_precision
            decimal_part = decimal_part[:display_precision].ljust(
                display_precision, "0"
            )
            formatted_str = (
                f"{integer_part}.{decimal_part}"
                if display_precision > 0
                else integer_part
            )
        elif display_precision > 0:
            # Add decimal point and zeros if needed
            formatted_str += "." + "0" * display_precision

        return formatted_str
    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        verbose_print(f"FormatDecimal Error processing '{value}': {e}")
        return str(value)  # Return original string representation on error


# ==============================================================================
# Async Market Info Fetcher
# ==============================================================================
async def get_market_info(
    exchange_instance: ccxtpro.Exchange, symbol: str
) -> dict[str, Any] | None:
    """Asynchronously fetches and returns market information (precision, limits) for a symbol.
    Ensures markets are loaded if necessary.

    Args:
        exchange_instance: An initialized ccxt.pro exchange instance.
        symbol: The market symbol (e.g., 'BTCUSDT').

    Returns:
        A dictionary containing market details, or None if the symbol is invalid
        or an error occurs.
    """
    attempt = 0
    max_attempts = 2
    while attempt < max_attempts:
        attempt += 1
        try:
            print_color(
                f"{Fore.CYAN}# Querying market runes for {symbol} (async, attempt {attempt})...",
                style=Style.DIM,
                end="\r",
            )

            # Ensure markets are loaded. Use `load_markets(True)` for a fresh fetch if needed.
            if not exchange_instance.markets or symbol not in exchange_instance.markets:
                verbose_print(f"Market list needs loading/refresh for {symbol}.")
                # load_markets can be I/O bound, so await it.
                await exchange_instance.load_markets(True)  # Force reload

            sys.stdout.write("\033[K")  # Clear the line after loading

            # Check again after loading
            if symbol not in exchange_instance.markets:
                print_color(
                    f"Symbol '{symbol}' still not found after async market reload.",
                    color=Fore.RED,
                    style=Style.BRIGHT,
                )
                # No need to retry immediately if reload failed, likely symbol is wrong
                return None

            market = exchange_instance.market(symbol)
            verbose_print(f"Async market info retrieved for {symbol}")

            # --- Extract Precision & Limits ---
            price_prec, amount_prec = 8, 8  # Sensible defaults
            min_amount = decimal.Decimal("0")
            price_tick_size, amount_step = (
                decimal.Decimal("1e-8"),
                decimal.Decimal("1e-8"),
            )  # Defaults

            try:
                price_prec = int(
                    market.get("precision", {}).get("price", 8)
                )  # Newer ccxt might provide integer precision directly
                if not isinstance(
                    price_prec, int
                ):  # Fallback for older ccxt or string format
                    price_prec = int(
                        decimal.Decimal(
                            str(market.get("precision", {}).get("price", "1e-8"))
                        ).log10()
                        * -1
                    )
            except (ValueError, TypeError, decimal.InvalidOperation, AttributeError):
                verbose_print(
                    f"Could not parse price precision for {symbol}, using default {price_prec}"
                )

            try:
                amount_prec = int(market.get("precision", {}).get("amount", 8))
                if not isinstance(amount_prec, int):
                    amount_prec = int(
                        decimal.Decimal(
                            str(market.get("precision", {}).get("amount", "1e-8"))
                        ).log10()
                        * -1
                    )
            except (ValueError, TypeError, decimal.InvalidOperation, AttributeError):
                verbose_print(
                    f"Could not parse amount precision for {symbol}, using default {amount_prec}"
                )

            try:
                min_amount = decimal.Decimal(
                    str(market.get("limits", {}).get("amount", {}).get("min", "0"))
                )
            except (ValueError, TypeError, decimal.InvalidOperation, AttributeError):
                verbose_print(
                    f"Could not parse min amount for {symbol}, using default {min_amount}"
                )

            # Calculate step sizes based on precision
            price_tick_size = (
                decimal.Decimal("1") / (decimal.Decimal("10") ** price_prec)
                if price_prec >= 0
                else decimal.Decimal("1")
            )
            amount_step = (
                decimal.Decimal("1") / (decimal.Decimal("10") ** amount_prec)
                if amount_prec >= 0
                else decimal.Decimal("1")
            )

            # Sometimes exchanges provide tick sizes directly, prefer those if available
            market_price_tick = market.get("precision", {}).get("price")
            if market_price_tick:
                try:
                    price_tick_size = decimal.Decimal(str(market_price_tick))
                except:
                    pass  # Stick to calculated if direct fails

            market_amount_step = market.get("precision", {}).get("amount")
            if market_amount_step:
                try:
                    amount_step = decimal.Decimal(str(market_amount_step))
                except:
                    pass  # Stick to calculated if direct fails

            return {
                "symbol": symbol,
                "price_precision": price_prec,
                "amount_precision": amount_prec,
                "min_amount": min_amount,
                "price_tick_size": price_tick_size,
                "amount_step": amount_step,
            }

        except ccxt.BadSymbol:
            sys.stdout.write("\033[K")
            print_color(
                f"Symbol '{symbol}' is invalid or not found on the exchange.",
                color=Fore.RED,
                style=Style.BRIGHT,
            )
            return None  # Don't retry if symbol is definitively bad
        except (TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
            sys.stdout.write("\033[K")
            print_color(
                f"Network error getting market info (async): {e}", color=Fore.YELLOW
            )
            if attempt >= max_attempts:
                print_color("Max attempts reached for market info.", color=Fore.RED)
                return None
            await asyncio.sleep(
                CONFIG["RETRY_DELAY_NETWORK_ERROR"] * attempt
            )  # Exponential backoff
        except ccxt.ExchangeNotAvailable as e:
            sys.stdout.write("\033[K")
            print_color(f"Exchange not available: {e}", color=Fore.RED)
            return None  # Don't retry if exchange is down
        except Exception as e:
            sys.stdout.write("\033[K")
            print_color(
                f"Critical error getting market info (async): {e}", color=Fore.RED
            )
            traceback.print_exc()
            return None  # Don't retry on unexpected critical errors

    return None  # Should be unreachable if max_attempts > 0


# ==============================================================================
# Indicator Calculation Functions (Using Decimal)
# ==============================================================================
def calculate_sma(
    data: list[str | float | int | decimal.Decimal], period: int
) -> decimal.Decimal | None:
    """Calculates the Simple Moving Average (SMA)."""
    if not data or len(data) < period:
        return None
    try:
        relevant_data = [decimal.Decimal(str(p)) for p in data[-period:]]
        return sum(relevant_data) / decimal.Decimal(period)
    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        verbose_print(f"SMA Calc Error (period {period}): {e}")
        return None


def calculate_ema(
    data: list[str | float | int | decimal.Decimal], period: int
) -> list[decimal.Decimal] | None:
    """Calculates the Exponential Moving Average (EMA) series."""
    if not data or len(data) < period:
        return None
    try:
        decimal_data = [decimal.Decimal(str(p)) for p in data]
        ema_values: list[decimal.Decimal | None] = [None] * len(data)
        multiplier = decimal.Decimal(2) / (decimal.Decimal(period) + 1)

        # Initial SMA for the first EMA value
        sma_init = sum(decimal_data[:period]) / decimal.Decimal(period)
        ema_values[period - 1] = sma_init

        # Calculate subsequent EMA values
        for i in range(period, len(data)):
            prev_ema = ema_values[i - 1]
            if prev_ema is None:
                continue  # Should not happen after initialization
            current_price = decimal_data[i]
            ema_values[i] = ((current_price - prev_ema) * multiplier) + prev_ema

        # Return only the valid (non-None) EMA values
        return [ema for ema in ema_values if ema is not None]
    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        verbose_print(f"EMA Calc Error (period {period}): {e}")
        return None


def calculate_momentum(
    data: list[str | float | int | decimal.Decimal], period: int
) -> decimal.Decimal | None:
    """Calculates the Momentum indicator."""
    if not data or len(data) <= period:
        return None  # Needs current and value 'period' steps ago
    try:
        current_price = decimal.Decimal(str(data[-1]))
        prior_price = decimal.Decimal(str(data[-period - 1]))
        return current_price - prior_price
    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        verbose_print(f"Momentum Calc Error (period {period}): {e}")
        return None


def calculate_fib_pivots(
    high: Any, low: Any, close: Any
) -> dict[str, decimal.Decimal] | None:
    """Calculates Fibonacci Pivot Points for the *next* period."""
    if None in [high, low, close]:
        return None
    try:
        h = decimal.Decimal(str(high))
        l = decimal.Decimal(str(low))
        c = decimal.Decimal(str(close))
        if h <= 0 or l <= 0 or c <= 0 or h < l:
            verbose_print(f"Invalid HLC for Pivot: H={h}, L={l}, C={c}")
            return None

        pp = (h + l + c) / decimal.Decimal(3)
        range_hl = h - l  # Range is always positive as h >= l

        return {
            "R3": pp + (range_hl * FIB_RATIOS["r3"]),
            "R2": pp + (range_hl * FIB_RATIOS["r2"]),
            "R1": pp + (range_hl * FIB_RATIOS["r1"]),
            "PP": pp,
            "S1": pp - (range_hl * FIB_RATIOS["s1"]),
            "S2": pp - (range_hl * FIB_RATIOS["s2"]),
            "S3": pp - (range_hl * FIB_RATIOS["s3"]),
        }
    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        verbose_print(f"Pivot Calc Error: {e}")
        return None


def calculate_rsi_manual(
    close_prices_list: list[Any], period: int = 14
) -> tuple[list[decimal.Decimal] | None, str | None]:
    """Calculates Relative Strength Index (RSI) manually using Wilder's smoothing."""
    if not close_prices_list or len(close_prices_list) <= period:
        return (
            None,
            f"Data short ({len(close_prices_list) if close_prices_list else 0} < {period + 1})",
        )
    try:
        prices = [decimal.Decimal(str(p)) for p in close_prices_list]
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        if not deltas:
            return None, "No price changes"

        gains = [d if d > 0 else decimal.Decimal("0") for d in deltas]
        losses = [-d if d < 0 else decimal.Decimal("0") for d in deltas]

        if len(gains) < period:
            return None, f"Deltas short ({len(gains)} < {period})"

        # Initial average gain/loss
        avg_gain = sum(gains[:period]) / decimal.Decimal(period)
        avg_loss = sum(losses[:period]) / decimal.Decimal(period)

        rsi_values = []
        # Calculate initial RSI
        if avg_loss == 0:
            rsi = decimal.Decimal("100")  # Avoid division by zero if no losses
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)

        # Calculate subsequent RSI using Wilder's smoothing
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / decimal.Decimal(period)
            avg_loss = (avg_loss * (period - 1) + losses[i]) / decimal.Decimal(period)
            if avg_loss == 0:
                rsi = decimal.Decimal("100")
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)

        return rsi_values, None  # Success
    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        verbose_print(f"RSI Calc Error (period {period}): {e}")
        return None, str(e)


def calculate_stoch_rsi_manual(
    rsi_values: list[decimal.Decimal], k_period: int = 14, d_period: int = 3
) -> tuple[decimal.Decimal | None, decimal.Decimal | None, str | None]:
    """Calculates Stochastic RSI (%K and %D) from a list of RSI values."""
    if not rsi_values or len(rsi_values) < k_period:
        return (
            None,
            None,
            f"RSI short ({len(rsi_values) if rsi_values else 0} < {k_period})",
        )

    try:
        # Ensure we only use valid, finite RSI values
        valid_rsi = [r for r in rsi_values if r is not None and r.is_finite()]
        if len(valid_rsi) < k_period:
            return None, None, f"Valid RSI short ({len(valid_rsi)} < {k_period})"

        stoch_k_values = []
        for i in range(k_period - 1, len(valid_rsi)):
            window = valid_rsi[i - k_period + 1 : i + 1]
            current_rsi = window[-1]
            min_rsi_in_window = min(window)
            max_rsi_in_window = max(window)

            if max_rsi_in_window == min_rsi_in_window:
                # Handle division by zero: if range is 0, stoch is often set to 50 or 100/0 depending on convention.
                # Let's use 50 as a neutral value. Check formula source if specific behavior is needed.
                stoch_k = decimal.Decimal("50")
            else:
                stoch_k = (
                    (current_rsi - min_rsi_in_window)
                    / (max_rsi_in_window - min_rsi_in_window)
                ) * 100
                # Clamp between 0 and 100
                stoch_k = max(
                    decimal.Decimal("0"), min(decimal.Decimal("100"), stoch_k)
                )
            stoch_k_values.append(stoch_k)

        if not stoch_k_values:
            return None, None, "%K list empty after calculation"

        latest_k = stoch_k_values[-1]
        latest_d = None

        # Calculate %D (SMA of %K) if enough %K values exist
        if len(stoch_k_values) >= d_period:
            latest_d = sum(stoch_k_values[-d_period:]) / decimal.Decimal(d_period)

        return latest_k, latest_d, None  # Success
    except (ValueError, TypeError, decimal.InvalidOperation, IndexError) as e:
        verbose_print(f"StochRSI Calc Error (K={k_period}, D={d_period}): {e}")
        return None, None, str(e)


# ==============================================================================
# Data Processing & Analysis (Order Book)
# ==============================================================================
def analyze_orderbook_data(
    orderbook: dict, market_info: dict, config: dict
) -> dict | None:
    """Analyzes raw order book data to calculate VWAP, totals, and format levels.

    Args:
        orderbook: The raw order book dictionary from ccxt.pro watch_order_book.
        market_info: Dictionary containing market precision details.
        config: The main configuration dictionary.

    Returns:
        An analyzed order book dictionary with calculated metrics and formatted
        levels, or None if the input is invalid.
    """
    if (
        not orderbook
        or not isinstance(orderbook.get("bids"), list)
        or not isinstance(orderbook.get("asks"), list)
    ):
        verbose_print("Invalid orderbook structure received.")
        return None

    market_info["price_precision"]
    amount_prec = market_info["amount_precision"]
    vol_disp_prec = config["VOLUME_DISPLAY_PRECISION"]
    vol_thr = config["VOLUME_THRESHOLDS"]
    display_depth = config["MAX_ORDERBOOK_DEPTH_DISPLAY"]

    analyzed_ob = {
        "symbol": orderbook.get("symbol", market_info.get("symbol", "N/A")),
        "timestamp": orderbook.get(
            "datetime",
            time.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ",
                time.gmtime(orderbook.get("timestamp", time.time() * 1000) / 1000),
            ),
        ),
        "asks": [],  # Formatted ask levels for display
        "bids": [],  # Formatted bid levels for display
        "ask_total_volume_fetched": decimal.Decimal(
            "0"
        ),  # Total volume in the fetched depth
        "bid_total_volume_fetched": decimal.Decimal(
            "0"
        ),  # Total volume in the fetched depth
        "ask_vwap_fetched": decimal.Decimal(
            "0"
        ),  # Volume Weighted Avg Price of fetched asks
        "bid_vwap_fetched": decimal.Decimal(
            "0"
        ),  # Volume Weighted Avg Price of fetched bids
        "volume_imbalance_ratio_fetched": decimal.Decimal(
            "0"
        ),  # Bid Volume / Ask Volume
        "cumulative_ask_volume_displayed": decimal.Decimal(
            "0"
        ),  # Cumulative volume within display_depth
        "cumulative_bid_volume_displayed": decimal.Decimal(
            "0"
        ),  # Cumulative volume within display_depth
    }

    # Process Asks
    ask_weighted_sum = decimal.Decimal("0")
    cumulative_ask_volume = decimal.Decimal("0")
    for i, level in enumerate(orderbook["asks"]):
        try:
            price = decimal.Decimal(str(level[0]))
            volume = decimal.Decimal(str(level[1]))
        except (IndexError, ValueError, TypeError, decimal.InvalidOperation):
            verbose_print(f"Skipping invalid ask level: {level}")
            continue

        analyzed_ob["ask_total_volume_fetched"] += volume
        ask_weighted_sum += price * volume

        if i < display_depth:
            cumulative_ask_volume += volume
            vol_str = format_decimal(volume, amount_prec, vol_disp_prec)
            color, style = (
                (Fore.LIGHTRED_EX, Style.BRIGHT)
                if volume >= vol_thr["high"]
                else (Fore.RED, Style.NORMAL)
                if volume >= vol_thr["medium"]
                else (Fore.WHITE, Style.DIM)
            )  # Dim normal volume
            analyzed_ob["asks"].append(
                {
                    "price": price,
                    "volume": volume,
                    "volume_str": vol_str,
                    "color": color,
                    "style": style,
                    "cumulative_volume": format_decimal(
                        cumulative_ask_volume, amount_prec, vol_disp_prec
                    ),
                }
            )
    analyzed_ob["cumulative_ask_volume_displayed"] = cumulative_ask_volume

    # Process Bids
    bid_weighted_sum = decimal.Decimal("0")
    cumulative_bid_volume = decimal.Decimal("0")
    for i, level in enumerate(orderbook["bids"]):
        try:
            price = decimal.Decimal(str(level[0]))
            volume = decimal.Decimal(str(level[1]))
        except (IndexError, ValueError, TypeError, decimal.InvalidOperation):
            verbose_print(f"Skipping invalid bid level: {level}")
            continue

        analyzed_ob["bid_total_volume_fetched"] += volume
        bid_weighted_sum += price * volume

        if i < display_depth:
            cumulative_bid_volume += volume
            vol_str = format_decimal(volume, amount_prec, vol_disp_prec)
            color, style = (
                (Fore.LIGHTGREEN_EX, Style.BRIGHT)
                if volume >= vol_thr["high"]
                else (Fore.GREEN, Style.NORMAL)
                if volume >= vol_thr["medium"]
                else (Fore.WHITE, Style.DIM)
            )  # Dim normal volume
            analyzed_ob["bids"].append(
                {
                    "price": price,
                    "volume": volume,
                    "volume_str": vol_str,
                    "color": color,
                    "style": style,
                    "cumulative_volume": format_decimal(
                        cumulative_bid_volume, amount_prec, vol_disp_prec
                    ),
                }
            )
    analyzed_ob["cumulative_bid_volume_displayed"] = cumulative_bid_volume

    # Calculate VWAP and Imbalance
    ask_tot = analyzed_ob["ask_total_volume_fetched"]
    bid_tot = analyzed_ob["bid_total_volume_fetched"]

    if ask_tot > 0:
        analyzed_ob["ask_vwap_fetched"] = ask_weighted_sum / ask_tot
        if bid_tot > 0:
            analyzed_ob["volume_imbalance_ratio_fetched"] = bid_tot / ask_tot
        else:  # Bids are zero, asks non-zero
            analyzed_ob["volume_imbalance_ratio_fetched"] = decimal.Decimal("0")
    elif bid_tot > 0:  # Asks are zero, bids non-zero
        analyzed_ob["volume_imbalance_ratio_fetched"] = decimal.Decimal(
            "inf"
        )  # Infinite imbalance towards bids
    else:  # Both zero
        analyzed_ob["volume_imbalance_ratio_fetched"] = decimal.Decimal(
            "1"
        )  # Or 0 or NaN? Let's use 1 for neutral.

    if bid_tot > 0:
        analyzed_ob["bid_vwap_fetched"] = bid_weighted_sum / bid_tot

    return analyzed_ob


# ==============================================================================
# WebSocket Watcher Tasks (Run Continuously)
# ==============================================================================
async def watch_ticker(exchange_pro: ccxtpro.Exchange, symbol: str) -> None:
    """Watches the ticker stream via WebSocket and updates shared state."""
    verbose_print(f"Starting ticker watcher for {symbol}...")
    latest_data["connection_status"]["ws_ticker"] = CONNECTING
    while True:
        try:
            ticker = await exchange_pro.watch_ticker(symbol)
            latest_data["ticker"] = ticker
            latest_data["last_update_times"]["ticker"] = time.time()
            if latest_data["connection_status"]["ws_ticker"] != OK:
                latest_data["connection_status"]["ws_ticker"] = OK
                verbose_print(f"Ticker WS connected successfully for {symbol}.")
            # Minimal log on successful update:
            # verbose_print(f"Ticker update: {ticker.get('last')}", style=Style.DIM)
        except (TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
            print_color(f"# Ticker WS Network Error: {e}", Fore.YELLOW)
            latest_data["connection_status"]["ws_ticker"] = ERROR
            await asyncio.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"])
        except ccxt.ExchangeError as e:  # Includes RateLimitExceeded
            print_color(f"# Ticker WS Exchange Error: {e}", Fore.RED)
            latest_data["connection_status"]["ws_ticker"] = ERROR
            delay = (
                CONFIG["RETRY_DELAY_RATE_LIMIT"]
                if isinstance(e, ccxt.RateLimitExceeded)
                else 30
            )
            await asyncio.sleep(delay)
        except Exception as e:
            print_color(
                f"# Ticker WS Unexpected Error: {e}", Fore.RED, style=Style.BRIGHT
            )
            latest_data["connection_status"]["ws_ticker"] = ERROR
            traceback.print_exc()
            await asyncio.sleep(45)  # Longer delay for unknown errors


async def watch_orderbook(exchange_pro: ccxtpro.Exchange, symbol: str) -> None:
    """Watches the order book stream via WebSocket, analyzes, and updates shared state."""
    verbose_print(f"Starting order book watcher for {symbol}...")
    latest_data["connection_status"]["ws_ob"] = CONNECTING
    while True:
        try:
            orderbook = await exchange_pro.watch_order_book(
                symbol, limit=CONFIG["ORDER_FETCH_LIMIT"]
            )
            market_info = latest_data.get("market_info")

            if market_info:
                analyzed_ob = analyze_orderbook_data(orderbook, market_info, CONFIG)
                if analyzed_ob:  # Only update if analysis was successful
                    latest_data["orderbook"] = analyzed_ob
                    latest_data["last_update_times"]["orderbook"] = time.time()
                    if latest_data["connection_status"]["ws_ob"] != OK:
                        latest_data["connection_status"]["ws_ob"] = OK
                        verbose_print(
                            f"Order Book WS connected successfully for {symbol}."
                        )
                    # Minimal log:
                    # verbose_print(f"OB update: A:{analyzed_ob['asks'][0]['price'] if analyzed_ob['asks'] else 'N/A'} B:{analyzed_ob['bids'][0]['price'] if analyzed_ob['bids'] else 'N/A'}", style=Style.DIM)
                else:
                    # Analysis failed, maybe log this?
                    verbose_print("Order book analysis failed.", color=Fore.YELLOW)
            else:
                # Market info not ready yet, wait briefly and retry implicitly
                verbose_print(
                    "Order book watcher waiting for market_info...",
                    color=Fore.YELLOW,
                    style=Style.DIM,
                )
                await asyncio.sleep(1)

        except (TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
            print_color(f"# Order Book WS Network Error: {e}", Fore.YELLOW)
            latest_data["connection_status"]["ws_ob"] = ERROR
            await asyncio.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"])
        except ccxt.ExchangeError as e:
            print_color(f"# Order Book WS Exchange Error: {e}", Fore.RED)
            latest_data["connection_status"]["ws_ob"] = ERROR
            delay = (
                CONFIG["RETRY_DELAY_RATE_LIMIT"]
                if isinstance(e, ccxt.RateLimitExceeded)
                else 30
            )
            await asyncio.sleep(delay)
        except Exception as e:
            print_color(
                f"# Order Book WS Unexpected Error: {e}", Fore.RED, style=Style.BRIGHT
            )
            latest_data["connection_status"]["ws_ob"] = ERROR
            traceback.print_exc()
            await asyncio.sleep(45)


# ==============================================================================
# Periodic Data Fetching Task (REST via ccxt.pro async methods)
# ==============================================================================
async def fetch_periodic_data(exchange_pro: ccxtpro.Exchange, symbol: str) -> None:
    """Periodically fetches balance, positions, and OHLCV using async REST calls."""
    bal_pos_interval = max(
        10, CONFIG["BALANCE_POS_FETCH_INTERVAL"]
    )  # Ensure minimum interval
    ohlcv_interval = max(60, CONFIG["OHLCV_FETCH_INTERVAL"])  # Ensure minimum interval
    min_ohlcv_needed = CONFIG["MIN_OHLCV_CANDLES"]
    ind_tf, piv_tf = CONFIG["INDICATOR_TIMEFRAME"], CONFIG["PIVOT_TIMEFRAME"]

    last_bal_pos_fetch_time = 0
    last_ohlcv_fetch_time = 0

    # Ensure market_info is available before starting periodic fetches that depend on it
    while not latest_data.get("market_info"):
        verbose_print(
            "Periodic fetcher waiting for market_info...",
            color=Fore.YELLOW,
            style=Style.DIM,
        )
        await asyncio.sleep(2)

    verbose_print(
        f"Starting periodic data fetcher (Bal/Pos ~{bal_pos_interval}s, OHLCV ~{ohlcv_interval}s)..."
    )

    while True:
        now = time.time()
        tasks_to_run = {}
        fetch_bal_pos = (now - last_bal_pos_fetch_time) >= bal_pos_interval
        fetch_ohlcv = (now - last_ohlcv_fetch_time) >= ohlcv_interval

        # Schedule Balance/Position Fetch
        if fetch_bal_pos:
            tasks_to_run["balance"] = exchange_pro.fetch_balance()
            # Fetch positions only for the target symbol for efficiency
            tasks_to_run["positions"] = exchange_pro.fetch_positions([symbol])

        # Schedule OHLCV Fetch
        if fetch_ohlcv:
            # Fetch slightly more history than strictly needed for indicators
            history_needed = min_ohlcv_needed + 5
            tasks_to_run["indicator_ohlcv"] = exchange_pro.fetch_ohlcv(
                symbol, ind_tf, limit=history_needed
            )
            # Fetch last 2 candles for pivot calculations (current incomplete, previous complete)
            tasks_to_run["pivot_ohlcv"] = exchange_pro.fetch_ohlcv(
                symbol, piv_tf, limit=2
            )

        if not tasks_to_run:
            # Nothing to fetch this cycle, sleep until next potential fetch time
            next_bal_pos_time = last_bal_pos_fetch_time + bal_pos_interval
            next_ohlcv_time = last_ohlcv_fetch_time + ohlcv_interval
            await asyncio.sleep(max(0.5, min(next_bal_pos_time, next_ohlcv_time) - now))
            continue

        verbose_print(f"Running periodic fetches: {list(tasks_to_run.keys())}")
        results = {}
        fetch_success = True
        try:
            # Use gather to run fetches concurrently
            results_list = await asyncio.gather(
                *tasks_to_run.values(), return_exceptions=True
            )
            results = dict(
                zip(tasks_to_run.keys(), results_list, strict=False)
            )  # Map results back to keys
            latest_data["connection_status"]["rest"] = (
                OK  # Assume OK unless an exception below overrides
            )

            # --- Process Results Carefully (Check for Exceptions) ---
            if "balance" in results:
                bal_res = results["balance"]
                if isinstance(bal_res, dict):
                    balance_asset = CONFIG["FETCH_BALANCE_ASSET"]
                    # Access balance safely using .get()
                    asset_balance = bal_res.get("total", {}).get(balance_asset)
                    if asset_balance is not None:
                        latest_data["balance"] = decimal.Decimal(str(asset_balance))
                        latest_data["last_update_times"]["balance"] = time.time()
                    else:
                        print_color(
                            f"# Balance Warning: Asset '{balance_asset}' not found in total balance.",
                            Fore.YELLOW,
                        )
                        latest_data["balance"] = (
                            None  # Mark as unavailable if asset not found
                        )
                    last_bal_pos_fetch_time = (
                        time.time()
                    )  # Update time even if asset wasn't found, fetch itself succeeded
                elif isinstance(bal_res, Exception):
                    print_color(f"# Error fetching balance: {bal_res}", Fore.YELLOW)
                    fetch_success = False
                # else: unexpected type, ignore

            if "positions" in results:
                pos_res = results["positions"]
                if isinstance(pos_res, list):
                    # Filter for the specific symbol and non-zero contracts, handle potential Decimal conversion error
                    valid_positions = []
                    for p in pos_res:
                        try:
                            contracts_str = p.get("contracts")
                            if (
                                p.get("symbol") == symbol
                                and contracts_str
                                and decimal.Decimal(str(contracts_str)) != 0
                            ):
                                valid_positions.append(p)
                        except (decimal.InvalidOperation, TypeError, KeyError):
                            verbose_print(
                                f"Skipping position with invalid contract amount: {p.get('contracts')}"
                            )
                    latest_data["positions"] = valid_positions
                    latest_data["last_update_times"]["positions"] = time.time()
                    if "balance" not in results or isinstance(
                        results.get("balance"), Exception
                    ):
                        last_bal_pos_fetch_time = (
                            time.time()
                        )  # Update time if balance failed but pos succeeded
                elif isinstance(pos_res, Exception):
                    print_color(f"# Error fetching positions: {pos_res}", Fore.YELLOW)
                    fetch_success = False
                # else: unexpected type, ignore

            # --- Process OHLCV Results ---
            if fetch_ohlcv:
                last_ohlcv_fetch_time = (
                    time.time()
                )  # Update fetch time regardless of outcome
                ind_res = results.get("indicator_ohlcv")
                if isinstance(ind_res, list):
                    if len(ind_res) >= min_ohlcv_needed:
                        latest_data["indicator_ohlcv"] = ind_res
                        latest_data["last_update_times"]["indicator_ohlcv"] = (
                            time.time()
                        )
                        verbose_print(
                            f"Indicator OHLCV updated ({len(ind_res)} candles, tf={ind_tf})"
                        )
                        await calculate_and_store_indicators()  # Trigger recalculation
                    else:
                        print_color(
                            f"# Warning: Insufficient Indicator OHLCV received ({len(ind_res)} < {min_ohlcv_needed}) for {ind_tf}",
                            Fore.YELLOW,
                        )
                        latest_data["indicator_ohlcv"] = (
                            None  # Clear old data if fetch is insufficient
                        )
                        await calculate_and_store_indicators()  # Clear indicators too
                elif isinstance(ind_res, Exception):
                    print_color(
                        f"# Error fetching indicator OHLCV ({ind_tf}): {ind_res}",
                        Fore.YELLOW,
                    )
                    fetch_success = False
                    latest_data["indicator_ohlcv"] = (
                        None  # Clear potentially stale data on error
                    )
                    await calculate_and_store_indicators()  # Clear indicators too
                # else: unexpected type, ignore

                piv_res = results.get("pivot_ohlcv")
                if isinstance(piv_res, list) and len(piv_res) > 0:
                    latest_data["pivot_ohlcv"] = piv_res
                    latest_data["last_update_times"]["pivot_ohlcv"] = time.time()
                    verbose_print(
                        f"Pivot OHLCV updated ({len(piv_res)} candles, tf={piv_tf})"
                    )
                    await calculate_and_store_pivots()  # Trigger recalculation
                elif isinstance(piv_res, Exception):
                    print_color(
                        f"# Error fetching pivot OHLCV ({piv_tf}): {piv_res}",
                        Fore.YELLOW,
                    )
                    fetch_success = False
                    latest_data["pivots"] = (
                        None  # Clear potentially stale data on error
                    )
                elif piv_res is not None:  # Fetched successfully, but list was empty
                    latest_data["pivot_ohlcv"] = piv_res  # Store empty list
                    latest_data["pivots"] = None  # Clear calculated pivots
                    verbose_print(
                        f"Pivot OHLCV fetch returned empty list for {piv_tf}."
                    )
                # else: unexpected type (like None), ignore

            if not fetch_success:
                latest_data["connection_status"]["rest"] = ERROR

        except Exception as e:  # Catch errors during the gather or processing phase
            print_color(
                f"# Critical Error in periodic data fetch/process: {e}", Fore.RED
            )
            latest_data["connection_status"]["rest"] = ERROR
            traceback.print_exc()

        # Sleep until the *next* balance/position interval is due.
        # This simplifies the logic compared to calculating based on multiple intervals.
        await asyncio.sleep(bal_pos_interval)


# ==============================================================================
# Indicator & Pivot Calculation Tasks (Run after data fetch)
# ==============================================================================
async def calculate_and_store_indicators() -> None:
    """Calculates all configured indicators based on data in `latest_data`."""
    verbose_print("Calculating indicators...")
    ohlcv = latest_data.get("indicator_ohlcv")
    min_candles_needed = CONFIG["MIN_OHLCV_CANDLES"]
    indicators: dict[str, dict] = {  # Initialize structure
        "sma1": {"value": None, "error": None},
        "sma2": {"value": None, "error": None},
        "ema1": {"value": None, "error": None},
        "ema2": {"value": None, "error": None},
        "momentum": {"value": None, "error": None},
        "stoch_rsi": {"k": None, "d": None, "error": None},
    }
    error_msg = None

    if not ohlcv or not isinstance(ohlcv, list):
        error_msg = "OHLCV data missing"
    elif len(ohlcv) < min_candles_needed:
        error_msg = f"OHLCV short ({len(ohlcv)} < {min_candles_needed})"

    if error_msg:
        for k in indicators:
            indicators[k]["error"] = error_msg
        latest_data["indicators"] = indicators
        verbose_print(f"Indicator calculation skipped: {error_msg}", color=Fore.YELLOW)
        return

    try:
        # Extract close prices (assuming OHLCV format: [timestamp, open, high, low, close, volume])
        close_prices = [c[4] for c in ohlcv if isinstance(c, list) and len(c) >= 5]

        if len(close_prices) < min_candles_needed:  # Double check after extraction
            raise ValueError(
                f"Close price extraction failed or insufficient ({len(close_prices)} < {min_candles_needed})"
            )

        # --- Calculate Indicators ---
        indicators["sma1"]["value"] = calculate_sma(close_prices, CONFIG["SMA_PERIOD"])
        indicators["sma2"]["value"] = calculate_sma(close_prices, CONFIG["SMA2_PERIOD"])

        ema1_series = calculate_ema(close_prices, CONFIG["EMA1_PERIOD"])
        indicators["ema1"]["value"] = ema1_series[-1] if ema1_series else None
        ema2_series = calculate_ema(close_prices, CONFIG["EMA2_PERIOD"])
        indicators["ema2"]["value"] = ema2_series[-1] if ema2_series else None

        indicators["momentum"]["value"] = calculate_momentum(
            close_prices, CONFIG["MOMENTUM_PERIOD"]
        )

        # Mark calculation errors if function returned None
        for k in ["sma1", "sma2", "ema1", "ema2", "momentum"]:
            if (
                indicators[k]["value"] is None and not indicators[k]["error"]
            ):  # Avoid overwriting initial error
                indicators[k]["error"] = "Calc Fail"

        # --- Stochastic RSI Chain ---
        rsi_list, rsi_err = calculate_rsi_manual(close_prices, CONFIG["RSI_PERIOD"])
        if rsi_err:
            indicators["stoch_rsi"]["error"] = f"RSI Error: {rsi_err}"
            verbose_print(
                f"StochRSI calc stopped due to RSI error: {rsi_err}", color=Fore.YELLOW
            )
        elif rsi_list:
            k, d, stoch_err = calculate_stoch_rsi_manual(
                rsi_list, CONFIG["STOCH_K_PERIOD"], CONFIG["STOCH_D_PERIOD"]
            )
            indicators["stoch_rsi"]["k"] = k
            indicators["stoch_rsi"]["d"] = d
            indicators["stoch_rsi"]["error"] = (
                stoch_err  # Store potential StochRSI calculation error
            )
            if stoch_err:
                verbose_print(
                    f"StochRSI calculation error: {stoch_err}", color=Fore.YELLOW
                )
        else:
            indicators["stoch_rsi"]["error"] = "RSI List Empty/Invalid"
            verbose_print(
                "StochRSI calc stopped: RSI list empty/invalid.", color=Fore.YELLOW
            )

        latest_data["indicators"] = indicators
        latest_data["last_update_times"]["indicators"] = time.time()
        verbose_print("Indicators calculated successfully.")

    except Exception as e:
        print_color(f"# Indicator Calculation Critical Error: {e}", Fore.RED)
        traceback.print_exc()
        # Mark all indicators as error on critical failure
        for k in indicators:
            indicators[k]["error"] = "Crit Calc Exception"
        latest_data["indicators"] = indicators


async def calculate_and_store_pivots() -> None:
    """Calculates Fibonacci pivot points based on data in `latest_data`."""
    verbose_print("Calculating pivots...")
    pivot_ohlcv = latest_data.get("pivot_ohlcv")
    calculated_pivots = None
    error_msg = None

    if not pivot_ohlcv or not isinstance(pivot_ohlcv, list) or len(pivot_ohlcv) == 0:
        error_msg = "Pivot OHLCV data missing or empty"
    else:
        # Pivots are calculated based on the *previous* completed candle.
        # If len is 1, it might be the incomplete current candle.
        # If len is 2, index 0 is the previous completed candle.
        prev_candle_index = (
            0 if len(pivot_ohlcv) >= 1 else -1
        )  # Use first candle if available

        if prev_candle_index != -1:
            prev_candle = pivot_ohlcv[prev_candle_index]
            if isinstance(prev_candle, list) and len(prev_candle) >= 5:
                # Indices: [timestamp, open, high, low, close, volume]
                high, low, close = prev_candle[2], prev_candle[3], prev_candle[4]
                calculated_pivots = calculate_fib_pivots(high, low, close)
                if calculated_pivots:
                    latest_data["pivots"] = calculated_pivots
                    latest_data["last_update_times"]["pivots"] = time.time()
                    verbose_print("Pivots calculated successfully.")
                    return  # Success exit
                else:
                    error_msg = "Pivot calculation function failed (invalid HLC?)"
            else:
                error_msg = f"Invalid previous candle format for pivots: {prev_candle}"
        else:
            error_msg = "Not enough candles in pivot OHLCV data."

    # If we reached here, there was an error or no calculation happened
    latest_data["pivots"] = None  # Clear any previous pivots
    if error_msg:
        verbose_print(f"Pivot calculation failed: {error_msg}", color=Fore.YELLOW)


# ==============================================================================
# Display Functions (Read from `latest_data`)
# ==============================================================================
def display_header(
    symbol: str, timestamp: str, balance_info: decimal.Decimal | None, config: dict
) -> None:
    """Displays the header section with symbol, timestamp, and balance."""
    print_color("=" * 85, Fore.CYAN)
    print_color(
        f"📜 Pyrmethus Market Vision: {Fore.MAGENTA}{Style.BRIGHT}{symbol}{Style.RESET_ALL} @ {Fore.YELLOW}{timestamp}",
        Fore.CYAN,
    )

    balance_str = f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    asset = config["FETCH_BALANCE_ASSET"]
    prec = config["BALANCE_DISPLAY_PRECISION"]
    if balance_info is not None:
        try:
            balance_str = f"{Fore.GREEN}{format_decimal(balance_info, prec, prec)} {asset}{Style.RESET_ALL}"
        except Exception as e:
            verbose_print(f"Balance Display Error: {e}")
            balance_str = f"{Fore.YELLOW}Error{Style.RESET_ALL}"

    print_color(f"💰 Balance ({asset}): {balance_str}")
    print_color("-" * 85, Fore.CYAN)


def display_ticker_and_trend(
    ticker_info: dict | None, indicators_info: dict, config: dict, market_info: dict
) -> decimal.Decimal | None:
    """Displays the last price and basic trend indication (vs. SMA1)."""
    price_prec = market_info["price_precision"]
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    last_price: decimal.Decimal | None = None
    curr_price_str = f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    price_color = Fore.WHITE

    if ticker_info and ticker_info.get("last") is not None:
        try:
            last_price = decimal.Decimal(str(ticker_info["last"]))
            sma1_data = indicators_info.get("sma1", {})
            sma1_value = sma1_data.get("value") if not sma1_data.get("error") else None

            if sma1_value:
                price_color = (
                    Fore.GREEN
                    if last_price > sma1_value
                    else Fore.RED
                    if last_price < sma1_value
                    else Fore.YELLOW
                )
            curr_price_str = f"{price_color}{Style.BRIGHT}{format_decimal(last_price, price_prec, min_disp_prec)}{Style.RESET_ALL}"
        except (ValueError, TypeError, decimal.InvalidOperation) as e:
            verbose_print(f"Ticker Price Processing Error: {e}")
            last_price = None
            curr_price_str = f"{Fore.YELLOW}Error{Style.RESET_ALL}"
        except Exception as e:
            verbose_print(f"Ticker Display Error: {e}")
            curr_price_str = f"{Fore.YELLOW}DispErr{Style.RESET_ALL}"

    sma1_data = indicators_info.get("sma1", {})
    sma1_value, sma1_error = sma1_data.get("value"), sma1_data.get("error")
    sma1_p = config["SMA_PERIOD"]
    tf = config["INDICATOR_TIMEFRAME"]
    trend_str = f"SMA({sma1_p}@{tf}): -"
    trend_color = Fore.YELLOW

    if sma1_error:
        trend_str = f"SMA({sma1_p}@{tf}): {sma1_error}"
        trend_color = Fore.YELLOW
    elif sma1_value is not None:
        sma1_fmt = format_decimal(sma1_value, price_prec, min_disp_prec)
        if last_price:
            trend_color = (
                Fore.GREEN
                if last_price > sma1_value
                else Fore.RED
                if last_price < sma1_value
                else Fore.YELLOW
            )
            trend_direction = (
                "Above"
                if last_price > sma1_value
                else "Below"
                if last_price < sma1_value
                else "On"
            )
            trend_str = f"{trend_direction} SMA ({sma1_fmt})"
        else:
            trend_str = f"SMA({sma1_p}@{tf}): {sma1_fmt} (No Price)"
            trend_color = Fore.WHITE  # Neutral color if price is missing
    # else: Keep default "SMA(...): -"

    print_color(
        f"  Last Price: {curr_price_str} | {trend_color}{trend_str}{Style.RESET_ALL}"
    )
    return last_price  # Return the decimal price for other display functions


def display_indicators(
    indicators_info: dict,
    config: dict,
    market_info: dict,
    last_price: decimal.Decimal
    | None,  # Currently unused, but could be for comparisons
) -> None:
    """Displays calculated indicator values."""
    price_prec = market_info["price_precision"]
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    stoch_prec = config["STOCH_RSI_DISPLAY_PRECISION"]
    config["INDICATOR_TIMEFRAME"]

    # --- Line 1: SMA2 & EMAs ---
    line1_parts = []
    sma2_data = indicators_info.get("sma2", {})
    sma2_val, sma2_err = sma2_data.get("value"), sma2_data.get("error")
    sma2_p = config["SMA2_PERIOD"]
    sma2_str = (
        f"SMA2({sma2_p}): {Fore.YELLOW}{sma2_err or 'N/A'}{Style.RESET_ALL}"
        if sma2_err or sma2_val is None
        else f"SMA2({sma2_p}): {format_decimal(sma2_val, price_prec, min_disp_prec)}"
    )
    line1_parts.append(sma2_str)

    ema1_data, ema2_data = (
        indicators_info.get("ema1", {}),
        indicators_info.get("ema2", {}),
    )
    ema1_val, ema1_err = ema1_data.get("value"), ema1_data.get("error")
    ema2_val, ema2_err = ema2_data.get("value"), ema2_data.get("error")
    ema_err = ema1_err or ema2_err
    ema1_p, ema2_p = config["EMA1_PERIOD"], config["EMA2_PERIOD"]
    ema_str = (
        f"EMA({ema1_p}/{ema2_p}): {Fore.YELLOW}{ema_err or 'N/A'}{Style.RESET_ALL}"
    )
    if not ema_err and ema1_val is not None and ema2_val is not None:
        ema1_fmt = format_decimal(ema1_val, price_prec, min_disp_prec)
        ema2_fmt = format_decimal(ema2_val, price_prec, min_disp_prec)
        ema_color = (
            Fore.GREEN
            if ema1_val > ema2_val
            else Fore.RED
            if ema1_val < ema2_val
            else Fore.YELLOW
        )
        ema_str = (
            f"EMA({ema1_p}/{ema2_p}): {ema_color}{ema1_fmt}/{ema2_fmt}{Style.RESET_ALL}"
        )
    line1_parts.append(ema_str)
    print_color(f"  {' | '.join(line1_parts)}")

    # --- Line 2: Momentum & StochRSI ---
    line2_parts = []
    mom_data = indicators_info.get("momentum", {})
    mom_val, mom_err = mom_data.get("value"), mom_data.get("error")
    mom_p = config["MOMENTUM_PERIOD"]
    mom_str = f"Mom({mom_p}): {Fore.YELLOW}{mom_err or 'N/A'}{Style.RESET_ALL}"
    if not mom_err and mom_val is not None:
        mom_color = (
            Fore.GREEN if mom_val > 0 else Fore.RED if mom_val < 0 else Fore.YELLOW
        )
        mom_fmt = format_decimal(
            mom_val, price_prec, min_disp_prec
        )  # Momentum is price difference
        mom_str = f"Mom({mom_p}): {mom_color}{mom_fmt}{Style.RESET_ALL}"
    line2_parts.append(mom_str)

    stoch_data = indicators_info.get("stoch_rsi", {})
    st_k, st_d, st_err = (
        stoch_data.get("k"),
        stoch_data.get("d"),
        stoch_data.get("error"),
    )
    rsi_p, k_p, d_p = (
        config["RSI_PERIOD"],
        config["STOCH_K_PERIOD"],
        config["STOCH_D_PERIOD"],
    )
    stoch_str = f"StochRSI({rsi_p},{k_p},{d_p}): {Fore.YELLOW}{(st_err[:15] + '...' if isinstance(st_err, str) and len(st_err) > 15 else st_err) or 'N/A'}{Style.RESET_ALL}"
    if not st_err and st_k is not None:
        k_fmt = format_decimal(st_k, stoch_prec)
        d_fmt = format_decimal(st_d, stoch_prec) if st_d is not None else "N/A"
        osold_thr = config["STOCH_RSI_OVERSOLD"]
        obought_thr = config["STOCH_RSI_OVERBOUGHT"]
        k_color, signal = Fore.WHITE, ""

        is_os = st_k < osold_thr and (
            st_d is None or st_d < osold_thr + 5
        )  # Condition slightly relaxed for D
        is_ob = st_k > obought_thr and (
            st_d is None or st_d > obought_thr - 5
        )  # Condition slightly relaxed for D

        if is_os:
            k_color, signal = Fore.GREEN, "(OS)"
        elif is_ob:
            k_color, signal = Fore.RED, "(OB)"
        elif st_d is not None:  # Check cross if not OS/OB
            k_color = (
                Fore.LIGHTGREEN_EX
                if st_k > st_d
                else Fore.LIGHTRED_EX
                if st_k < st_d
                else Fore.YELLOW
            )

        # Apply color to K, D, and signal
        stoch_str = f"StochRSI: {k_color}K={k_fmt}{Style.RESET_ALL} D={k_color}{d_fmt}{Style.RESET_ALL} {k_color}{signal}{Style.RESET_ALL}"

    line2_parts.append(stoch_str)
    print_color(f"  {' | '.join(line2_parts)}")


def display_position(
    position_info: dict,  # Processed position info
    ticker_info: dict
    | None,  # Needed for calculating PnL if exchange doesn't provide it
    market_info: dict,
    config: dict,
) -> None:
    """Displays the current position details and PnL."""
    pnl_prec = config["PNL_PRECISION"]
    price_prec = market_info["price_precision"]
    amount_prec = market_info["amount_precision"]
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]

    pnl_str = f"{Fore.LIGHTBLACK_EX}Position: None{Style.RESET_ALL}"  # Default message

    if position_info.get("has_position"):
        pos = position_info["position"]  # The raw position dictionary
        try:
            side = pos.get("side", "N/A").lower()
            size_str = pos.get("contracts", "0")
            entry_str = pos.get("entryPrice", "0")
            quote_asset = pos.get("quoteAsset", config["FETCH_BALANCE_ASSET"])
            pnl_val = position_info.get(
                "unrealizedPnl"
            )  # Use pre-processed decimal value

            size = decimal.Decimal(size_str)
            entry = decimal.Decimal(entry_str) if entry_str else decimal.Decimal("0")

            size_fmt = format_decimal(size, amount_prec)
            entry_fmt = format_decimal(entry, price_prec, min_disp_prec)
            side_color = (
                Fore.GREEN
                if side == "long"
                else Fore.RED
                if side == "short"
                else Fore.YELLOW
            )
            side_display = side.capitalize()

            # Attempt to calculate PnL manually if not provided and possible
            if (
                pnl_val is None
                and ticker_info
                and ticker_info.get("last")
                and entry > 0
                and size != 0
            ):
                last_p = decimal.Decimal(str(ticker_info["last"]))
                # Note: This calculation might not perfectly match exchange PnL due to fees, funding, etc.
                if side == "long":
                    pnl_val = (last_p - entry) * size
                elif side == "short":
                    pnl_val = (entry - last_p) * size
                # else: Hedge mode or unknown side, cannot calculate simply

            pnl_val_str, pnl_color = "N/A", Fore.WHITE
            if pnl_val is not None:
                pnl_val_str = format_decimal(pnl_val, pnl_prec)
                pnl_color = Fore.GREEN if pnl_val >= 0 else Fore.RED

            pnl_str = (
                f"Position: {side_color}{side_display} {size_fmt}{Style.RESET_ALL} "
                f"@ {Fore.YELLOW}{entry_fmt}{Style.RESET_ALL} | "
                f"uPNL: {pnl_color}{pnl_val_str} {quote_asset}{Style.RESET_ALL}"
            )

        except (ValueError, TypeError, decimal.InvalidOperation, KeyError) as e:
            verbose_print(f"Position Display Data Error: {e} - Data: {pos}")
            pnl_str = f"{Fore.YELLOW}Position Data Error{Style.RESET_ALL}"
        except Exception as e:
            verbose_print(f"Position Display Unexpected Error: {e}")
            pnl_str = f"{Fore.YELLOW}Position Display Error{Style.RESET_ALL}"

    print_color(f"  {pnl_str}")


def display_pivots(
    pivots_info: dict | None,
    last_price: decimal.Decimal | None,
    market_info: dict,
    config: dict,
) -> None:
    """Displays the calculated Fibonacci pivot points."""
    print_color(
        f"--- Fibonacci Pivots (Prev {config['PIVOT_TIMEFRAME']}) ---", Fore.BLUE
    )
    if not pivots_info:
        print_color(f"  {Fore.YELLOW}Pivot data unavailable.{Style.RESET_ALL}")
        return

    price_prec = market_info["price_precision"]
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    # Adjust width dynamically based on expected price format length
    width = max(12, price_prec + 7)  # Base width + precision + decimal point + buffer

    levels = ["R3", "R2", "R1", "PP", "S1", "S2", "S3"]
    lines = {}

    for level in levels:
        value = pivots_info.get(level)
        if value is not None:
            val_str = format_decimal(value, price_prec, min_disp_prec)
            color = (
                Fore.RED
                if "R" in level
                else Fore.GREEN
                if "S" in level
                else Fore.YELLOW
            )
            highlight = ""
            # Highlight if price is very close to a pivot level
            if last_price and value > 0:
                try:
                    # Check relative difference, threshold 0.1%
                    if abs(last_price - value) / value < decimal.Decimal("0.001"):
                        highlight = (
                            Back.LIGHTBLACK_EX
                            + Fore.WHITE
                            + Style.BRIGHT
                            + " *NEAR* "
                            + Style.RESET_ALL
                        )
                except (decimal.InvalidOperation, ZeroDivisionError):
                    pass  # Avoid errors if value is zero

            lines[level] = (
                f"{color}{level:>3}: {val_str.rjust(width)}{Style.RESET_ALL}{highlight}"
            )
        else:
            lines[level] = (
                f"{level:>3}: {'N/A'.rjust(width)}"  # Ensure N/A is also right-justified
            )

    # Print pivots in pairs (R3/S3, R2/S2, R1/S1) centered around PP


def display_orderbook(
    analyzed_ob: dict | None, market_info: dict, config: dict
) -> tuple[dict[int, decimal.Decimal], dict[int, decimal.Decimal]]:
    """Displays the formatted order book side-by-side.

    Returns:
        Tuple[Dict[int, Decimal], Dict[int, Decimal]]: Mappings of display index (1-based)
                                                      to price for asks and bids respectively,
                                                      used for interactive ordering.
    """
    print_color("--- Order Book Depths ---", Fore.BLUE)
    ask_map: dict[int, decimal.Decimal] = {}
    bid_map: dict[int, decimal.Decimal] = {}

    if not analyzed_ob:
        print_color(f"  {Fore.YELLOW}Order book data unavailable.{Style.RESET_ALL}")
        return ask_map, bid_map

    p_prec = market_info["price_precision"]
    market_info["amount_precision"]
    min_p_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    v_disp_prec = config["VOLUME_DISPLAY_PRECISION"]
    depth = config["MAX_ORDERBOOK_DEPTH_DISPLAY"]

    # Calculate column widths dynamically
    idx_w = len(f"[A{depth}]") + 1  # Width for index like '[A30] '
    p_w = max(10, p_prec + 4)  # Width for price
    v_w = max(10, v_disp_prec + 5)  # Width for volume
    cum_v_w = max(12, v_disp_prec + 7)  # Width for cumulative volume display
    total_col_w = idx_w + p_w + v_w + cum_v_w + 1  # Total width for one side

    ask_lines, bid_lines = [], []

    # Prepare Ask lines (display asks top-down, best ask at the bottom visually)
    # Reverse the asks list from analyzed_ob for display order
    display_asks = list(reversed(analyzed_ob.get("asks", [])))
    for idx, ask in enumerate(display_asks):
        if idx >= depth:
            break
        display_index = idx + 1  # 1-based index for user selection
        idx_str = f"[A{display_index}]".ljust(idx_w)
        price_str = format_decimal(ask["price"], p_prec, min_p_prec)
        # Format cumulative volume for display (less prominent)
        cum_v_str = (
            f"{Fore.LIGHTBLACK_EX}(Cum:{ask['cumulative_volume']}){Style.RESET_ALL}"
        )
        line = (
            f"{Fore.CYAN}{idx_str}"
            f"{Style.NORMAL}{Fore.WHITE}{price_str:<{p_w}}"
            f"{ask['style']}{ask['color']}{ask['volume_str']:<{v_w}}{Style.RESET_ALL} "
            f"{cum_v_str:<{cum_v_w}}"
        )
        ask_lines.append(line)
        ask_map[display_index] = ask["price"]  # Map display index to price

    # Prepare Bid lines (display bids top-down, best bid at the top)
    for idx, bid in enumerate(analyzed_ob.get("bids", [])):
        if idx >= depth:
            break
        display_index = idx + 1  # 1-based index
        idx_str = f"[B{display_index}]".ljust(idx_w)
        price_str = format_decimal(bid["price"], p_prec, min_p_prec)
        cum_v_str = (
            f"{Fore.LIGHTBLACK_EX}(Cum:{bid['cumulative_volume']}){Style.RESET_ALL}"
        )
        line = (
            f"{Fore.CYAN}{idx_str}"
            f"{Style.NORMAL}{Fore.WHITE}{price_str:<{p_w}}"
            f"{bid['style']}{bid['color']}{bid['volume_str']:<{v_w}}{Style.RESET_ALL} "
            f"{cum_v_str:<{cum_v_w}}"
        )
        bid_lines.append(line)
        bid_map[display_index] = bid["price"]  # Map display index to price

    # Print Headers
    print_color(
        f"{'Asks'.center(total_col_w)}{'Bids'.center(total_col_w)}", Fore.LIGHTBLACK_EX
    )
    print_color(f"{'-' * total_col_w} {'-' * total_col_w}", Fore.LIGHTBLACK_EX)

    # Print side-by-side
    max_rows = max(len(ask_lines), len(bid_lines))
    for i in range(max_rows):
        ask_lines[i] if i < len(ask_lines) else " " * total_col_w
        bid_lines[i] if i < len(bid_lines) else ""

    # Calculate and print Spread
    best_ask = display_asks[0]["price"] if display_asks else decimal.Decimal("NaN")
    best_bid = (
        analyzed_ob["bids"][0]["price"]
        if analyzed_ob.get("bids")
        else decimal.Decimal("NaN")
    )
    spread = (
        best_ask - best_bid
        if best_ask.is_finite() and best_bid.is_finite()
        else decimal.Decimal("NaN")
    )
    spread_str = (
        format_decimal(spread, p_prec, min_p_prec) if spread.is_finite() else "N/A"
    )
    print_color(f"\n--- Spread: {spread_str} ---", Fore.MAGENTA, Style.DIM)

    return ask_map, bid_map


def display_volume_analysis(
    analyzed_ob: dict | None, market_info: dict, config: dict
) -> None:
    """Displays summary volume analysis based on fetched order book depth."""
    if not analyzed_ob:
        return  # Don't display if no data

    a_prec = market_info["amount_precision"]
    p_prec = market_info["price_precision"]
    v_disp_prec = config["VOLUME_DISPLAY_PRECISION"]
    min_p_prec = config["MIN_PRICE_DISPLAY_PRECISION"]

    print_color("\n--- Volume Analysis (Fetched Depth) ---", Fore.BLUE)
    tot_a = analyzed_ob.get("ask_total_volume_fetched", decimal.Decimal("0"))
    tot_b = analyzed_ob.get("bid_total_volume_fetched", decimal.Decimal("0"))
    cum_a_disp = analyzed_ob.get(
        "cumulative_ask_volume_displayed", decimal.Decimal("0")
    )
    cum_b_disp = analyzed_ob.get(
        "cumulative_bid_volume_displayed", decimal.Decimal("0")
    )
    depth_disp = config["MAX_ORDERBOOK_DEPTH_DISPLAY"]

    print_color(
        f"  Total Ask Vol: {Fore.RED}{format_decimal(tot_a, a_prec, v_disp_prec)}{Style.RESET_ALL} | "
        f"Total Bid Vol: {Fore.GREEN}{format_decimal(tot_b, a_prec, v_disp_prec)}{Style.RESET_ALL}"
    )
    print_color(
        f"  Cum Ask (Top {depth_disp}): {Fore.RED}{format_decimal(cum_a_disp, a_prec, v_disp_prec)}{Style.RESET_ALL} | "
        f"Cum Bid (Top {depth_disp}): {Fore.GREEN}{format_decimal(cum_b_disp, a_prec, v_disp_prec)}{Style.RESET_ALL}"
    )

    imb_ratio = analyzed_ob.get("volume_imbalance_ratio_fetched", decimal.Decimal("0"))
    imb_color, imb_str = Fore.WHITE, "N/A"
    if imb_ratio.is_infinite():
        imb_color, imb_str = Fore.LIGHTGREEN_EX, "Inf (Strong Bid)"
    elif imb_ratio.is_finite():
        imb_str = format_decimal(imb_ratio, 2)  # Display ratio with 2 decimal places
        # Color based on ratio thresholds
        if imb_ratio > decimal.Decimal("1.5"):
            imb_color = Fore.GREEN  # More bids
        elif imb_ratio < decimal.Decimal("0.67"):
            imb_color = Fore.RED  # More asks (0.67 is approx 1/1.5)
        elif imb_ratio.is_zero() and tot_a > 0:
            imb_color = Fore.LIGHTRED_EX  # Zero bids, non-zero asks
        else:
            imb_color = Fore.YELLOW  # Relatively balanced

    ask_vwap = analyzed_ob.get("ask_vwap_fetched", decimal.Decimal("0"))
    bid_vwap = analyzed_ob.get("bid_vwap_fetched", decimal.Decimal("0"))
    ask_vwap_str = (
        format_decimal(ask_vwap, p_prec, min_p_prec) if ask_vwap > 0 else "N/A"
    )
    bid_vwap_str = (
        format_decimal(bid_vwap, p_prec, min_p_prec) if bid_vwap > 0 else "N/A"
    )

    print_color(
        f"  Imbalance (B/A): {imb_color}{imb_str}{Style.RESET_ALL} | "
        f"VWAP Ask: {Fore.YELLOW}{ask_vwap_str}{Style.RESET_ALL} | "
        f"VWAP Bid: {Fore.YELLOW}{bid_vwap_str}{Style.RESET_ALL}"
    )

    # Simple pressure interpretation based on imbalance
    print_color("--- Pressure Reading ---", Fore.BLUE)
    if imb_ratio.is_infinite():
        print_color(
            "  Extreme Bid Dominance (Buy Pressure)", Fore.LIGHTGREEN_EX, Style.BRIGHT
        )
    elif imb_ratio.is_zero() and tot_a > 0:
        print_color(
            "  Extreme Ask Dominance (Sell Pressure)", Fore.LIGHTRED_EX, Style.BRIGHT
        )
    elif imb_ratio > decimal.Decimal("1.75"):  # Stricter threshold for "Strong"
        print_color("  Strong Buy Pressure", Fore.GREEN, Style.BRIGHT)
    elif imb_ratio < decimal.Decimal("0.57"):  # Stricter threshold (approx 1/1.75)
        print_color("  Strong Sell Pressure", Fore.RED, Style.BRIGHT)
    elif imb_ratio > decimal.Decimal("1.2"):
        print_color("  Moderate Buy Pressure", Fore.GREEN)
    elif imb_ratio < decimal.Decimal("0.83"):  # approx 1/1.2
        print_color("  Moderate Sell Pressure", Fore.RED)
    else:
        print_color("  Volume Relatively Balanced", Fore.WHITE)

    print_color("=" * 85, Fore.CYAN)  # Footer line


def display_combined_analysis_async(
    shared_data: dict, market_info: dict, config: dict
) -> tuple[dict[int, decimal.Decimal], dict[int, decimal.Decimal]]:
    """Orchestrates the display of all analysis components using data from the shared state.

    Args:
        shared_data: The global `latest_data` dictionary.
        market_info: The market details dictionary.
        config: The main configuration dictionary.

    Returns:
        Tuple[Dict[int, Decimal], Dict[int, Decimal]]: Ask and Bid price maps from display_orderbook.
    """
    # --- Safely extract data from shared state ---
    ticker_info = shared_data.get("ticker")
    analyzed_ob = shared_data.get("orderbook")
    indicators_info = shared_data.get("indicators", {})  # Default to empty dict
    positions_list = shared_data.get("positions", [])  # Default to empty list
    pivots_info = shared_data.get("pivots")
    balance_info = shared_data.get("balance")  # Already a Decimal or None

    # Process position info for easier display handling
    position_info_processed = {
        "has_position": False,
        "position": None,
        "unrealizedPnl": None,
    }
    if positions_list:  # Assuming only one position per symbol in non-hedge mode or we only care about the first one
        current_pos = positions_list[0]
        position_info_processed["has_position"] = True
        position_info_processed["position"] = current_pos
        try:
            pnl_raw = current_pos.get("unrealizedPnl")
            if pnl_raw is not None:
                position_info_processed["unrealizedPnl"] = decimal.Decimal(str(pnl_raw))
        except (decimal.InvalidOperation, TypeError, KeyError):
            verbose_print(f"Could not process PnL for position: {pnl_raw}")
            position_info_processed["unrealizedPnl"] = (
                None  # Keep as None if processing fails
            )

    # Determine timestamp for display header
    ts_ob_update = shared_data.get("last_update_times", {}).get("orderbook")
    ts_tk_update = shared_data.get("last_update_times", {}).get("ticker")
    timestamp_str = "N/A"
    latest_ts = 0
    source = ""
    if analyzed_ob and analyzed_ob.get("timestamp"):
        timestamp_str = analyzed_ob["timestamp"]
        source = "(OB)"
    elif ticker_info and ticker_info.get("datetime"):
        timestamp_str = ticker_info["datetime"]
        source = "(Tk)"
    elif ts_ob_update and ts_ob_update > latest_ts:
        latest_ts = ts_ob_update
        source = "(OB Update)"
    elif ts_tk_update and ts_tk_update > latest_ts:
        latest_ts = ts_tk_update
        source = "(Tk Update)"

    if latest_ts > 0:
        timestamp_str = (
            time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(latest_ts)) + source
        )
    elif timestamp_str == "N/A":  # Fallback if no timestamps available
        timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()) + "(Now)"

    # --- Clear screen and display sections ---

    symbol = market_info.get("symbol", "UNKNOWN")
    display_header(symbol, timestamp_str, balance_info, config)

    last_price = display_ticker_and_trend(
        ticker_info, indicators_info, config, market_info
    )

    display_indicators(indicators_info, config, market_info, last_price)

    display_position(position_info_processed, ticker_info, market_info, config)

    display_pivots(pivots_info, last_price, market_info, config)

    ask_map, bid_map = display_orderbook(
        analyzed_ob, market_info, config
    )  # Returns maps for interactive orders

    display_volume_analysis(analyzed_ob, market_info, config)  # Includes footer line

    # Display Connection Status
    status_parts = []
    conn_statuses = shared_data.get("connection_status", {})
    for key in CONN_STATUS_KEYS:  # Iterate in defined order
        status = conn_statuses.get(key, "N/A")
        color = (
            Fore.GREEN
            if status == OK
            else Fore.YELLOW
            if status == CONNECTING
            else Fore.RED
            if status == ERROR
            else Fore.WHITE
        )  # Default/Init
        status_parts.append(f"{key.upper()}:{color}{status}{Style.RESET_ALL}")
    print_color(
        f"--- Status: {' | '.join(status_parts)} ---",
        color=Fore.MAGENTA,
        style=Style.DIM,
    )

    return ask_map, bid_map


# ==============================================================================
# Async Trading Functions
# ==============================================================================
async def place_market_order_async(
    exchange_pro: ccxtpro.Exchange,
    symbol: str,
    side: str,  # 'buy' or 'sell'
    amount_str: str,
    market_info: dict,
    config: dict,
) -> None:
    """Handles the logic for placing a market order asynchronously."""
    print_color(
        f"\n{Fore.CYAN}# Preparing ASYNC {side.upper()} market order...{Style.RESET_ALL}"
    )
    side = side.lower()
    if side not in ["buy", "sell"]:
        print_color(f"Invalid side '{side}'. Must be 'buy' or 'sell'.", Fore.RED)
        return

    try:
        amount = decimal.Decimal(amount_str)
        min_amount = market_info.get("min_amount", decimal.Decimal("0"))
        amount_prec = market_info["amount_precision"]
        amount_step = market_info.get(
            "amount_step", decimal.Decimal("1") / (decimal.Decimal("10") ** amount_prec)
        )

        if amount <= 0:
            print_color("Order amount must be positive.", Fore.YELLOW)
            return
        if min_amount > 0 and amount < min_amount:
            print_color(
                f"Amount ({amount}) is less than minimum required ({format_decimal(min_amount, amount_prec)}).",
                Fore.YELLOW,
            )
            return

        # Adjust amount to comply with step size (round down to be safe)
        if amount_step > 0 and (amount % amount_step) != 0:
            original_amount = amount
            amount = (amount // amount_step) * amount_step
            print_color(
                f"Amount adjusted for step size ({amount_step}): {original_amount} -> {amount}",
                Fore.YELLOW,
            )
            if amount <= 0 or (min_amount > 0 and amount < min_amount):
                print_color(
                    "Adjusted amount is invalid (zero or below minimum). Order cancelled.",
                    Fore.RED,
                )
                return

        final_amount_str = format_decimal(
            amount, amount_prec
        )  # Format for confirmation and API

    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        print_color(
            f"Invalid amount specified: '{amount_str}'. Error: {e}", Fore.YELLOW
        )
        return
    except Exception as e:
        print_color(f"Error processing amount: {e}", Fore.RED)
        return

    side_color = Fore.GREEN if side == "buy" else Fore.RED
    prompt = (
        f"{Style.BRIGHT}Confirm MARKET {side_color}{side.upper()}{Style.RESET_ALL} "
        f"{Fore.YELLOW}{final_amount_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL}? "
        f"({Fore.GREEN}y{Style.RESET_ALL}/{Fore.RED}n{Style.RESET_ALL}): {Style.RESET_ALL}"
    )

    try:
        # Run synchronous input in a separate thread to avoid blocking asyncio event loop
        confirm = await asyncio.to_thread(input, prompt)
        confirm = confirm.strip().lower()
    except (EOFError, KeyboardInterrupt):
        print_color("\nOrder cancelled by user input.", Fore.YELLOW)
        return
    except Exception as e:
        print_color(f"\nError reading input: {e}", Fore.RED)
        return

    if confirm == "y" or confirm == "yes":
        print_color(
            f"{Fore.CYAN}# Transmitting ASYNC market order for {final_amount_str} {symbol}...",
            style=Style.DIM,
            end="\r",
        )
        order_result = None
        error_message = None
        try:
            # Include positionIdx for Hedge Mode if configured
            params = {}
            position_idx = config.get("POSITION_IDX", 0)
            if position_idx != 0:
                params["positionIdx"] = position_idx
                verbose_print(f"Using positionIdx: {position_idx}")

            # Use float for amount in ccxt create order methods
            order_result = await exchange_pro.create_market_order(
                symbol, side, float(amount), params=params
            )
            sys.stdout.write("\033[K")  # Clear the "Transmitting..." line

            order_id = order_result.get("id", "N/A")
            avg_price = order_result.get("average")
            filled_amount = order_result.get("filled")
            status = order_result.get("status", "unknown")

            msg = f"✅ Market {side.upper()} Order {status.upper()} [{order_id}]"
            details = []
            if filled_amount:
                details.append(f"Filled: {format_decimal(filled_amount, amount_prec)}")
            if avg_price:
                details.append(
                    f"Avg Price: {format_decimal(avg_price, market_info['price_precision'])}"
                )
            if details:
                msg += f" ({', '.join(details)})"

            print_color(f"\n{msg}", Fore.GREEN, Style.BRIGHT)
            termux_toast(f"{symbol} Market {side.upper()} {status}: {order_id}")

        except ccxt.InsufficientFunds as e:
            error_message = f"Insufficient Funds: {e}"
        except ccxt.InvalidOrder as e:
            error_message = f"Invalid Order: {e}"
        except ccxt.ExchangeError as e:
            error_message = f"Exchange Error: {e}"
        except Exception as e:
            error_message = f"Unexpected Error: {e}"
            traceback.print_exc()  # Print stack trace for unexpected errors

        finally:
            sys.stdout.write("\033[K")  # Ensure line is cleared even on error
            if error_message:
                print_color(f"\n❌ Market Order Failed: {error_message}", Fore.RED)
                termux_toast(f"{symbol} Mkt Fail: {error_message[:30]}...", "long")
    else:
        print_color("Market order cancelled by user.", Fore.YELLOW)


async def place_limit_order_async(
    exchange_pro: ccxtpro.Exchange,
    symbol: str,
    side: str,  # 'buy' or 'sell'
    amount_str: str,
    price_str: str,
    market_info: dict,
    config: dict,
) -> None:
    """Handles the logic for placing a limit order asynchronously."""
    print_color(
        f"\n{Fore.CYAN}# Preparing ASYNC {side.upper()} limit order...{Style.RESET_ALL}"
    )
    side = side.lower()
    if side not in ["buy", "sell"]:
        print_color(f"Invalid side '{side}'. Must be 'buy' or 'sell'.", Fore.RED)
        return

    try:
        amount = decimal.Decimal(amount_str)
        price = decimal.Decimal(price_str)
        min_amount = market_info.get("min_amount", decimal.Decimal("0"))
        amount_prec = market_info["amount_precision"]
        price_prec = market_info["price_precision"]
        amount_step = market_info.get(
            "amount_step", decimal.Decimal("1") / (decimal.Decimal("10") ** amount_prec)
        )
        price_tick = market_info.get(
            "price_tick_size",
            decimal.Decimal("1") / (decimal.Decimal("10") ** price_prec),
        )

        # --- Validate Amount ---
        if amount <= 0:
            print_color("Order amount must be positive.", Fore.YELLOW)
            return
        if min_amount > 0 and amount < min_amount:
            print_color(
                f"Amount ({amount}) is less than minimum required ({format_decimal(min_amount, amount_prec)}).",
                Fore.YELLOW,
            )
            return
        # Adjust amount for step size (round down)
        if amount_step > 0 and (amount % amount_step) != 0:
            original_amount = amount
            amount = (amount // amount_step) * amount_step
            print_color(
                f"Amount adjusted for step size ({amount_step}): {original_amount} -> {amount}",
                Fore.YELLOW,
            )
            if amount <= 0 or (min_amount > 0 and amount < min_amount):
                print_color("Adjusted amount is invalid. Order cancelled.", Fore.RED)
                return

        # --- Validate Price ---
        if price <= 0:
            print_color("Order price must be positive.", Fore.YELLOW)
            return
        # Adjust price for tick size (round to nearest tick)
        if price_tick > 0 and (price % price_tick) != 0:
            original_price = price
            # Use quantize for correct rounding based on tick size decimal places
            price = price.quantize(price_tick, rounding=decimal.ROUND_HALF_UP)
            print_color(
                f"Price adjusted for tick size ({price_tick}): {original_price} -> {price}",
                Fore.YELLOW,
            )
            if price <= 0:
                print_color(
                    "Adjusted price is invalid (zero or negative). Order cancelled.",
                    Fore.RED,
                )
                return

        final_amount_str = format_decimal(amount, amount_prec)
        final_price_str = format_decimal(price, price_prec)

    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        print_color(
            f"Invalid amount ('{amount_str}') or price ('{price_str}'). Error: {e}",
            Fore.YELLOW,
        )
        return
    except Exception as e:
        print_color(f"Error processing amount/price: {e}", Fore.RED)
        return

    side_color = Fore.GREEN if side == "buy" else Fore.RED
    prompt = (
        f"{Style.BRIGHT}Confirm LIMIT {side_color}{side.upper()}{Style.RESET_ALL} "
        f"{Fore.YELLOW}{final_amount_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL} @ "
        f"{Fore.YELLOW}{final_price_str}{Style.RESET_ALL}? "
        f"({Fore.GREEN}y{Style.RESET_ALL}/{Fore.RED}n{Style.RESET_ALL}): {Style.RESET_ALL}"
    )

    try:
        confirm = await asyncio.to_thread(input, prompt)
        confirm = confirm.strip().lower()
    except (EOFError, KeyboardInterrupt):
        print_color("\nOrder cancelled by user input.", Fore.YELLOW)
        return
    except Exception as e:
        print_color(f"\nError reading input: {e}", Fore.RED)
        return

    if confirm == "y" or confirm == "yes":
        print_color(
            f"{Fore.CYAN}# Transmitting ASYNC limit order @ {final_price_str}...",
            style=Style.DIM,
            end="\r",
        )
        order_result = None
        error_message = None
        try:
            params = {}
            position_idx = config.get("POSITION_IDX", 0)
            if position_idx != 0:
                params["positionIdx"] = position_idx
                verbose_print(f"Using positionIdx: {position_idx}")

            # Use float for amount and price in ccxt create order methods
            order_result = await exchange_pro.create_limit_order(
                symbol, side, float(amount), float(price), params=params
            )
            sys.stdout.write("\033[K")  # Clear the "Transmitting..." line

            order_id = order_result.get("id", "N/A")
            order_price = order_result.get("price")  # Price the order was placed at
            order_amount = order_result.get("amount")  # Amount the order was placed for
            status = order_result.get(
                "status", "unknown"
            )  # e.g., 'open', 'closed' (if filled immediately)

            msg = f"✅ Limit {side.upper()} Order {status.upper()} [{order_id}]"
            details = []
            if order_amount:
                details.append(f"Amount: {format_decimal(order_amount, amount_prec)}")
            if order_price:
                details.append(f"Price: {format_decimal(order_price, price_prec)}")
            # Add filled info if available (might be partially/fully filled quickly)
            filled_amount = order_result.get("filled")
            if filled_amount and filled_amount > 0:
                details.append(f"Filled: {format_decimal(filled_amount, amount_prec)}")

            if details:
                msg += f" ({', '.join(details)})"

            print_color(f"\n{msg}", Fore.GREEN, Style.BRIGHT)
            termux_toast(f"{symbol} Limit {side.upper()} {status}: {order_id}")

        except ccxt.InsufficientFunds as e:
            error_message = f"Insufficient Funds: {e}"
        except ccxt.InvalidOrder as e:
            error_message = f"Invalid Order: {e}"  # e.g., price too far, size too small
        except ccxt.ExchangeError as e:
            error_message = f"Exchange Error: {e}"
        except Exception as e:
            error_message = f"Unexpected Error: {e}"
            traceback.print_exc()

        finally:
            sys.stdout.write("\033[K")  # Ensure line is cleared
            if error_message:
                print_color(f"\n❌ Limit Order Failed: {error_message}", Fore.RED)
                termux_toast(f"{symbol} Lim Fail: {error_message[:30]}...", "long")
    else:
        print_color("Limit order cancelled by user.", Fore.YELLOW)


async def place_limit_order_interactive_async(
    exchange_pro: ccxtpro.Exchange,
    symbol: str,
    side: str,  # 'buy' or 'sell'
    ask_map: dict[int, decimal.Decimal],  # Display Index -> Ask Price
    bid_map: dict[int, decimal.Decimal],  # Display Index -> Bid Price
    market_info: dict,
    config: dict,
) -> None:
    """Handles interactive limit order placement using displayed order book levels."""
    print_color(
        f"\n{Fore.BLUE}--- Interactive Limit Order ({side.upper()}) ---{Style.RESET_ALL}"
    )
    side = side.lower()

    # Choose the relevant side of the book: Bids for selling, Asks for buying
    target_map, prompt_char = (bid_map, "B") if side == "sell" else (ask_map, "A")

    if not target_map:
        print_color(
            f"Cannot place interactive {side} order: Relevant order book side is empty.",
            Fore.YELLOW,
        )
        return

    selected_price: decimal.Decimal | None = None
    while selected_price is None:
        prompt = (
            f"{Style.BRIGHT}Select Order Book Index "
            f"({prompt_char}1-{prompt_char}{len(target_map)}, '{Fore.YELLOW}c{Style.RESET_ALL}' to cancel): {Style.RESET_ALL}"
        )
        try:
            index_str = await asyncio.to_thread(input, prompt)
            index_str = index_str.strip().upper()

            if index_str == "C":
                print_color("Interactive order cancelled.", Fore.YELLOW)
                return

            # Validate input format (e.g., "A5" or "B12")
            if not index_str.startswith(prompt_char) or not index_str[1:].isdigit():
                print_color(
                    f"Invalid format. Enter like '{prompt_char}1', '{prompt_char}2', etc.",
                    Fore.YELLOW,
                )
                continue

            index = int(index_str[1:])
            selected_price = target_map.get(index)

            if selected_price is None:
                print_color(
                    f"Index {index_str} not found in the displayed order book.",
                    Fore.YELLOW,
                )
            else:
                # Price selected, break the loop
                break

        except (ValueError, IndexError):
            print_color("Invalid index number.", Fore.YELLOW)
        except (EOFError, KeyboardInterrupt):
            print_color("\nInteractive order cancelled by user input.", Fore.YELLOW)
            return
        except Exception as e:
            print_color(f"\nError reading index input: {e}", Fore.RED)
            return

    # Format selected price for display
    price_fmt = format_decimal(selected_price, market_info["price_precision"])
    print_color(
        f"Selected Price: {Fore.YELLOW}{price_fmt}{Style.RESET_ALL} (from index {prompt_char}{index})"
    )

    # Get quantity from user
    while True:
        qty_prompt = (
            f"{Style.BRIGHT}Enter Quantity for {side.upper()} @ {price_fmt} "
            f"('{Fore.YELLOW}c{Style.RESET_ALL}' to cancel): {Style.RESET_ALL}"
        )
        try:
            qty_str = await asyncio.to_thread(input, qty_prompt)
            qty_str = qty_str.strip()

            if qty_str.lower() == "c":
                print_color("Interactive order cancelled.", Fore.YELLOW)
                return

            # Validate quantity immediately before proceeding
            test_qty = decimal.Decimal(qty_str)
            if test_qty <= 0:
                print_color("Quantity must be positive.", Fore.YELLOW)
                continue
            else:
                # Valid quantity entered, break loop
                break

        except (decimal.InvalidOperation, ValueError):
            print_color(
                f"Invalid quantity format: '{qty_str}'. Please enter a number.",
                Fore.YELLOW,
            )
        except (EOFError, KeyboardInterrupt):
            print_color("\nInteractive order cancelled by user input.", Fore.YELLOW)
            return
        except Exception as e:
            print_color(f"\nError reading quantity input: {e}", Fore.RED)
            return

    # Now place the limit order using the selected price and entered quantity
    await place_limit_order_async(
        exchange_pro, symbol, side, qty_str, str(selected_price), market_info, config
    )


# ==============================================================================
# Initial Data Fetching and Calculation Trigger
# ==============================================================================
async def fetch_and_recalculate_data(
    exchange_pro: ccxtpro.Exchange, symbol: str, config: dict
):
    """Helper function to fetch initial/periodic OHLCV and trigger recalculations.
    Can be called initially and potentially periodically if needed outside main fetch loop.
    """
    verbose_print("Running initial/manual data fetch and recalculation...")
    min_ohlcv_needed = config["MIN_OHLCV_CANDLES"]
    ind_tf, piv_tf = config["INDICATOR_TIMEFRAME"], config["PIVOT_TIMEFRAME"]
    history_needed = min_ohlcv_needed + 5  # Fetch slight buffer

    fetch_success = True
    try:
        # Fetch OHLCV data concurrently
        results = await asyncio.gather(
            exchange_pro.fetch_ohlcv(symbol, ind_tf, limit=history_needed),
            exchange_pro.fetch_ohlcv(symbol, piv_tf, limit=2),
            return_exceptions=True,  # Capture exceptions instead of raising
        )
        ind_ohlcv_res, piv_ohlcv_res = results[0], results[1]

        # --- Process Indicator OHLCV Result ---
        if isinstance(ind_ohlcv_res, list):
            if len(ind_ohlcv_res) >= min_ohlcv_needed:
                latest_data["indicator_ohlcv"] = ind_ohlcv_res
                latest_data["last_update_times"]["indicator_ohlcv"] = time.time()
                verbose_print(
                    f"Manual Recalc: Indicator OHLCV fetched ({len(ind_ohlcv_res)})."
                )
                await calculate_and_store_indicators()
            else:
                print_color(
                    f"# Recalc Warn: Insufficient Indicator OHLCV ({len(ind_ohlcv_res)}<{min_ohlcv_needed}) for {ind_tf}",
                    Fore.YELLOW,
                )
                latest_data["indicator_ohlcv"] = None
                await calculate_and_store_indicators()  # Clear indicators
                fetch_success = False  # Mark as partial failure
        elif isinstance(ind_ohlcv_res, Exception):
            print_color(
                f"# Recalc Error (Indicator OHLCV {ind_tf}): {ind_ohlcv_res}",
                Fore.YELLOW,
            )
            latest_data["indicator_ohlcv"] = None
            await calculate_and_store_indicators()  # Clear indicators
            fetch_success = False
        # else: unexpected type, ignore

        # --- Process Pivot OHLCV Result ---
        if isinstance(piv_ohlcv_res, list) and len(piv_ohlcv_res) > 0:
            latest_data["pivot_ohlcv"] = piv_ohlcv_res
            latest_data["last_update_times"]["pivot_ohlcv"] = time.time()
            verbose_print(f"Manual Recalc: Pivot OHLCV fetched ({len(piv_ohlcv_res)}).")
            await calculate_and_store_pivots()
        elif isinstance(piv_ohlcv_res, Exception):
            print_color(
                f"# Recalc Error (Pivot OHLCV {piv_tf}): {piv_ohlcv_res}", Fore.YELLOW
            )
            latest_data["pivots"] = None  # Clear pivots on error
            fetch_success = False
        elif piv_ohlcv_res is not None:  # Empty list returned
            latest_data["pivot_ohlcv"] = piv_ohlcv_res  # Store empty list
            latest_data["pivots"] = None  # Clear pivots
            verbose_print(
                f"Manual Recalc: Pivot OHLCV fetch returned empty list for {piv_tf}."
            )
        # else: unexpected type, ignore

    except Exception as e:
        print_color(f"# Critical Error during manual fetch/recalc: {e}", Fore.RED)
        traceback.print_exc()
        fetch_success = False
        # Clear potentially corrupted calculated data
        latest_data["indicators"] = {}
        latest_data["pivots"] = None

    return fetch_success


async def initial_setup_and_data_fetch(
    exchange_pro: ccxtpro.Exchange, symbol: str, config: dict
) -> bool:
    """Performs initial setup: Fetches market info, initial OHLCV, Balance, Positions.
    Waits briefly for WS connections to potentially establish first.
    """
    await asyncio.sleep(2)  # Small delay to allow WS connections to start establishing
    print_color(
        "# Performing initial data fetch (OHLCV, Balance, Positions)...",
        Fore.CYAN,
        Style.DIM,
    )

    market_info = await get_market_info(exchange_pro, symbol)
    if not market_info:
        print_color(
            "CRITICAL: Failed to get market info during initial setup. Cannot proceed.",
            Fore.RED,
            Style.BRIGHT,
        )
        # In a real application, might raise an exception or attempt retry/exit
        return False  # Indicate setup failure

    latest_data["market_info"] = market_info  # Store globally FIRST

    # Fetch initial OHLCV and calculate indicators/pivots
    ohlcv_fetch_ok = await fetch_and_recalculate_data(exchange_pro, symbol, config)
    if not ohlcv_fetch_ok:
        print_color(
            "# Warning: Initial OHLCV fetch or calculation failed. Indicators/Pivots may be unavailable.",
            Fore.YELLOW,
        )

    # Fetch initial balance and positions concurrently
    bal_pos_fetch_ok = True
    try:
        results = await asyncio.gather(
            exchange_pro.fetch_balance(),
            exchange_pro.fetch_positions([symbol]),  # Only fetch for the current symbol
            return_exceptions=True,
        )
        bal_res, pos_res = results[0], results[1]

        # Process Balance
        if isinstance(bal_res, dict):
            balance_asset = config["FETCH_BALANCE_ASSET"]
            asset_balance = bal_res.get("total", {}).get(balance_asset)
            if asset_balance is not None:
                latest_data["balance"] = decimal.Decimal(str(asset_balance))
                latest_data["last_update_times"]["balance"] = time.time()
                verbose_print(
                    f"Initial Balance fetched successfully for {balance_asset}."
                )
            else:
                print_color(
                    f"# Initial Balance Warning: Asset '{balance_asset}' not found.",
                    Fore.YELLOW,
                )
                latest_data["balance"] = None
        elif isinstance(bal_res, Exception):
            print_color(f"# Initial Balance fetch failed: {bal_res}", Fore.YELLOW)
            bal_pos_fetch_ok = False
        # else: unexpected type

        # Process Positions
        if isinstance(pos_res, list):
            valid_positions = []
            for p in pos_res:
                try:
                    contracts_str = p.get("contracts")
                    if (
                        p.get("symbol") == symbol
                        and contracts_str
                        and decimal.Decimal(str(contracts_str)) != 0
                    ):
                        valid_positions.append(p)
                except (decimal.InvalidOperation, TypeError, KeyError):
                    pass  # Ignore invalid positions silently on init
            latest_data["positions"] = valid_positions
            latest_data["last_update_times"]["positions"] = time.time()
            verbose_print(
                f"Initial Positions fetched: Found {len(valid_positions)} relevant position(s)."
            )
        elif isinstance(pos_res, Exception):
            print_color(f"# Initial Positions fetch failed: {pos_res}", Fore.YELLOW)
            bal_pos_fetch_ok = False
        # else: unexpected type

    except Exception as e:
        print_color(f"# Error during initial balance/position fetch: {e}", Fore.RED)
        traceback.print_exc()
        bal_pos_fetch_ok = False

    if ohlcv_fetch_ok and bal_pos_fetch_ok:
        print_color("# Initial data fetch complete.", Fore.GREEN, Style.DIM)
    else:
        print_color(
            "# Initial data fetch partially failed. Check warnings/errors.", Fore.YELLOW
        )

    return True  # Indicate setup (market info fetch) succeeded, even if other fetches failed


# ==============================================================================
# Main Execution Function (Async)
# ==============================================================================
async def main_async() -> None:
    """Async main function: Initializes, starts background tasks, runs display/interaction loop."""
    global exchange, latest_data  # Allow modification of the global exchange instance

    # --- Print Header ---
    print_color("*" * 85, Fore.RED, Style.BRIGHT)
    print_color(
        "   🔥 Pyrmethus Market Analyzer v3.1 ASYNC Stable 🔥", Fore.RED, Style.BRIGHT
    )
    print_color("   Harnessing WebSockets & Asyncio. Use Responsibly!", Fore.YELLOW)
    print_color("*" * 85, Fore.RED, Style.BRIGHT)

    # --- Check API Keys ---
    if not CONFIG["API_KEY"] or not CONFIG["API_SECRET"]:
        print_color(
            "API Key and/or Secret not found in environment variables (.env).",
            Fore.RED,
            Style.BRIGHT,
        )
        print_color("Please set BYBIT_API_KEY and BYBIT_API_SECRET.", Fore.YELLOW)
        return  # Cannot proceed without credentials

    # --- Initialize Exchange ---
    print_color(
        f"{Fore.CYAN}# Binding async spirit to Bybit ({CONFIG['EXCHANGE_TYPE'].upper()})...{Style.DIM}"
    )
    try:
        exchange = ccxtpro.bybit(
            {
                "apiKey": CONFIG["API_KEY"],
                "secret": CONFIG["API_SECRET"],
                "options": {
                    "defaultType": CONFIG[
                        "EXCHANGE_TYPE"
                    ],  # e.g., 'linear', 'inverse', 'spot'
                    "adjustForTimeDifference": True,
                    # Add specific options if needed, e.g., hedge mode via API if supported
                },
                "enableRateLimit": True,
                "newUpdates": True,  # Use new unified WebSocket update format if available
                "timeout": CONFIG["CONNECT_TIMEOUT"],  # Connection timeout
            }
        )
        # Test connection asynchronously
        print_color(f"{Fore.CYAN}# Testing connection...{Style.DIM}", end="\r")
        await (
            exchange.fetch_time()
        )  # Simple async API call to test auth and connectivity
        sys.stdout.write("\033[K")  # Clear line
        print_color("Async connection established.", Fore.GREEN)
    except ccxt.AuthenticationError:
        sys.stdout.write("\033[K")
        print_color(
            "Authentication Failed! Check API Key and Secret.", Fore.RED, Style.BRIGHT
        )
        if exchange:
            await exchange.close()
        return
    except (TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
        sys.stdout.write("\033[K")
        print_color(f"Connection Failed (Network/Timeout): {e}", Fore.RED)
        if exchange:
            await exchange.close()
        return
    except Exception as e:
        sys.stdout.write("\033[K")
        print_color(f"Initialization Failed: {e}", Fore.RED)
        traceback.print_exc()
        if exchange:
            await exchange.close()
        return

    # --- Verify Symbol & Get Market Info ---
    # This now happens inside initial_setup_and_data_fetch
    # We need market_info before starting WS watchers that might depend on it.

    # --- Initial Data Fetch ---
    # This replaces the direct market_info call and subsequent fetches here
    setup_ok = await initial_setup_and_data_fetch(exchange, CONFIG["SYMBOL"], CONFIG)
    if not setup_ok:
        print_color("Initial setup failed. Exiting.", Fore.RED)
        await exchange.close()
        return

    # Re-extract validated market info and symbol after setup
    market_info = latest_data["market_info"]
    symbol = market_info["symbol"]  # Use the potentially validated symbol
    CONFIG["SYMBOL"] = (
        symbol  # Update config if symbol was changed during setup (though not implemented above)
    )

    print_color(f"Monitoring Market: {Fore.MAGENTA}{symbol}{Style.RESET_ALL}")
    print_color(
        f"  Precision - Price: {market_info['price_precision']}, Amount: {market_info['amount_precision']}"
    )
    print_color(
        f"  Limits    - Min Amount: {market_info['min_amount']}, Tick: {market_info['price_tick_size']}, Step: {market_info['amount_step']}"
    )
    print_color(
        f"  Trade Mode: {Fore.YELLOW}{CONFIG['DEFAULT_ORDER_TYPE'].capitalize()}{Style.RESET_ALL}"
        + (
            f" (Selection: {CONFIG['LIMIT_ORDER_SELECTION_TYPE']})"
            if CONFIG["DEFAULT_ORDER_TYPE"] == "limit"
            else ""
        )
        + (
            f" | Hedge Mode Index: {CONFIG['POSITION_IDX']}"
            if CONFIG["POSITION_IDX"] != 0
            else " | One-Way Mode"
        )
    )
    print_color(
        f"  Indicators TF: {CONFIG['INDICATOR_TIMEFRAME']}, Pivots TF: {CONFIG['PIVOT_TIMEFRAME']}"
    )

    # --- Start Background Tasks ---
    tasks = []
    try:
        # Start WebSocket watchers first
        tasks.append(
            asyncio.create_task(watch_ticker(exchange, symbol), name="TickerWatcher")
        )
        tasks.append(
            asyncio.create_task(
                watch_orderbook(exchange, symbol), name="OrderbookWatcher"
            )
        )
        # Start periodic REST fetcher
        tasks.append(
            asyncio.create_task(
                fetch_periodic_data(exchange, symbol), name="PeriodicDataFetcher"
            )
        )

        print_color(
            f"\n{Fore.CYAN}Starting background tasks. Press Ctrl+C to exit.{Style.RESET_ALL}"
        )

        # --- Main Display & Interaction Loop ---
        ask_map: dict[int, decimal.Decimal] = {}
        bid_map: dict[int, decimal.Decimal] = {}
        data_error_streak = 0
        last_input_was_error = False

        while True:
            # 1. Display Current State
            try:
                ask_map, bid_map = display_combined_analysis_async(
                    latest_data, market_info, CONFIG
                )
            except Exception as display_e:
                # Keep running even if display fails momentarily
                print_color(
                    "\n--- Display Error ---", color=Fore.RED, style=Style.BRIGHT
                )
                print_color(f"{display_e}", color=Fore.RED)
                traceback.print_exc()  # Print full traceback for debugging
                print_color(
                    "--- End Display Error ---", color=Fore.RED, style=Style.BRIGHT
                )

            # 2. Check Connection Status & Handle Potential Delays
            fetch_error_occurred = any(
                latest_data["connection_status"].get(k) == ERROR
                for k in CONN_STATUS_KEYS
            )

            if fetch_error_occurred:
                data_error_streak += 1
                # Exponential backoff for repeated errors, capped at 2 minutes
                wait_time = min(
                    CONFIG["RETRY_DELAY_NETWORK_ERROR"]
                    * (2 ** min(data_error_streak, 5)),
                    120,
                )
                print_color(
                    f"Connection/Fetch error detected. Waiting {wait_time}s (Streak: {data_error_streak})...",
                    Fore.YELLOW,
                    Style.DIM,
                )
                await asyncio.sleep(wait_time)
                continue  # Skip input prompt and retry display after delay
            elif data_error_streak > 0:
                verbose_print("Connection restored.", Fore.GREEN)
                data_error_streak = 0  # Reset streak on success

            # 3. Handle User Input (Non-blocking with Timeout)
            action_prompt = f"\n{Style.BRIGHT}{Fore.BLUE}Action ({Fore.GREEN}B{Style.RESET_ALL}uy/{Fore.RED}S{Style.RESET_ALL}ell/{Fore.YELLOW}R{Style.RESET_ALL}efresh/{Fore.CYAN}X{Style.RESET_ALL} Exit): {Style.RESET_ALL}"
            action = None
            try:
                # Wait for input, but timeout after REFRESH_INTERVAL to redraw the screen
                action = await asyncio.wait_for(
                    asyncio.to_thread(input, action_prompt),
                    timeout=CONFIG["REFRESH_INTERVAL"],
                )
                action = action.strip().lower()
                last_input_was_error = (
                    False  # Reset error flag on successful input read
                )

            except TimeoutError:
                action = "refresh"  # Timeout means auto-refresh
            except (EOFError, KeyboardInterrupt):
                print_color("\nInput interrupted.", Fore.YELLOW)
                action = "exit"
            except Exception as input_e:
                # Handle potential errors from input() itself, though less common
                if not last_input_was_error:  # Avoid spamming error messages
                    print_color(f"\nError reading input: {input_e}", Fore.RED)
                    last_input_was_error = True
                action = "refresh"  # Treat input error as refresh for now
                await asyncio.sleep(1)  # Short pause after input error

            # 4. Process Action
            if action in ["b", "buy"]:
                side = "buy"
                order_type = CONFIG["DEFAULT_ORDER_TYPE"]
                if order_type == "limit":
                    if CONFIG["LIMIT_ORDER_SELECTION_TYPE"] == "interactive":
                        await place_limit_order_interactive_async(
                            exchange,
                            symbol,
                            side,
                            ask_map,
                            bid_map,
                            market_info,
                            CONFIG,
                        )
                    else:  # Manual limit order
                        try:
                            price_str = await asyncio.to_thread(
                                input, f"Enter Limit Price ({side.upper()}): "
                            )
                            qty_str = await asyncio.to_thread(input, "Enter Quantity: ")
                            if price_str and qty_str:
                                await place_limit_order_async(
                                    exchange,
                                    symbol,
                                    side,
                                    qty_str,
                                    price_str,
                                    market_info,
                                    CONFIG,
                                )
                            else:
                                print_color(
                                    "Price and Quantity are required for manual limit order.",
                                    Fore.YELLOW,
                                )
                        except (EOFError, KeyboardInterrupt):
                            print_color("\nOrder input cancelled.", Fore.YELLOW)
                        except Exception as e:
                            print_color(f"\nOrder input error: {e}", Fore.RED)
                elif order_type == "market":
                    try:
                        qty_str = await asyncio.to_thread(
                            input,
                            f"Enter {Fore.GREEN}{side.upper()}{Style.RESET_ALL} Quantity: ",
                        )
                        if qty_str:
                            await place_market_order_async(
                                exchange, symbol, side, qty_str, market_info, CONFIG
                            )
                        else:
                            print_color(
                                "Quantity is required for market order.", Fore.YELLOW
                            )
                    except (EOFError, KeyboardInterrupt):
                        print_color("\nOrder input cancelled.", Fore.YELLOW)
                    except Exception as e:
                        print_color(f"\nOrder input error: {e}", Fore.RED)
                else:
                    print_color(
                        f"Unsupported DEFAULT_ORDER_TYPE: {order_type}", Fore.RED
                    )
                await asyncio.sleep(
                    1.5
                )  # Pause briefly after order attempt to allow user to see result

            elif action in ["s", "sell"]:
                side = "sell"
                order_type = CONFIG["DEFAULT_ORDER_TYPE"]
                if order_type == "limit":
                    if CONFIG["LIMIT_ORDER_SELECTION_TYPE"] == "interactive":
                        await place_limit_order_interactive_async(
                            exchange,
                            symbol,
                            side,
                            ask_map,
                            bid_map,
                            market_info,
                            CONFIG,
                        )
                    else:  # Manual limit order
                        try:
                            price_str = await asyncio.to_thread(
                                input, f"Enter Limit Price ({side.upper()}): "
                            )
                            qty_str = await asyncio.to_thread(input, "Enter Quantity: ")
                            if price_str and qty_str:
                                await place_limit_order_async(
                                    exchange,
                                    symbol,
                                    side,
                                    qty_str,
                                    price_str,
                                    market_info,
                                    CONFIG,
                                )
                            else:
                                print_color(
                                    "Price and Quantity are required for manual limit order.",
                                    Fore.YELLOW,
                                )
                        except (EOFError, KeyboardInterrupt):
                            print_color("\nOrder input cancelled.", Fore.YELLOW)
                        except Exception as e:
                            print_color(f"\nOrder input error: {e}", Fore.RED)
                elif order_type == "market":
                    try:
                        qty_str = await asyncio.to_thread(
                            input,
                            f"Enter {Fore.RED}{side.upper()}{Style.RESET_ALL} Quantity: ",
                        )
                        if qty_str:
                            await place_market_order_async(
                                exchange, symbol, side, qty_str, market_info, CONFIG
                            )
                        else:
                            print_color(
                                "Quantity is required for market order.", Fore.YELLOW
                            )
                    except (EOFError, KeyboardInterrupt):
                        print_color("\nOrder input cancelled.", Fore.YELLOW)
                    except Exception as e:
                        print_color(f"\nOrder input error: {e}", Fore.RED)
                else:
                    print_color(
                        f"Unsupported DEFAULT_ORDER_TYPE: {order_type}", Fore.RED
                    )
                await asyncio.sleep(1.5)  # Pause briefly

            elif action in ["r", "refresh", ""]:
                verbose_print("Refreshing display...", style=Style.DIM)
                # Loop will continue and redraw naturally
                # Add a tiny sleep to prevent tight loop if REFRESH_INTERVAL is very small
                await asyncio.sleep(0.05)

            elif action in ["x", "exit"]:
                print_color(
                    "Exit command received. Dispelling arcane energies...", Fore.YELLOW
                )
                break  # Exit the main loop

            elif (
                action is not None
            ):  # Only print if action was not None (i.e., not just a refresh timeout)
                print_color(
                    f"Unknown command: '{action}'. Options: b, s, r, x", Fore.YELLOW
                )
                await asyncio.sleep(1)  # Pause briefly on unknown command

    except KeyboardInterrupt:
        print_color("\nCtrl+C detected. Banishing spirits...", Fore.YELLOW)
    except ccxt.AuthenticationError:
        # This might happen if credentials expire or are revoked during runtime
        print_color(
            "\nAuthentication Error during operation! Halting.", Fore.RED, Style.BRIGHT
        )
    except Exception as e:
        print_color("\n--- Critical Error in Main Loop ---", Fore.RED, Style.BRIGHT)
        print_color(f"Error: {e}", Fore.RED)
        traceback.print_exc()
        print_color("--- End Critical Error ---", Fore.RED, Style.BRIGHT)
        termux_toast("Pyrmethus CRITICAL Loop Error", "long")

    finally:
        # --- Cleanup ---
        print_color(
            "\nClosing connections and cancelling background tasks...", Fore.YELLOW
        )
        # Cancel all running tasks
        for task in tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to finish cancelling
        # Capture cancellation errors silently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            verbose_print("Background tasks cancelled.")

        # Close the exchange connection
        if exchange and hasattr(exchange, "close"):
            try:
                await exchange.close()
                print_color("Exchange connection closed gracefully.", Fore.YELLOW)
            except Exception as close_e:
                print_color(f"Error closing exchange connection: {close_e}", Fore.RED)

        print_color(
            "Wizard Pyrmethus departs. Analysis complete.", Fore.MAGENTA, Style.BRIGHT
        )


# ==============================================================================
# Script Entry Point
# ==============================================================================
if __name__ == "__main__":
    try:
        # Run the main asynchronous function
        asyncio.run(main_async())
    except KeyboardInterrupt:
        # Handle Ctrl+C if pressed before the main loop's handler catches it
        print_color(
            "\nShutdown initiated by user (Ctrl+C detected at top level).", Fore.YELLOW
        )
    except Exception:
        # Catch any unexpected errors during asyncio.run() or setup
        traceback.print_exc()
    finally:
        pass
