# ==============================================================================
# 🔥 Pyrmethus's Arcane Market Analyzer v3.3 Ultimate ASYNC Edition 🔥
# ==============================================================================
# Harnessing WebSockets (ccxt.pro) for real-time updates & asyncio concurrency.
# Features: Enhanced UI, Paper Trading, TP/SL Orders, Trailing Stops, Robust Config/Error Handling.
# Woven with CCXT.pro, Colorama. Use with wisdom and manage risk.
# v3.3 - Fix Bybit V5 'category' parameter requirement for various endpoints.
# ==============================================================================
import asyncio
import decimal
import os
import subprocess
import sys
import time
import traceback
import uuid  # For unique order link IDs
from collections.abc import Callable
from typing import Any

import ccxt  # Keep standard ccxt for exceptions

# CCXT Pro for WebSocket and async REST support
import ccxt.pro as ccxtpro
from colorama import Back, Fore, Style, init
from dotenv import load_dotenv

# Initialize Colorama & Decimal Precision
init(autoreset=True)
decimal.getcontext().prec = 50

# Load .env
load_dotenv()


# ==============================================================================
# Configuration Loading (Robust)
# ==============================================================================
def get_config_value(key: str, default: Any, cast_type: Callable = str) -> Any:
    """Gets value from environment or uses default, casting to specified type."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        if cast_type == bool:
            if value.lower() in ("true", "1", "yes", "y"):
                return True
            if value.lower() in ("false", "0", "no", "n"):
                return False
            raise ValueError(f"Invalid boolean string: {value}")
        if cast_type == decimal.Decimal:
            return decimal.Decimal(value)
        return cast_type(value)
    except (ValueError, TypeError, decimal.InvalidOperation):
        return default


CONFIG = {
    # --- Core ---
    "API_KEY": get_config_value("BYBIT_API_KEY", None),
    "API_SECRET": get_config_value("BYBIT_API_SECRET", None),
    "VERBOSE_DEBUG": get_config_value("VERBOSE_DEBUG", False, bool),
    "PAPER_TRADING_ENABLED": get_config_value(
        "PAPER_TRADING_ENABLED", False, bool
    ),  # Paper Trading Mode
    # --- Market & Order Book ---
    "DEFAULT_SYMBOL": get_config_value(
        "BYBIT_SYMBOL", "BTC/USDT:USDT", str
    ).upper(),  # Default Symbol from Config
    "EXCHANGE_TYPE": get_config_value(
        "BYBIT_EXCHANGE_TYPE", "linear", str
    ),  # linear, inverse, or option (use linear/inverse for futures/perps)
    "VOLUME_THRESHOLDS": {
        "high": get_config_value(
            "VOLUME_THRESHOLD_HIGH", decimal.Decimal("10"), decimal.Decimal
        ),
        "medium": get_config_value(
            "VOLUME_THRESHOLD_MEDIUM", decimal.Decimal("2"), decimal.Decimal
        ),
    },
    "REFRESH_INTERVAL": get_config_value(
        "REFRESH_INTERVAL", 5, int
    ),  # Display refresh rate (lower is ok with WS)
    "MAX_ORDERBOOK_DEPTH_DISPLAY": get_config_value(
        "MAX_ORDERBOOK_DEPTH_DISPLAY", 30, int
    ),
    "ORDER_FETCH_LIMIT": get_config_value(
        "ORDER_FETCH_LIMIT", 50, int
    ),  # WS orderbook depth
    "CONNECT_TIMEOUT": get_config_value("CONNECT_TIMEOUT", 35000, int),
    "RETRY_DELAY_NETWORK_ERROR": get_config_value("RETRY_DELAY_NETWORK_ERROR", 10, int),
    "RETRY_DELAY_RATE_LIMIT": get_config_value("RETRY_DELAY_RATE_LIMIT", 60, int),
    # --- Indicators ---
    "INDICATOR_TIMEFRAME": get_config_value("INDICATOR_TIMEFRAME", "15m", str),
    "SMA_PERIOD": get_config_value("SMA_PERIOD", 9, int),
    "SMA2_PERIOD": get_config_value("SMA2_PERIOD", 50, int),
    "EMA1_PERIOD": get_config_value("EMA1_PERIOD", 12, int),
    "EMA2_PERIOD": get_config_value("EMA2_PERIOD", 89, int),
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
    "MIN_OHLCV_CANDLES": max(
        get_config_value("SMA_PERIOD", 9, int),
        get_config_value("SMA2_PERIOD", 50, int),
        get_config_value("EMA1_PERIOD", 12, int),
        get_config_value("EMA2_PERIOD", 89, int),
        get_config_value("MOMENTUM_PERIOD", 10, int) + 1,
        get_config_value("RSI_PERIOD", 14, int)
        + get_config_value("STOCH_K_PERIOD", 14, int)
        + get_config_value("STOCH_D_PERIOD", 3, int),
    )
    + 5,  # Fetch buffer
    # --- Display ---
    "PIVOT_TIMEFRAME": get_config_value(
        "PIVOT_TIMEFRAME", "1d", str
    ),  # Default Daily Pivots
    "PNL_PRECISION": get_config_value("PNL_PRECISION", 2, int),
    "MIN_PRICE_DISPLAY_PRECISION": get_config_value(
        "MIN_PRICE_DISPLAY_PRECISION", 3, int
    ),
    "STOCH_RSI_DISPLAY_PRECISION": get_config_value(
        "STOCH_RSI_DISPLAY_PRECISION", 2, int
    ),
    "VOLUME_DISPLAY_PRECISION": get_config_value("VOLUME_DISPLAY_PRECISION", 2, int),
    "BALANCE_DISPLAY_PRECISION": get_config_value("BALANCE_DISPLAY_PRECISION", 2, int),
    # --- Trading ---
    "FETCH_BALANCE_ASSET": get_config_value(
        "FETCH_BALANCE_ASSET", "USDT", str
    ),  # Asset to display balance in
    "DEFAULT_ORDER_TYPE": get_config_value("DEFAULT_ORDER_TYPE", "limit", str).lower(),
    "LIMIT_ORDER_SELECTION_TYPE": get_config_value(
        "LIMIT_ORDER_SELECTION_TYPE", "interactive", str
    ).lower(),
    "POSITION_IDX": get_config_value(
        "BYBIT_POSITION_IDX", 0, int
    ),  # For Hedge Mode (0=One-Way, 1=Buy Hedge, 2=Sell Hedge)
    "PAPER_INITIAL_BALANCE": get_config_value(
        "PAPER_INITIAL_BALANCE", 10000, int
    ),  # Initial balance for paper trading
    "ADD_TP_SL_TO_ORDERS": get_config_value(
        "ADD_TP_SL_TO_ORDERS", True, bool
    ),  # Prompt for TP/SL on basic orders
    "SL_TRIGGER_TYPE": get_config_value(
        "SL_TRIGGER_TYPE", "MarkPrice", str
    ),  # MarkPrice, LastPrice, IndexPrice
    "TP_TRIGGER_TYPE": get_config_value(
        "TP_TRIGGER_TYPE", "MarkPrice", str
    ),  # MarkPrice, LastPrice, IndexPrice
    # --- Intervals ---
    "BALANCE_POS_FETCH_INTERVAL": get_config_value(
        "BALANCE_POS_FETCH_INTERVAL", 45, int
    ),  # Fetch balance/pos less often
    "OHLCV_FETCH_INTERVAL": get_config_value(
        "OHLCV_FETCH_INTERVAL", 300, int
    ),  # Fetch history every 5 mins default
    "PAPER_FILL_CHECK_INTERVAL": get_config_value(
        "PAPER_FILL_CHECK_INTERVAL", 3, int
    ),  # Check paper fills every 3 seconds
}
# Initialize symbol at runtime
CONFIG["SYMBOL"] = CONFIG["DEFAULT_SYMBOL"]

# Fibonacci Ratios
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
latest_data: dict[str, Any] = {
    "ticker": None,
    "orderbook": None,
    "balance": None,
    "positions": [],
    "open_orders": [],
    "indicator_ohlcv": None,
    "pivot_ohlcv": None,
    "indicators": {},
    "pivots": None,
    "market_info": None,
    "last_update_times": {},
    "connection_status": {"ws_ticker": "init", "ws_ob": "init", "rest": "init"},
}
exchange: ccxtpro.Exchange | None = None  # Global exchange instance
paper_balance = decimal.Decimal(
    str(CONFIG["PAPER_INITIAL_BALANCE"])
)  # Paper trading balance
paper_positions: dict[
    str, dict
] = {}  # Paper trading positions: {symbol: {side: 'long/short', size: Decimal, entry_price: Decimal}}
paper_orders: list[dict] = []  # Paper trading open orders
paper_trade_log: list[str] = []  # Log of paper trades


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
    """Prints colorized text."""


def verbose_print(text: str, color: str = Fore.CYAN, style: str = Style.DIM) -> None:
    """Prints only if VERBOSE_DEBUG is True."""
    if CONFIG.get("VERBOSE_DEBUG", False):
        print_color(f"# DEBUG: {text}", color=color, style=style)


def termux_toast(message: str, duration: str = "short") -> None:
    """Displays a Termux toast notification."""
    try:
        safe_message = "".join(
            c for c in str(message) if c.isalnum() or c in " .,!?-:+%$/=()[]{}"
        )[:100]
        subprocess.run(
            ["termux-toast", "-d", duration, safe_message],
            check=True,
            capture_output=True,
            timeout=5,
        )
    except FileNotFoundError:
        pass
    except Exception as e:
        verbose_print(f"Toast error: {e}")


def format_decimal(
    value: decimal.Decimal | str | int | float | None,
    reported_precision: int,
    min_display_precision: int | None = None,
) -> str:
    """Formats decimal values for display."""
    if value is None:
        return "N/A"
    try:
        d_value = (
            decimal.Decimal(str(value))
            if not isinstance(value, decimal.Decimal)
            else value
        )
        display_precision = max(int(reported_precision), 0)
        if min_display_precision is not None:
            display_precision = max(
                display_precision, max(int(min_display_precision), 0)
            )
        quantizer = decimal.Decimal("1") / (decimal.Decimal("10") ** display_precision)
        rounded_value = d_value.quantize(quantizer, rounding=decimal.ROUND_HALF_UP)
        formatted_str = f"{rounded_value:f}"
        if "." in formatted_str:
            integer_part, decimal_part = formatted_str.split(".")
            decimal_part = decimal_part[:display_precision].ljust(
                display_precision, "0"
            )
            formatted_str = (
                f"{integer_part}.{decimal_part}"
                if display_precision > 0
                else integer_part
            )
        elif display_precision > 0:
            formatted_str += "." + "0" * display_precision
        return formatted_str
    except Exception as e:
        verbose_print(f"FormatDecimal Error: {e}")
        return str(value)


def generate_order_link_id() -> str:
    """Generates a unique order link ID."""
    return f"pyrm-{uuid.uuid4().hex[:12]}"


# ==============================================================================
# Async Market Info Fetcher
# ==============================================================================
async def get_market_info(
    exchange_instance: ccxtpro.Exchange, symbol: str
) -> dict[str, Any] | None:
    """ASYNCHRONOUSLY Fetches and returns market information."""
    try:
        print_color(
            f"{Fore.CYAN}# Querying market runes for {symbol} (async)...",
            style=Style.DIM,
            end="\r",
        )
        if not exchange_instance.markets or symbol not in exchange_instance.markets:
            verbose_print(f"Market list needs loading/refresh for {symbol}.")
            await exchange_instance.load_markets(True)  # Use await here!
        sys.stdout.write("\033[K")
        if symbol not in exchange_instance.markets:
            print_color(
                f"Symbol '{symbol}' still not found after async market reload.",
                color=Fore.RED,
                style=Style.BRIGHT,
            )
            return None
        market = exchange_instance.market(symbol)
        verbose_print(f"Async market info retrieved for {symbol}")
        price_prec = 8
        amount_prec = 8
        min_amount = decimal.Decimal("0")
        try:
            price_prec = int(
                decimal.Decimal(
                    str(market.get("precision", {}).get("price", "1e-8"))
                ).log10()
                * -1
            )
        except:
            verbose_print(
                f"Could not parse price precision for {symbol}, using default {price_prec}"
            )
        try:
            amount_prec = int(
                decimal.Decimal(
                    str(market.get("precision", {}).get("amount", "1e-8"))
                ).log10()
                * -1
            )
        except:
            verbose_print(
                f"Could not parse amount precision for {symbol}, using default {amount_prec}"
            )
        try:
            min_amount = decimal.Decimal(
                str(market.get("limits", {}).get("amount", {}).get("min", "0"))
            )
        except:
            verbose_print(
                f"Could not parse min amount for {symbol}, using default {min_amount}"
            )
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
        return {
            "price_precision": price_prec,
            "amount_precision": amount_prec,
            "min_amount": min_amount,
            "price_tick_size": price_tick_size,
            "amount_step": amount_step,
            "symbol": symbol,
        }
    except ccxt.BadSymbol:
        sys.stdout.write("\033[K")
        print_color(f"Symbol '{symbol}' invalid.", color=Fore.RED, style=Style.BRIGHT)
        return None
    except ccxt.NetworkError as e:
        sys.stdout.write("\033[K")
        print_color(
            f"Network error getting market info (async): {e}", color=Fore.YELLOW
        )
        return None
    except Exception as e:
        sys.stdout.write("\033[K")
        print_color(f"Critical error getting market info (async): {e}", color=Fore.RED)
        traceback.print_exc()
        return None


# ==============================================================================
# Indicator Calculation Functions
# ==============================================================================
def calculate_sma(
    data: list[str | float | int | decimal.Decimal], period: int
) -> decimal.Decimal | None:
    if not data or len(data) < period:
        return None
    try:
        return sum(decimal.Decimal(str(p)) for p in data[-period:]) / decimal.Decimal(
            period
        )
    except Exception as e:
        verbose_print(f"SMA Calc Error: {e}")
        return None


def calculate_ema(
    data: list[str | float | int | decimal.Decimal], period: int
) -> list[decimal.Decimal] | None:
    if not data or len(data) < period:
        return None
    try:
        decimal_data = [decimal.Decimal(str(p)) for p in data]
        ema_values: list[decimal.Decimal | None] = [None] * len(data)
        mult = decimal.Decimal(2) / (decimal.Decimal(period) + 1)
        sma_init = sum(decimal_data[:period]) / decimal.Decimal(period)
        ema_values[period - 1] = sma_init
        for i in range(period, len(data)):
            if ema_values[i - 1] is None:
                continue
            ema_values[i] = (decimal_data[i] - ema_values[i - 1]) * mult + ema_values[
                i - 1
            ]
        return [ema for ema in ema_values if ema is not None]
    except Exception as e:
        verbose_print(f"EMA Calc Error: {e}")
        return None


def calculate_momentum(
    data: list[str | float | int | decimal.Decimal], period: int
) -> decimal.Decimal | None:
    if not data or len(data) <= period:
        return None
    try:
        return decimal.Decimal(str(data[-1])) - decimal.Decimal(str(data[-period - 1]))
    except Exception as e:
        verbose_print(f"Momentum Calc Error: {e}")
        return None


def calculate_fib_pivots(
    high: Any | None, low: Any | None, close: Any | None
) -> dict[str, decimal.Decimal] | None:
    if None in [high, low, close]:
        return None
    try:
        h, l, c = (
            decimal.Decimal(str(high)),
            decimal.Decimal(str(low)),
            decimal.Decimal(str(close)),
        )
        if h <= 0 or l <= 0 or c <= 0 or h < l:
            return None
        pp = (h + l + c) / 3
        range_hl = max(h - l, decimal.Decimal("0"))
        return {
            "R3": pp + (range_hl * FIB_RATIOS["r3"]),
            "R2": pp + (range_hl * FIB_RATIOS["r2"]),
            "R1": pp + (range_hl * FIB_RATIOS["r1"]),
            "PP": pp,
            "S1": pp - (range_hl * FIB_RATIOS["s1"]),
            "S2": pp - (range_hl * FIB_RATIOS["s2"]),
            "S3": pp - (range_hl * FIB_RATIOS["s3"]),
        }
    except Exception as e:
        verbose_print(f"Pivot Calc Error: {e}")
        return None


def calculate_rsi_manual(
    close_prices_list: list[Any], period: int = 14
) -> tuple[list[decimal.Decimal] | None, str | None]:
    if not close_prices_list or len(close_prices_list) <= period:
        return None, f"Data short ({len(close_prices_list)}<{period + 1})"
    try:
        prices = [decimal.Decimal(str(p)) for p in close_prices_list]
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        if not deltas:
            return None, "No changes"
        gains = [d if d > 0 else decimal.Decimal("0") for d in deltas]
        losses = [-d if d < 0 else decimal.Decimal("0") for d in deltas]
        if len(gains) < period:
            return None, f"Deltas short ({len(gains)}<{period})"
        avg_gain, avg_loss = (
            sum(gains[:period]) / decimal.Decimal(period),
            sum(losses[:period]) / decimal.Decimal(period),
        )
        rsi_values = []
        rs = decimal.Decimal("inf") if avg_loss == 0 else avg_gain / avg_loss
        rsi_values.append(
            100 - (100 / (1 + rs)) if rs.is_finite() else decimal.Decimal("100")
        )
        for i in range(period, len(gains)):
            avg_gain, avg_loss = (
                (avg_gain * (period - 1) + gains[i]) / decimal.Decimal(period),
                (avg_loss * (period - 1) + losses[i]) / decimal.Decimal(period),
            )
            rs = decimal.Decimal("inf") if avg_loss == 0 else avg_gain / avg_loss
            rsi_values.append(
                100 - (100 / (1 + rs)) if rs.is_finite() else decimal.Decimal("100")
            )
        return rsi_values, None
    except Exception as e:
        verbose_print(f"RSI Calc Error: {e}")
        return None, str(e)


def calculate_stoch_rsi_manual(
    rsi_values: list[decimal.Decimal], k_period: int = 14, d_period: int = 3
) -> tuple[decimal.Decimal | None, decimal.Decimal | None, str | None]:
    if not rsi_values or len(rsi_values) < k_period:
        return None, None, f"RSI short ({len(rsi_values)}<{k_period})"
    try:
        valid_rsi = [r for r in rsi_values if r is not None and r.is_finite()]
        if len(valid_rsi) < k_period:
            return None, None, f"Valid RSI short ({len(valid_rsi)}<{k_period})"
        stoch_k_values = []
        for i in range(k_period - 1, len(valid_rsi)):
            window = valid_rsi[i - k_period + 1 : i + 1]
            curr, mini, maxi = window[-1], min(window), max(window)
            stoch_k = (
                decimal.Decimal("50")
                if maxi == mini
                else max(
                    decimal.Decimal("0"),
                    min(decimal.Decimal("100"), ((curr - mini) / (maxi - mini)) * 100),
                )
            )
            stoch_k_values.append(stoch_k)
        if not stoch_k_values:
            return None, None, "%K empty"
        latest_k = stoch_k_values[-1]
        if len(stoch_k_values) < d_period:
            return latest_k, None, f"%K short ({len(stoch_k_values)}<{d_period})"
        latest_d = sum(stoch_k_values[-d_period:]) / decimal.Decimal(d_period)
        return latest_k, latest_d, None
    except Exception as e:
        verbose_print(f"StochRSI Calc Error: {e}")
        return None, None, str(e)


# ==============================================================================
# Data Processing & Analysis
# ==============================================================================
def analyze_orderbook_data(
    orderbook: dict, market_info: dict, config: dict
) -> dict | None:
    if (
        not orderbook
        or not isinstance(orderbook.get("bids"), list)
        or not isinstance(orderbook.get("asks"), list)
    ):
        return None
    _price_prec, amount_prec, vol_disp_prec = (
        market_info["price_precision"],
        market_info["amount_precision"],
        config["VOLUME_DISPLAY_PRECISION"],
    )
    vol_thr, display_depth = (
        config["VOLUME_THRESHOLDS"],
        config["MAX_ORDERBOOK_DEPTH_DISPLAY"],
    )
    analyzed_ob = {
        "symbol": orderbook.get("symbol", market_info["symbol"]),
        "timestamp": orderbook.get(
            "datetime",
            time.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ",
                time.gmtime(orderbook.get("timestamp", time.time() * 1000) / 1000),
            ),
        ),
        "asks": [],
        "bids": [],
        "ask_total_volume_fetched": decimal.Decimal("0"),
        "bid_total_volume_fetched": decimal.Decimal("0"),
        "ask_vwap_fetched": decimal.Decimal("0"),
        "bid_vwap_fetched": decimal.Decimal("0"),
        "volume_imbalance_ratio_fetched": decimal.Decimal("0"),
        "cumulative_ask_volume_displayed": decimal.Decimal("0"),
        "cumulative_bid_volume_displayed": decimal.Decimal("0"),
    }
    ask_sum = decimal.Decimal("0")
    for i, level in enumerate(orderbook["asks"]):
        try:
            price, volume = (
                decimal.Decimal(str(level[0])),
                decimal.Decimal(str(level[1])),
            )
        except:
            continue
        analyzed_ob["ask_total_volume_fetched"] += volume
        ask_sum += price * volume
        if i < display_depth:
            analyzed_ob["cumulative_ask_volume_displayed"] += volume
            vol_str = format_decimal(volume, amount_prec, vol_disp_prec)
            color, style = (
                (Fore.LIGHTRED_EX, Style.BRIGHT)
                if volume >= vol_thr["high"]
                else (Fore.RED, Style.NORMAL)
                if volume >= vol_thr["medium"]
                else (Fore.WHITE, Style.NORMAL)
            )
            analyzed_ob["asks"].append(
                {
                    "price": price,
                    "volume": volume,
                    "volume_str": vol_str,
                    "color": color,
                    "style": style,
                    "cumulative_volume": format_decimal(
                        analyzed_ob["cumulative_ask_volume_displayed"],
                        amount_prec,
                        vol_disp_prec,
                    ),
                }
            )
    bid_sum = decimal.Decimal("0")
    for i, level in enumerate(orderbook["bids"]):
        try:
            price, volume = (
                decimal.Decimal(str(level[0])),
                decimal.Decimal(str(level[1])),
            )
        except:
            continue
        analyzed_ob["bid_total_volume_fetched"] += volume
        bid_sum += price * volume
        if i < display_depth:
            analyzed_ob["cumulative_bid_volume_displayed"] += volume
            vol_str = format_decimal(volume, amount_prec, vol_disp_prec)
            color, style = (
                (Fore.LIGHTGREEN_EX, Style.BRIGHT)
                if volume >= vol_thr["high"]
                else (Fore.GREEN, Style.NORMAL)
                if volume >= vol_thr["medium"]
                else (Fore.WHITE, Style.NORMAL)
            )
            analyzed_ob["bids"].append(
                {
                    "price": price,
                    "volume": volume,
                    "volume_str": vol_str,
                    "color": color,
                    "style": style,
                    "cumulative_volume": format_decimal(
                        analyzed_ob["cumulative_bid_volume_displayed"],
                        amount_prec,
                        vol_disp_prec,
                    ),
                }
            )
    ask_tot, bid_tot = (
        analyzed_ob["ask_total_volume_fetched"],
        analyzed_ob["bid_total_volume_fetched"],
    )
    if ask_tot > 0:
        (
            analyzed_ob["ask_vwap_fetched"],
            analyzed_ob["volume_imbalance_ratio_fetched"],
        ) = (
            ask_sum / ask_tot,
            bid_tot / ask_tot,
        )
    else:
        analyzed_ob["volume_imbalance_ratio_fetched"] = (
            decimal.Decimal("inf") if bid_tot > 0 else decimal.Decimal("0")
        )
    if bid_tot > 0:
        analyzed_ob["bid_vwap_fetched"] = bid_sum / bid_tot
    return analyzed_ob


# ==============================================================================
# WebSocket Watcher Tasks
# ==============================================================================
async def watch_ticker(exchange_pro: ccxtpro.Exchange, symbol: str) -> None:
    verbose_print(f"Starting ticker watcher for {symbol}")
    latest_data["connection_status"]["ws_ticker"] = "connecting"
    while True:
        try:
            ticker = await exchange_pro.watch_ticker(symbol)
            latest_data["ticker"] = ticker
            latest_data["last_update_times"]["ticker"] = time.time()
            if latest_data["connection_status"]["ws_ticker"] != "ok":
                latest_data["connection_status"]["ws_ticker"] = "ok"
                verbose_print(f"Ticker WS connected for {symbol}")
        except (TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
            print_color(f"# Ticker WS Net Err: {e}", Fore.YELLOW)
            latest_data["connection_status"]["ws_ticker"] = "error"
            await asyncio.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"])
        except ccxt.ExchangeError as e:
            print_color(f"# Ticker WS Exch Err: {e}", Fore.RED)
            latest_data["connection_status"]["ws_ticker"] = "error"
            await asyncio.sleep(CONFIG["RETRY_DELAY_RATE_LIMIT"])
        except Exception as e:
            print_color(f"# Ticker WS Error: {e}", Fore.RED, style=Style.BRIGHT)
            latest_data["connection_status"]["ws_ticker"] = "error"
            traceback.print_exc()
            await asyncio.sleep(30)


async def watch_orderbook(exchange_pro: ccxtpro.Exchange, symbol: str) -> None:
    verbose_print(f"Starting orderbook watcher for {symbol}")
    latest_data["connection_status"]["ws_ob"] = "connecting"
    while True:
        try:
            orderbook = await exchange_pro.watch_order_book(
                symbol, limit=CONFIG["ORDER_FETCH_LIMIT"]
            )
            market_info = latest_data.get("market_info")
            if market_info:
                analyzed_ob = analyze_orderbook_data(orderbook, market_info, CONFIG)
                if analyzed_ob:
                    latest_data["orderbook"] = analyzed_ob
                    latest_data["last_update_times"]["orderbook"] = time.time()
                    if latest_data["connection_status"]["ws_ob"] != "ok":
                        latest_data["connection_status"]["ws_ob"] = "ok"
                        verbose_print(f"OrderBook WS connected for {symbol}")
            else:
                await asyncio.sleep(0.5)  # Wait for market_info if not ready
        except (TimeoutError, ccxt.NetworkError, ccxt.RequestTimeout) as e:
            print_color(f"# OB WS Net Err: {e}", Fore.YELLOW)
            latest_data["connection_status"]["ws_ob"] = "error"
            await asyncio.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"])
        except ccxt.ExchangeError as e:
            print_color(f"# OB WS Exch Err: {e}", Fore.RED)
            latest_data["connection_status"]["ws_ob"] = "error"
            await asyncio.sleep(CONFIG["RETRY_DELAY_RATE_LIMIT"])
        except Exception as e:
            print_color(f"# OB WS Error: {e}", Fore.RED, style=Style.BRIGHT)
            latest_data["connection_status"]["ws_ob"] = "error"
            traceback.print_exc()
            await asyncio.sleep(30)


# ==============================================================================
# Periodic Data Fetching Task (REST via ccxt.pro async methods)
# ==============================================================================
async def fetch_periodic_data(exchange_pro: ccxtpro.Exchange, symbol: str) -> None:
    bal_pos_interval = CONFIG["BALANCE_POS_FETCH_INTERVAL"]
    ohlcv_interval = CONFIG["OHLCV_FETCH_INTERVAL"]
    last_ohlcv_fetch_time = 0
    min_ohlcv = CONFIG["MIN_OHLCV_CANDLES"]
    ind_tf, piv_tf = CONFIG["INDICATOR_TIMEFRAME"], CONFIG["PIVOT_TIMEFRAME"]
    global paper_balance, paper_positions, paper_orders

    # --- Bybit V5 requires explicit category ---
    fetch_params = {"category": CONFIG["EXCHANGE_TYPE"]}
    verbose_print(f"Using fetch params: {fetch_params} for periodic REST calls.")

    while True:
        now = time.time()
        fetch_live_data = not CONFIG["PAPER_TRADING_ENABLED"]
        fetch_ohlcv = (now - last_ohlcv_fetch_time) >= ohlcv_interval

        tasks_to_run = {}
        if fetch_live_data:
            # Pass category parameter explicitly for Bybit V5
            tasks_to_run["balance"] = (
                exchange_pro.fetch_balance()
            )  # Balance often works without category, check if needed
            tasks_to_run["positions"] = exchange_pro.fetch_positions(
                [symbol], params=fetch_params
            )
            tasks_to_run["open_orders"] = exchange_pro.fetch_open_orders(
                symbol, params=fetch_params
            )
        if fetch_ohlcv:
            history_needed = min_ohlcv + 5
            tasks_to_run["indicator_ohlcv"] = exchange_pro.fetch_ohlcv(
                symbol, ind_tf, limit=history_needed
            )
            tasks_to_run["pivot_ohlcv"] = exchange_pro.fetch_ohlcv(
                symbol, piv_tf, limit=2
            )

        if not tasks_to_run and not CONFIG["PAPER_TRADING_ENABLED"]:
            await asyncio.sleep(bal_pos_interval)
            continue

        if tasks_to_run:
            verbose_print(f"Running periodic fetches: {list(tasks_to_run.keys())}")
            try:
                results = await asyncio.gather(
                    *tasks_to_run.values(), return_exceptions=True
                )
                latest_data["connection_status"]["rest"] = "ok"

                res_map = dict(zip(tasks_to_run.keys(), results, strict=False))

                if "balance" in res_map and fetch_live_data:
                    bal_res = res_map["balance"]
                    if isinstance(bal_res, dict):
                        # Find the correct balance based on asset and category if needed (Bybit response structure can vary)
                        CONFIG["EXCHANGE_TYPE"].upper()  # e.g., LINEAR, INVERSE
                        # Try common Bybit V5 structure first
                        total_balance = (
                            bal_res.get("result", {})
                            .get("list", [{}])[0]
                            .get("totalEquity")
                        )
                        asset_balance = (
                            bal_res.get("result", {})
                            .get("list", [{}])[0]
                            .get("coin", [{}])[0]
                            .get("walletBalance")
                            if CONFIG["FETCH_BALANCE_ASSET"] in str(bal_res)
                            else None
                        )

                        # Fallback to older structure or ccxt unified structure
                        if total_balance is None and asset_balance is None:
                            total_balance = bal_res.get("total", {}).get(
                                CONFIG["FETCH_BALANCE_ASSET"]
                            )

                        # Decide which balance to use (total equity or specific asset)
                        display_balance = (
                            asset_balance
                            if asset_balance is not None
                            else total_balance
                        )

                        latest_data["balance"] = display_balance
                        latest_data["last_update_times"]["balance"] = time.time()
                        verbose_print(
                            f"Balance fetched: Total={total_balance}, Asset={asset_balance}, Display={display_balance}"
                        )
                    elif isinstance(bal_res, Exception):
                        print_color(f"# Err fetch balance: {bal_res}", Fore.YELLOW)
                        latest_data["connection_status"]["rest"] = "error"
                        # Check if the error is specifically about category for balance fetch
                        if "category" in str(bal_res):
                            print_color(
                                f"{Fore.YELLOW}# Balance fetch might also require 'category'. Consider adding it if issues persist.{Style.RESET_ALL}"
                            )

                if "positions" in res_map and fetch_live_data:
                    pos_res = res_map["positions"]
                    if isinstance(pos_res, list):
                        # Filter positions matching the current symbol and non-zero size
                        latest_data["positions"] = [
                            p
                            for p in pos_res
                            if p.get("symbol") == symbol
                            and p.get("contracts") is not None
                            and decimal.Decimal(str(p["contracts"])) != 0
                        ]
                        latest_data["last_update_times"]["positions"] = time.time()
                        verbose_print(
                            f"Positions fetched: {len(latest_data['positions'])} active for {symbol}"
                        )
                    elif isinstance(pos_res, Exception):
                        print_color(f"# Err fetch positions: {pos_res}", Fore.YELLOW)
                        latest_data["connection_status"]["rest"] = "error"

                if "open_orders" in res_map and fetch_live_data:
                    ord_res = res_map["open_orders"]
                    if isinstance(ord_res, list):
                        latest_data["open_orders"] = ord_res
                        latest_data["last_update_times"]["open_orders"] = time.time()
                        verbose_print(
                            f"Open orders fetched: {len(latest_data['open_orders'])} for {symbol}"
                        )
                    elif isinstance(ord_res, Exception):
                        print_color(f"# Err fetch open orders: {ord_res}", Fore.YELLOW)
                        latest_data["connection_status"]["rest"] = "error"

                if fetch_ohlcv:
                    last_ohlcv_fetch_time = now
                    ind_res = res_map.get("indicator_ohlcv")
                    if isinstance(ind_res, list):
                        if len(ind_res) >= min_ohlcv:
                            latest_data["indicator_ohlcv"] = ind_res
                            latest_data["last_update_times"]["indicator_ohlcv"] = (
                                time.time()
                            )
                            verbose_print(f"Ind OHLCV updated ({len(ind_res)})")
                            await (
                                calculate_and_store_indicators()
                            )  # Recalculate after new data
                        else:
                            print_color(
                                f"# Warn: Insufficient Ind OHLCV ({len(ind_res)}<{min_ohlcv})",
                                Fore.YELLOW,
                            )
                            latest_data["indicator_ohlcv"] = None
                    elif isinstance(ind_res, Exception):
                        print_color(f"# Err fetch ind OHLCV: {ind_res}", Fore.YELLOW)
                        latest_data["connection_status"]["rest"] = "error"
                        latest_data["indicator_ohlcv"] = None

                    piv_res = res_map.get("pivot_ohlcv")
                    if isinstance(piv_res, list) and len(piv_res) > 0:
                        latest_data["pivot_ohlcv"] = piv_res
                        latest_data["last_update_times"]["pivot_ohlcv"] = time.time()
                        verbose_print(f"Piv OHLCV updated ({len(piv_res)})")
                        await calculate_and_store_pivots()  # Recalculate after new data
                    elif isinstance(piv_res, Exception):
                        print_color(f"# Err fetch piv OHLCV: {piv_res}", Fore.YELLOW)
                        latest_data["connection_status"]["rest"] = "error"
                        latest_data["pivots"] = None
                    elif not isinstance(piv_res, Exception) and len(piv_res) == 0:
                        print_color(
                            f"# Warn: Pivot OHLCV fetch returned empty list for {symbol} {piv_tf}",
                            Fore.YELLOW,
                        )
                        latest_data["pivots"] = (
                            None  # Ensure pivots cleared if fetch empty
                        )
                    elif not isinstance(piv_res, Exception):
                        latest_data["pivots"] = None  # Catch other non-exception cases

            except Exception as e:
                print_color(f"# Error in periodic gather: {e}", Fore.RED)
                latest_data["connection_status"]["rest"] = "error"
                traceback.print_exc()

        if CONFIG["PAPER_TRADING_ENABLED"]:
            # Update paper data representations
            latest_data["balance"] = paper_balance
            latest_data["last_update_times"]["balance"] = time.time()
            latest_data["open_orders"] = paper_orders
            latest_data["last_update_times"]["open_orders"] = time.time()
            current_symbol_pos = paper_positions.get(symbol)
            if current_symbol_pos:
                # Calculate paper PNL using latest ticker price if available
                ticker = latest_data.get("ticker")
                last_price = decimal.Decimal("0")
                if ticker and ticker.get("last"):
                    last_price = decimal.Decimal(str(ticker["last"]))
                elif current_symbol_pos["entry_price"]:
                    last_price = current_symbol_pos[
                        "entry_price"
                    ]  # Fallback if no ticker yet

                pnl = decimal.Decimal("0")
                if (
                    last_price > 0
                    and current_symbol_pos.get("entry_price") is not None
                    and current_symbol_pos.get("size") is not None
                ):
                    entry_price = current_symbol_pos["entry_price"]
                    size = current_symbol_pos["size"]
                    if current_symbol_pos["side"] == "long":
                        pnl = (last_price - entry_price) * size
                    elif current_symbol_pos["side"] == "short":
                        pnl = (entry_price - last_price) * size

                latest_data["positions"] = [
                    {
                        "symbol": symbol,
                        "side": current_symbol_pos[
                            "side"
                        ].capitalize(),  # CCXT usually returns lowercase 'long'/'short'
                        "contracts": str(current_symbol_pos["size"]),
                        "entryPrice": str(current_symbol_pos["entry_price"]),
                        "unrealizedPnl": str(pnl),
                        "liquidationPrice": None,  # Paper trading doesn't simulate liquidation easily
                        "leverage": None,
                        "marginType": "isolated",  # Assume isolated for paper
                        "collateral": None,
                        "info": {"paper": True},  # Add marker
                    }
                ]
            else:
                latest_data["positions"] = []
            latest_data["last_update_times"]["positions"] = time.time()

        # Wait before next fetch cycle
        await asyncio.sleep(bal_pos_interval)


# ==============================================================================
# Indicator & Pivot Calculation Tasks
# ==============================================================================
async def calculate_and_store_indicators() -> None:
    verbose_print("Calculating indicators...")
    ohlcv, min_candles = latest_data.get("indicator_ohlcv"), CONFIG["MIN_OHLCV_CANDLES"]
    indicators: dict[str, dict] = {
        k: {"value": None, "error": None}
        for k in ["sma1", "sma2", "ema1", "ema2", "momentum", "stoch_rsi"]
    }
    error_msg = None
    if not ohlcv or not isinstance(ohlcv, list) or len(ohlcv) < min_candles:
        error_msg = f"OHLCV Missing/Short ({len(ohlcv) if ohlcv else 0}<{min_candles})"
    if error_msg:
        for k in indicators:
            indicators[k]["error"] = error_msg
        latest_data["indicators"] = indicators
        verbose_print(f"Indicator calc skipped: {error_msg}")
        return
    try:
        close_prices = [c[4] for c in ohlcv if isinstance(c, list) and len(c) >= 5]
        if len(close_prices) < min_candles:
            raise ValueError("Close price extraction failed/short")
        indicators["sma1"]["value"] = calculate_sma(close_prices, CONFIG["SMA_PERIOD"])
        indicators["sma2"]["value"] = calculate_sma(close_prices, CONFIG["SMA2_PERIOD"])
        ema1_res = calculate_ema(close_prices, CONFIG["EMA1_PERIOD"])
        indicators["ema1"]["value"] = ema1_res[-1] if ema1_res else None
        ema2_res = calculate_ema(close_prices, CONFIG["EMA2_PERIOD"])
        indicators["ema2"]["value"] = ema2_res[-1] if ema2_res else None
        indicators["momentum"]["value"] = calculate_momentum(
            close_prices, CONFIG["MOMENTUM_PERIOD"]
        )
        for k in ["sma1", "sma2", "ema1", "ema2", "momentum"]:
            if indicators[k]["value"] is None:
                indicators[k]["error"] = "Calc Fail"
        rsi_list, rsi_err = calculate_rsi_manual(close_prices, CONFIG["RSI_PERIOD"])
        if rsi_err:
            indicators["stoch_rsi"]["error"] = f"RSI: {rsi_err}"
        elif rsi_list:
            k, d, stoch_err = calculate_stoch_rsi_manual(
                rsi_list, CONFIG["STOCH_K_PERIOD"], CONFIG["STOCH_D_PERIOD"]
            )
            indicators["stoch_rsi"].update({"k": k, "d": d, "error": stoch_err})
        else:
            indicators["stoch_rsi"]["error"] = "RSI List Empty"
        latest_data["indicators"] = indicators
        latest_data["last_update_times"]["indicators"] = time.time()
        verbose_print("Indicators calculated.")
    except Exception as e:
        print_color(f"# Indicator Calc Error: {e}", Fore.RED)
        traceback.print_exc()
        for k in indicators:
            indicators[k]["error"] = "Calc Exception"
            latest_data["indicators"] = indicators


async def calculate_and_store_pivots() -> None:
    verbose_print("Calculating pivots...")
    pivot_ohlcv = latest_data.get("pivot_ohlcv")
    calculated_pivots = None
    if pivot_ohlcv and len(pivot_ohlcv) > 0:
        # Use the second to last candle available (index -2 or 0 if only 1 available) for previous period's HLC
        prev_candle_index = -2 if len(pivot_ohlcv) >= 2 else 0
        prev_candle = pivot_ohlcv[prev_candle_index]
        if isinstance(prev_candle, list) and len(prev_candle) >= 5:
            calculated_pivots = calculate_fib_pivots(
                prev_candle[2], prev_candle[3], prev_candle[4]
            )  # High, Low, Close
            if calculated_pivots:
                latest_data["pivots"] = calculated_pivots
                latest_data["last_update_times"]["pivots"] = time.time()
                verbose_print("Pivots calculated.")
            else:
                latest_data["pivots"] = None
                verbose_print("Pivot calculation failed (invalid HLC?).")
        else:
            latest_data["pivots"] = None
            verbose_print(f"Invalid prev candle format for pivots: {prev_candle}")
    else:
        latest_data["pivots"] = None
        verbose_print("No/Empty pivot OHLCV data.")


# ==============================================================================
# Display Functions
# ==============================================================================
def display_combined_analysis_async(
    shared_data: dict, market_info: dict, config: dict
) -> tuple[dict[int, decimal.Decimal], dict[int, decimal.Decimal]]:
    global exchange  # Added global declaration
    ticker_info = shared_data.get("ticker")
    analyzed_ob = shared_data.get("orderbook")
    indicators_info = shared_data.get("indicators", {})
    positions_list = shared_data.get("positions", [])
    pivots_info = shared_data.get("pivots")
    balance_info = shared_data.get("balance")
    open_orders_list = shared_data.get("open_orders", [])
    position_info_processed = {
        "has_position": False,
        "position": None,
        "unrealizedPnl": None,
    }
    if positions_list:
        position_info_processed["has_position"] = True
        current_pos = positions_list[0]
        position_info_processed["position"] = current_pos
        try:
            pnl_raw = current_pos.get("unrealizedPnl")
            position_info_processed["unrealizedPnl"] = (
                decimal.Decimal(str(pnl_raw)) if pnl_raw is not None else None
            )
        except:
            position_info_processed["unrealizedPnl"] = None
    ts_ob, ts_tk = (
        shared_data.get("last_update_times", {}).get("orderbook"),
        shared_data.get("last_update_times", {}).get("ticker"),
    )
    timestamp_str = (
        analyzed_ob["timestamp"]
        if analyzed_ob and analyzed_ob.get("timestamp")
        else ticker_info["datetime"]
        if ticker_info and ticker_info.get("datetime")
        else time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(ts_ob)) + "(OB)"
        if ts_ob
        else time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(ts_tk)) + "(Tk)"
        if ts_tk
        else time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()) + "(Now)"
    )
    symbol = market_info["symbol"]
    display_header(symbol, timestamp_str, balance_info, config)
    last_price = display_ticker_and_trend(
        ticker_info, indicators_info, config, market_info
    )
    display_indicators(indicators_info, config, market_info, last_price)
    display_position(position_info_processed, ticker_info, market_info, config)
    display_open_orders(open_orders_list, market_info, config)
    display_pivots(pivots_info, last_price, market_info, config)
    ask_map, bid_map = display_orderbook(analyzed_ob, market_info, config)
    display_volume_analysis(analyzed_ob, market_info, config)
    stat_str = " | ".join(
        f"{k.upper()}:{Fore.GREEN if v == 'ok' else Fore.YELLOW if v == 'connecting' else Fore.RED}{v}{Style.RESET_ALL}"
        for k, v in shared_data.get("connection_status", {}).items()
    )
    paper_trading_status = (
        f" | {Fore.MAGENTA}{Style.BRIGHT}PAPER-TRADING-MODE: {Fore.GREEN}ON{Style.RESET_ALL}"
        if config["PAPER_TRADING_ENABLED"]
        else ""
    )
    print_color(
        f"--- Status: {stat_str}{paper_trading_status} ---",
        color=Fore.MAGENTA,
        style=Style.DIM,
    )
    return ask_map, bid_map


def display_header(
    symbol: str, timestamp: str, balance_info: Any | None, config: dict
) -> None:
    print_color("=" * 85, Fore.CYAN)
    print_color(
        f"📜 Pyrmethus Market Vision: {Fore.MAGENTA}{Style.BRIGHT}{symbol}{Style.RESET_ALL} @ {Fore.YELLOW}{timestamp}",
        Fore.CYAN,
    )
    balance_str, asset, prec = (
        f"{Fore.YELLOW}N/A{Style.RESET_ALL}",
        config["FETCH_BALANCE_ASSET"],
        config["BALANCE_DISPLAY_PRECISION"],
    )
    if balance_info is not None:
        try:
            balance_str = f"{Fore.GREEN}{format_decimal(balance_info, prec, prec)} {asset}{Style.RESET_ALL}"
        except Exception as e:
            verbose_print(f"Balance Disp Err: {e}")
            balance_str = f"{Fore.YELLOW}Error ({balance_info}){Style.RESET_ALL}"
    balance_label = f"💰 Balance ({asset}):"
    if config["PAPER_TRADING_ENABLED"]:
        balance_label = f"💰 Paper Balance ({asset}):"
        balance_str = f"{Fore.CYAN}{format_decimal(paper_balance, prec, prec)} {asset}{Style.RESET_ALL} {Fore.MAGENTA}{Style.BRIGHT}(PAPER){Style.RESET_ALL}"
    print_color(f"{balance_label} {balance_str}")
    print_color("-" * 85, Fore.CYAN)


def display_ticker_and_trend(
    ticker_info: dict | None, indicators_info: dict, config: dict, market_info: dict
) -> decimal.Decimal | None:
    price_prec, min_disp_prec = (
        market_info["price_precision"],
        config["MIN_PRICE_DISPLAY_PRECISION"],
    )
    last_price: decimal.Decimal | None = None
    curr_price_str, price_color = f"{Fore.YELLOW}N/A{Style.RESET_ALL}", Fore.WHITE
    if ticker_info and ticker_info.get("last") is not None:
        try:
            last_price = decimal.Decimal(str(ticker_info["last"]))
            sma1 = indicators_info.get("sma1", {}).get("value")
            if sma1:
                price_color = (
                    Fore.GREEN
                    if last_price > sma1
                    else Fore.RED
                    if last_price < sma1
                    else Fore.YELLOW
                )
            curr_price_str = f"{price_color}{Style.BRIGHT}{format_decimal(last_price, price_prec, min_disp_prec)}{Style.RESET_ALL}"
        except Exception as e:
            verbose_print(f"Ticker Disp Err: {e}")
            last_price = None
            curr_price_str = f"{Fore.YELLOW}Error{Style.RESET_ALL}"
    sma1_val, sma1_err = (
        indicators_info.get("sma1", {}).get("value"),
        indicators_info.get("sma1", {}).get("error"),
    )
    sma1_p, tf = config["SMA_PERIOD"], config["INDICATOR_TIMEFRAME"]
    trend_str, trend_color = f"SMA({sma1_p}@{tf}): -", Fore.YELLOW
    if sma1_err:
        trend_str, trend_color = f"SMA({sma1_p}@{tf}): {sma1_err}", Fore.YELLOW
    elif sma1_val and last_price:
        sma1_fmt = format_decimal(sma1_val, price_prec, min_disp_prec)
        trend_color = (
            Fore.GREEN
            if last_price > sma1_val
            else Fore.RED
            if last_price < sma1_val
            else Fore.YELLOW
        )
        trend_str = f"{'Above' if last_price > sma1_val else 'Below' if last_price < sma1_val else 'On'} SMA ({sma1_fmt})"
    elif sma1_val:
        trend_str, trend_color = (
            f"SMA({sma1_p}@{tf}): {format_decimal(sma1_val, price_prec, min_disp_prec)} (No Price)",
            Fore.WHITE,
        )
    else:
        trend_str = f"SMA({sma1_p}@{tf}): Unavailable"
    print_color(
        f"  Last Price: {curr_price_str} | {trend_color}{trend_str}{Style.RESET_ALL}"
    )
    return last_price


def display_indicators(
    indicators_info: dict,
    config: dict,
    market_info: dict,
    last_price: decimal.Decimal | None,
) -> None:
    price_prec, min_disp_prec, stoch_prec, _tf = (
        market_info["price_precision"],
        config["MIN_PRICE_DISPLAY_PRECISION"],
        config["STOCH_RSI_DISPLAY_PRECISION"],
        config["INDICATOR_TIMEFRAME"],
    )
    line1, line2 = [], []
    sma2_v, sma2_e, sma2_p = (
        indicators_info.get("sma2", {}).get("value"),
        indicators_info.get("sma2", {}).get("error"),
        config["SMA2_PERIOD"],
    )
    line1.append(
        f"SMA2({sma2_p}): {Fore.YELLOW}{'Err' if sma2_e else 'N/A' if sma2_v is None else format_decimal(sma2_v, price_prec, min_disp_prec)}{Style.RESET_ALL}"
    )
    ema1_v, ema2_v, ema_e = (
        indicators_info.get("ema1", {}).get("value"),
        indicators_info.get("ema2", {}).get("value"),
        indicators_info.get("ema1", {}).get("error")
        or indicators_info.get("ema2", {}).get("error"),
    )
    ema1_p, ema2_p = config["EMA1_PERIOD"], config["EMA2_PERIOD"]
    ema_str = f"EMA({ema1_p}/{ema2_p}): {Fore.YELLOW}{'Err' if ema_e else 'N/A'}{Style.RESET_ALL}"
    if ema1_v and ema2_v:
        ema_str = f"EMA({ema1_p}/{ema2_p}): {(Fore.GREEN if ema1_v > ema2_v else Fore.RED if ema1_v < ema2_v else Fore.YELLOW)}{format_decimal(ema1_v, price_prec, min_disp_prec)}/{format_decimal(ema2_v, price_prec, min_disp_prec)}{Style.RESET_ALL}"
    line1.append(ema_str)
    print_color(f"  {' | '.join(line1)}")
    mom_v, mom_e, mom_p = (
        indicators_info.get("momentum", {}).get("value"),
        indicators_info.get("momentum", {}).get("error"),
        config["MOMENTUM_PERIOD"],
    )
    mom_str = f"Mom({mom_p}): {Fore.YELLOW}{'Err' if mom_e else 'N/A'}{Style.RESET_ALL}"
    if mom_v is not None:
        mom_str = f"Mom({mom_p}): {(Fore.GREEN if mom_v > 0 else Fore.RED if mom_v < 0 else Fore.YELLOW)}{format_decimal(mom_v, price_prec, min_disp_prec)}{Style.RESET_ALL}"
    line2.append(mom_str)
    st_k, st_d, st_e = (
        indicators_info.get("stoch_rsi", {}).get("k"),
        indicators_info.get("stoch_rsi", {}).get("d"),
        indicators_info.get("stoch_rsi", {}).get("error"),
    )
    rsi_p, k_p, d_p = (
        config["RSI_PERIOD"],
        config["STOCH_K_PERIOD"],
        config["STOCH_D_PERIOD"],
    )
    stoch_str = f"StochRSI({rsi_p},{k_p},{d_p}): {Fore.YELLOW}{st_e[:10] + '..' if isinstance(st_e, str) else 'Err' if st_e else 'N/A'}{Style.RESET_ALL}"
    if st_k is not None:
        k_f, d_f = (
            format_decimal(st_k, stoch_prec),
            format_decimal(st_d, stoch_prec) if st_d is not None else "N/A",
        )
        osold, obought = config["STOCH_RSI_OVERSOLD"], config["STOCH_RSI_OVERBOUGHT"]
        k_color, signal = Fore.WHITE, ""
        if st_k < osold and (st_d is None or st_d < osold):
            k_color, signal = Fore.GREEN, "(OS)"
        elif st_k > obought and (st_d is None or st_d > obought):
            k_color, signal = Fore.RED, "(OB)"
        elif st_d is not None:
            k_color = (
                Fore.LIGHTGREEN_EX
                if st_k > st_d
                else Fore.LIGHTRED_EX
                if st_k < st_d
                else Fore.WHITE
            )
        stoch_str = f"StochRSI: {k_color}K={k_f}{Style.RESET_ALL} D={k_color}{d_f}{Style.RESET_ALL} {k_color}{signal}{Style.RESET_ALL}"
    line2.append(stoch_str)
    print_color(f"  {' | '.join(line2)}")


def display_position(
    position_info: dict, ticker_info: dict | None, market_info: dict, config: dict
) -> None:
    pnl_prec, price_prec, amount_prec, min_disp_prec = (
        config["PNL_PRECISION"],
        market_info["price_precision"],
        market_info["amount_precision"],
        config["MIN_PRICE_DISPLAY_PRECISION"],
    )
    pnl_str = f"{Fore.LIGHTBLACK_EX}Position: None{Style.RESET_ALL}"
    if position_info.get("has_position"):
        pos = position_info["position"]
        side, size_str, entry_str = (
            pos.get("side", "N/A").lower(),
            pos.get("contracts", "0"),
            pos.get("entryPrice", "0"),
        )  # Use lower case side
        quote = pos.get("quoteAsset", config["FETCH_BALANCE_ASSET"])
        pnl_val = position_info.get("unrealizedPnl")
        try:
            size, entry = (
                decimal.Decimal(size_str),
                decimal.Decimal(entry_str) if entry_str else decimal.Decimal("0"),
            )
            size_fmt, entry_fmt = (
                format_decimal(size, amount_prec),
                format_decimal(entry, price_prec, min_disp_prec),
            )
            side_color = (
                Fore.GREEN
                if side == "long"
                else Fore.RED
                if side == "short"
                else Fore.WHITE
            )  # Check against lower case
            if (
                pnl_val is None
                and ticker_info
                and ticker_info.get("last")
                and entry > 0
                and size != 0
            ):
                last_p = decimal.Decimal(str(ticker_info["last"]))
                pnl_val = (
                    (last_p - entry) * size
                    if side == "long"
                    else (entry - last_p) * size
                )  # Check against lower case
            pnl_val_str, pnl_color = (
                ("N/A", Fore.WHITE)
                if pnl_val is None
                else (
                    format_decimal(pnl_val, pnl_prec),
                    Fore.GREEN if pnl_val >= 0 else Fore.RED,
                )
            )
            pnl_str = f"Position: {side_color}{side.upper()} {size_fmt}{Style.RESET_ALL} @ {Fore.YELLOW}{entry_fmt}{Style.RESET_ALL} | uPNL: {pnl_color}{pnl_val_str} {quote}{Style.RESET_ALL}"
        except Exception as e:
            verbose_print(f"Pos Disp Err: {e}")
            pnl_str = f"{Fore.YELLOW}Position Data Err{Style.RESET_ALL}"
    print_color(f"  {pnl_str}")


def display_open_orders(
    open_orders_list: list[dict], market_info: dict, config: dict
) -> None:
    print_color("--- Open Orders ---", Fore.BLUE)
    if not open_orders_list:
        print_color(
            f"  {Fore.YELLOW}No open orders for {market_info['symbol']}.{Style.RESET_ALL}"
        )
        return
    price_prec, amount_prec, min_disp_prec = (
        market_info["price_precision"],
        market_info["amount_precision"],
        config["MIN_PRICE_DISPLAY_PRECISION"],
    )
    for order in open_orders_list:
        oid, side, otype, status = (
            order.get("id", "N/A"),
            order.get("side", "N/A").capitalize(),
            order.get("type", "N/A").capitalize(),
            order.get("status", "N/A"),
        )
        amount, price = (
            format_decimal(order.get("amount"), amount_prec),
            format_decimal(order.get("price"), price_prec, min_disp_prec)
            if order.get("price")
            else "Market",
        )
        filled = format_decimal(order.get("filled", 0), amount_prec)
        side_color = (
            Fore.GREEN
            if side.lower() == "buy"
            else Fore.RED
            if side.lower() == "sell"
            else Fore.WHITE
        )
        status_color = (
            Fore.CYAN
            if status == "open"
            else Fore.YELLOW
            if status == "partial"
            else Fore.WHITE
        )
        tp_sl_info = ""
        # Check unified 'info' dict first, then specific keys for TP/SL
        info_dict = order.get("info", {}) if isinstance(order.get("info"), dict) else {}
        tp = (
            info_dict.get("takeProfit")
            or info_dict.get("tpslMode") == "Full"
            and info_dict.get("tpLimitPrice")
            or order.get("takeProfitPrice")
        )
        sl = (
            info_dict.get("stopLoss")
            or info_dict.get("tpslMode") == "Full"
            and info_dict.get("slLimitPrice")
            or order.get("stopLossPrice")
        )
        if tp:
            tp_sl_info += f" TP:{Fore.GREEN}{format_decimal(tp, price_prec, min_disp_prec)}{Style.RESET_ALL}"
        if sl:
            tp_sl_info += f" SL:{Fore.RED}{format_decimal(sl, price_prec, min_disp_prec)}{Style.RESET_ALL}"
        print_color(
            f"  ID: {Fore.MAGENTA}{oid[:8]}..{Style.RESET_ALL} | {side_color}{side}{Style.RESET_ALL} {amount} @ {Fore.YELLOW}{price}{Style.RESET_ALL} (F:{filled}) | T:{Fore.CYAN}{otype}{Style.RESET_ALL} | S:{status_color}{status}{Style.RESET_ALL}{tp_sl_info}"
        )


def display_pivots(
    pivots_info: dict | None,
    last_price: decimal.Decimal | None,
    market_info: dict,
    config: dict,
) -> None:
    print_color(
        f"--- Fibonacci Pivots (Prev {config['PIVOT_TIMEFRAME']}) ---", Fore.BLUE
    )
    if not pivots_info:
        print_color(f"  {Fore.YELLOW}Pivot data unavailable.{Style.RESET_ALL}")
        return
    price_prec, min_disp_prec, width = (
        market_info["price_precision"],
        config["MIN_PRICE_DISPLAY_PRECISION"],
        max(12, market_info["price_precision"] + 7),
    )
    levels, lines = ["R3", "R2", "R1", "PP", "S1", "S2", "S3"], {}
    for level in levels:
        value = pivots_info.get(level)
        if value is not None:
            val_str, color = (
                format_decimal(value, price_prec, min_disp_prec),
                Fore.RED
                if "R" in level
                else Fore.GREEN
                if "S" in level
                else Fore.YELLOW,
            )
            hl = ""
            if last_price and value > 0:
                try:
                    # Highlight if price is very close (e.g., within 0.1% of the pivot value)
                    if abs(last_price - value) / value < decimal.Decimal("0.001"):
                        hl = (
                            Back.LIGHTBLACK_EX
                            + Fore.WHITE
                            + Style.BRIGHT
                            + " *NEAR* "
                            + Style.RESET_ALL
                        )
                except:
                    pass  # Avoid division by zero if value is somehow 0
            lines[level] = (
                f"{color}{level}: {val_str.rjust(width)}{Style.RESET_ALL}{hl}"
            )
        else:
            lines[level] = f"{level}: {'N/A'.rjust(width)}"


def display_orderbook(
    analyzed_ob: dict | None, market_info: dict, config: dict
) -> tuple[dict[int, decimal.Decimal], dict[int, decimal.Decimal]]:
    print_color("--- Order Book Depths ---", Fore.BLUE)
    ask_map, bid_map = {}, {}
    if not analyzed_ob:
        print_color(f"  {Fore.YELLOW}Order book data unavailable.{Style.RESET_ALL}")
        return ask_map, bid_map
    p_prec, _a_prec, min_p_prec, v_disp_prec, depth = (
        market_info["price_precision"],
        market_info["amount_precision"],
        config["MIN_PRICE_DISPLAY_PRECISION"],
        config["VOLUME_DISPLAY_PRECISION"],
        config["MAX_ORDERBOOK_DEPTH_DISPLAY"],
    )
    idx_w, p_w, v_w, cum_v_w = (
        len(f"[A{depth}]") + 1,
        max(10, p_prec + 4),
        max(10, v_disp_prec + 5),
        max(12, v_disp_prec + 7),
    )
    ask_lines, bid_lines = [], []
    display_asks = list(
        reversed(analyzed_ob["asks"])
    )  # Show best ask at bottom near bids
    for idx, ask in enumerate(display_asks):
        if idx >= depth:
            break
        idx_num = len(display_asks) - idx  # Number from best ask (A1) upwards
        idx_str, p_str = (
            f"[A{idx_num}]".ljust(idx_w),
            format_decimal(ask["price"], p_prec, min_p_prec),
        )
        cum_v_str = (
            f"{Fore.LIGHTBLACK_EX}(Cum:{ask['cumulative_volume']}){Style.RESET_ALL}"
        )
        ask_lines.append(
            f"{Fore.CYAN}{idx_str}{Style.NORMAL}{Fore.WHITE}{p_str:<{p_w}}{ask['style']}{ask['color']}{ask['volume_str']:<{v_w}}{Style.RESET_ALL} {cum_v_str:<{cum_v_w}}"
        )
        ask_map[idx_num] = ask["price"]  # Map A1, A2 etc.
    for idx, bid in enumerate(analyzed_ob["bids"]):
        if idx >= depth:
            break
        idx_num = idx + 1  # Number from best bid (B1) downwards
        idx_str, p_str = (
            f"[B{idx_num}]".ljust(idx_w),
            format_decimal(bid["price"], p_prec, min_p_prec),
        )
        cum_v_str = (
            f"{Fore.LIGHTBLACK_EX}(Cum:{bid['cumulative_volume']}){Style.RESET_ALL}"
        )
        bid_lines.append(
            f"{Fore.CYAN}{idx_str}{Style.NORMAL}{Fore.WHITE}{p_str:<{p_w}}{bid['style']}{bid['color']}{bid['volume_str']:<{v_w}}{Style.RESET_ALL} {cum_v_str:<{cum_v_w}}"
        )
        bid_map[idx_num] = bid["price"]  # Map B1, B2 etc.

    # Display asks descending, then bids ascending
    col_w = idx_w + p_w + v_w + cum_v_w + 3
    print_color(f"{'Asks'.center(col_w)}", Fore.LIGHTBLACK_EX)
    print_color(f"{'-' * col_w}", Fore.LIGHTBLACK_EX)
    for _line in reversed(ask_lines):
        pass  # Print asks from worst (top) to best (bottom)

    # Spread calculation (use A1 and B1 from the maps)
    best_a = ask_map.get(1, decimal.Decimal("NaN"))
    best_b = bid_map.get(1, decimal.Decimal("NaN"))
    spread = (
        best_a - best_b
        if best_a.is_finite() and best_b.is_finite()
        else decimal.Decimal("NaN")
    )
    spread_str = (
        format_decimal(spread, p_prec, min_p_prec) if spread.is_finite() else "N/A"
    )
    print_color(
        f"\n--- Spread: {spread_str} ---".center(col_w), Fore.MAGENTA, Style.DIM
    )

    print_color(f"{'Bids'.center(col_w)}", Fore.LIGHTBLACK_EX)
    print_color(f"{'-' * col_w}", Fore.LIGHTBLACK_EX)
    for _line in bid_lines:
        pass  # Print bids from best (top) to worst (bottom)

    return ask_map, bid_map


def display_volume_analysis(
    analyzed_ob: dict | None, market_info: dict, config: dict
) -> None:
    if not analyzed_ob:
        return
    a_prec, p_prec, v_disp_prec, min_p_prec = (
        market_info["amount_precision"],
        market_info["price_precision"],
        config["VOLUME_DISPLAY_PRECISION"],
        config["MIN_PRICE_DISPLAY_PRECISION"],
    )
    print_color("\n--- Volume Analysis (Fetched Depth) ---", Fore.BLUE)
    tot_a, tot_b = (
        analyzed_ob["ask_total_volume_fetched"],
        analyzed_ob["bid_total_volume_fetched"],
    )
    cum_a, cum_b = (
        analyzed_ob["cumulative_ask_volume_displayed"],
        analyzed_ob["cumulative_bid_volume_displayed"],
    )
    print_color(
        f"  Total Ask: {Fore.RED}{format_decimal(tot_a, a_prec, v_disp_prec)}{Style.RESET_ALL} | Total Bid: {Fore.GREEN}{format_decimal(tot_b, a_prec, v_disp_prec)}{Style.RESET_ALL}"
    )
    print_color(
        f"  Cum Ask(Disp): {Fore.RED}{format_decimal(cum_a, a_prec, v_disp_prec)}{Style.RESET_ALL} | Cum Bid(Disp): {Fore.GREEN}{format_decimal(cum_b, a_prec, v_disp_prec)}{Style.RESET_ALL}"
    )
    imb = analyzed_ob["volume_imbalance_ratio_fetched"]
    imb_c, imb_s = Fore.WHITE, "N/A"
    if imb.is_infinite():
        imb_c, imb_s = Fore.LIGHTGREEN_EX, "Inf (>>1)"
    elif imb.is_finite():
        imb_s = format_decimal(imb, 2)
        imb_c = (
            Fore.GREEN
            if imb > decimal.Decimal("1.5")
            else Fore.RED
            if imb < decimal.Decimal("0.67") and not imb.is_zero()
            else Fore.LIGHTRED_EX
            if imb.is_zero() and tot_a > 0
            else Fore.WHITE
        )
    ask_vwap, bid_vwap = (
        format_decimal(analyzed_ob["ask_vwap_fetched"], p_prec, min_p_prec),
        format_decimal(analyzed_ob["bid_vwap_fetched"], p_prec, min_p_prec),
    )
    print_color(
        f"  Imbalance(B/A): {imb_c}{imb_s}{Style.RESET_ALL} | VWAP Ask: {Fore.YELLOW}{ask_vwap}{Style.RESET_ALL} | VWAP Bid: {Fore.YELLOW}{bid_vwap}{Style.RESET_ALL}"
    )
    print_color("--- Pressure Reading ---", Fore.BLUE)
    if imb.is_infinite():
        print_color("  Extreme Bid Dominance", Fore.LIGHTYELLOW_EX)
    elif imb.is_zero() and tot_a > 0:
        print_color("  Extreme Ask Dominance", Fore.LIGHTYELLOW_EX)
    elif imb > decimal.Decimal("1.5"):
        print_color("  Strong Buy Pressure", Fore.GREEN, Style.BRIGHT)
    elif imb < decimal.Decimal("0.67") and not imb.is_zero():
        print_color("  Strong Sell Pressure", Fore.RED, Style.BRIGHT)
    else:
        print_color("  Volume Relatively Balanced", Fore.WHITE)
    print_color("=" * 85, Fore.CYAN)


# --- Paper Trading Simulation Functions ---
def simulate_order_fill(
    order_type: str,
    side: str,
    price: decimal.Decimal | None,
    amount: decimal.Decimal,
    symbol: str,
) -> dict | None:
    ticker = latest_data.get("ticker")
    orderbook = latest_data.get("orderbook")
    fill_price = None
    # Prioritize order book for more realistic fills if available
    if order_type.lower() == "market":
        if orderbook:
            fill_price = (
                orderbook["asks"][0]["price"]
                if side == "buy" and orderbook.get("asks")
                else orderbook["bids"][0]["price"]
                if side == "sell" and orderbook.get("bids")
                else None
            )
        elif ticker and ticker.get("last"):  # Fallback to ticker
            fill_price = decimal.Decimal(str(ticker["last"]))
    elif order_type.lower() == "limit" and price is not None:
        # Check if limit price is immediately crossable based on OB or ticker
        crossable = False
        if orderbook:
            if (
                side == "buy"
                and orderbook.get("asks")
                and price >= orderbook["asks"][0]["price"]
                or side == "sell"
                and orderbook.get("bids")
                and price <= orderbook["bids"][0]["price"]
            ):
                crossable = True
        elif ticker and ticker.get("last"):  # Fallback to ticker check
            last_p = decimal.Decimal(str(ticker["last"]))
            if (side == "buy" and price >= last_p) or (
                side == "sell" and price <= last_p
            ):
                crossable = (
                    True  # Simpler check: just needs to be better than last price
                )

        # Fill immediately if crossable, otherwise it remains open
        if crossable:
            fill_price = price

    return {"price": fill_price, "amount": amount} if fill_price else None


def update_paper_position(
    symbol: str, side: str, amount: decimal.Decimal, fill_price: decimal.Decimal
) -> None:
    global paper_positions, paper_balance
    current_pos = paper_positions.get(symbol)
    pnl_prec = CONFIG["PNL_PRECISION"]
    if current_pos:
        if current_pos["side"] == side:  # Increasing position
            new_size = current_pos["size"] + amount
            new_entry_price = (
                (current_pos["entry_price"] * current_pos["size"])
                + (fill_price * amount)
            ) / new_size
            paper_trade_log.append(
                f"INCREASED {side} {amount} {symbol} @ {fill_price}. New Avg Entry: {format_decimal(new_entry_price, latest_data['market_info']['price_precision'])}"
            )
            current_pos["size"], current_pos["entry_price"] = new_size, new_entry_price
        else:  # Reducing or flipping position
            close_amount = min(amount, current_pos["size"])
            realized_pnl = decimal.Decimal("0")
            if close_amount > 0:
                realized_pnl = (
                    (fill_price - current_pos["entry_price"]) * close_amount
                    if current_pos["side"] == "long"
                    else (current_pos["entry_price"] - fill_price) * close_amount
                )
                paper_balance += realized_pnl
                paper_trade_log.append(
                    f"CLOSED/REDUCED {current_pos['side']} {close_amount} {symbol} @ {fill_price}. PNL: {format_decimal(realized_pnl, pnl_prec)}"
                )

            if (
                amount >= current_pos["size"]
            ):  # Closed fully, potentially opening opposite
                remaining_open_amount = amount - current_pos["size"]
                del paper_positions[symbol]  # Remove old position first
                if remaining_open_amount > 0:  # Flipped to new position
                    paper_positions[symbol] = {
                        "side": side,
                        "size": remaining_open_amount,
                        "entry_price": fill_price,
                    }
                    paper_trade_log.append(
                        f"FLIPPED & OPENED {side} {remaining_open_amount} {symbol} @ {fill_price}"
                    )
            else:  # Reduced position size
                current_pos["size"] -= close_amount
                paper_trade_log.append(
                    f"REDUCED {current_pos['side']} {symbol} to {current_pos['size']}"
                )
    else:  # Opening new position
        paper_positions[symbol] = {
            "side": side,
            "size": amount,
            "entry_price": fill_price,
        }
        paper_trade_log.append(f"OPENED {side} {amount} {symbol} @ {fill_price}")


# --- Trading Functions (Async Versions) ---
async def place_market_order_async(
    exchange_pro: ccxtpro.Exchange,
    symbol: str,
    side: str,
    amount_str: str,
    market_info: dict,
    config: dict,
    params: dict | None = None,
) -> None:
    print_color(
        f"{Fore.CYAN}# Preparing ASYNC {side.upper()} market order...{Style.RESET_ALL}"
    )
    params = params or {}
    try:
        amount = decimal.Decimal(amount_str)
        min_amount, amount_prec, amount_step = (
            market_info.get("min_amount", decimal.Decimal("0")),
            market_info["amount_precision"],
            market_info.get("amount_step", decimal.Decimal("0")),
        )
        if amount <= 0:
            print_color("Amount must be positive.", Fore.YELLOW)
            return
        if min_amount > 0 and amount < min_amount:
            print_color(
                f"Amount < min ({format_decimal(min_amount, amount_prec)}).",
                Fore.YELLOW,
            )
            return
        if amount_step > 0 and (amount % amount_step) != 0:
            original_amount = amount
            amount = (amount // amount_step) * amount_step
            print_color(f"Amount rounded: {original_amount} -> {amount}", Fore.YELLOW)
            if amount <= 0 or (min_amount > 0 and amount < min_amount):
                print_color("Rounded amount invalid.", Fore.RED)
                return
        amount_str = format_decimal(
            amount, amount_prec
        )  # Use validated/rounded amount string
    except Exception as e:
        print_color(f"Invalid amount: {e}", Fore.YELLOW)
        return

    tp_price_str, sl_price_str = "N/A", "N/A"
    # --- Prepare Base Params (including category) ---
    order_params_ccxt = {
        "positionIdx": config["POSITION_IDX"],
        "category": config["EXCHANGE_TYPE"],
    }

    if config.get("ADD_TP_SL_TO_ORDERS", True):
        tp_price, sl_price = await get_tp_sl_from_user_async(
            market_info, side
        )  # Use async version
        if tp_price:
            params["takeProfit"] = float(tp_price)
            tp_price_str = format_decimal(tp_price, market_info["price_precision"])
        if sl_price:
            params["stopLoss"] = float(sl_price)
            sl_price_str = format_decimal(sl_price, market_info["price_precision"])
        params["tpTriggerBy"] = config["TP_TRIGGER_TYPE"]
        params["slTriggerBy"] = config["SL_TRIGGER_TYPE"]

    order_params_ccxt.update(params)  # Merge custom params with base params

    side_color = Fore.GREEN if side == "buy" else Fore.RED
    tp_sl_info = (
        f" TP: {Fore.GREEN}{tp_price_str}{Style.RESET_ALL} SL: {Fore.RED}{sl_price_str}{Style.RESET_ALL}"
        if config.get("ADD_TP_SL_TO_ORDERS", True)
        else ""
    )
    prompt = f"{Style.BRIGHT}Confirm MARKET {side_color}{side.upper()}{Style.RESET_ALL} {Fore.YELLOW}{amount_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL}?{tp_sl_info} (yes/no): "
    if not await user_confirm_async(prompt):
        print_color("Order cancelled.", Fore.CYAN)
        return

    if config["PAPER_TRADING_ENABLED"]:
        fill_info = simulate_order_fill("market", side, None, amount, symbol)
        if fill_info:
            fill_price = fill_info["price"]
            update_paper_position(symbol, side, amount, fill_price)
            print_color(
                f"{Fore.MAGENTA}{Style.BRIGHT}# Paper:{Style.RESET_ALL} MARKET {side.upper()} {Fore.YELLOW}{amount_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL} Executed @ {Fore.YELLOW}{format_decimal(fill_price, market_info['price_precision'])}{Style.RESET_ALL}",
                Fore.CYAN,
            )
            termux_toast(
                f"Paper MARKET {side.upper()} {amount_str} {symbol} Executed",
                duration="long",
            )
        else:
            print_color(
                f"{Fore.YELLOW}# Paper: Could not simulate MARKET fill (no price?). Order NOT placed.{Style.RESET_ALL}",
                Fore.YELLOW,
            )
            termux_toast(
                f"Paper MARKET {side.upper()} {amount_str} {symbol} Failed", "long"
            )
    else:
        try:
            print_color(f"{Fore.CYAN}# Placing market order...{Style.DIM}", end=" ")
            # Use validated amount and combined params
            order = await exchange_pro.create_market_order(
                symbol, side, float(amount), params=order_params_ccxt
            )
            print_color(f"{Fore.GREEN}Done.{Style.RESET_ALL}")
            oid, filled_price = (
                (order.get("id", "N/A"), order.get("average", "N/A"))
                if isinstance(order, dict)
                else ("N/A", "N/A")
            )
            print_color(
                f"{Fore.GREEN}{Style.BRIGHT}Market Order {side.upper()} {amount_str} {symbol} Placed/Filled @ Avg ~{filled_price}. ID: {oid}{Style.RESET_ALL}",
                Fore.CYAN,
            )
            termux_toast(f"Market Order {side.upper()} {amount_str} Placed", "long")
            verbose_print(f"Order details: {order}")
        except Exception as e:
            print_color(f"{Fore.RED}Failed. Error: {e}{Style.RESET_ALL}", Fore.RED)
            termux_toast(f"Market Order Failed: {e}", duration="long")
            traceback.print_exc()


async def place_limit_order_async(
    exchange_pro: ccxtpro.Exchange,
    symbol: str,
    side: str,
    amount_str: str,
    price_str: str,
    market_info: dict,
    config: dict,
    params: dict | None = None,
) -> None:
    print_color(
        f"{Fore.CYAN}# Preparing ASYNC {side.upper()} limit order...{Style.RESET_ALL}"
    )
    params = params or {}
    try:
        amount, price = decimal.Decimal(amount_str), decimal.Decimal(price_str)
        min_amount, amount_prec, amount_step = (
            market_info.get("min_amount", decimal.Decimal("0")),
            market_info["amount_precision"],
            market_info.get("amount_step", decimal.Decimal("0")),
        )
        price_prec, price_tick = (
            market_info["price_precision"],
            market_info["price_tick_size"],
        )
        if amount <= 0 or price <= 0:
            print_color("Amount/Price must be positive.", Fore.YELLOW)
            return
        if min_amount > 0 and amount < min_amount:
            print_color(
                f"Amount < min ({format_decimal(min_amount, amount_prec)}).",
                Fore.YELLOW,
            )
            return
        if amount_step > 0 and (amount % amount_step) != 0:
            original_amount = amount
            amount = (amount // amount_step) * amount_step
            print_color(f"Amount rounded: {original_amount} -> {amount}", Fore.YELLOW)
            if amount <= 0 or (min_amount > 0 and amount < min_amount):
                print_color("Rounded amount invalid.", Fore.RED)
                return
        if price_tick > 0 and (price % price_tick) != 0:
            original_price = price
            price = price.quantize(
                price_tick,
                rounding=decimal.ROUND_DOWN if side == "buy" else decimal.ROUND_UP,
            )  # Round conservatively
            print_color(
                f"Price rounded ({'down' if side == 'buy' else 'up'}): {original_price} -> {price}",
                Fore.YELLOW,
            )
            if price <= 0:
                print_color("Rounded price invalid.", Fore.RED)
                return
        amount_str, price_str = (
            format_decimal(amount, amount_prec),
            format_decimal(price, price_prec),
        )  # Use validated/rounded strings
    except Exception as e:
        print_color(f"Invalid amount/price: {e}", Fore.YELLOW)
        return

    tp_price_str, sl_price_str = "N/A", "N/A"
    # --- Prepare Base Params (including category) ---
    order_params_ccxt = {
        "positionIdx": config["POSITION_IDX"],
        "category": config["EXCHANGE_TYPE"],
    }

    if config.get("ADD_TP_SL_TO_ORDERS", True):
        tp_price, sl_price = await get_tp_sl_from_user_async(
            market_info, side
        )  # Use async version
        if tp_price:
            params["takeProfit"] = float(tp_price)
            tp_price_str = format_decimal(tp_price, market_info["price_precision"])
        if sl_price:
            params["stopLoss"] = float(sl_price)
            sl_price_str = format_decimal(sl_price, market_info["price_precision"])
        params["tpTriggerBy"] = config["TP_TRIGGER_TYPE"]
        params["slTriggerBy"] = config["SL_TRIGGER_TYPE"]

    order_params_ccxt.update(params)  # Merge custom params with base params

    side_color = Fore.GREEN if side == "buy" else Fore.RED
    tp_sl_info = (
        f" TP: {Fore.GREEN}{tp_price_str}{Style.RESET_ALL} SL: {Fore.RED}{sl_price_str}{Style.RESET_ALL}"
        if config.get("ADD_TP_SL_TO_ORDERS", True)
        else ""
    )
    prompt = f"{Style.BRIGHT}Confirm LIMIT {side_color}{side.upper()}{Style.RESET_ALL} {Fore.YELLOW}{amount_str}{Style.RESET_ALL} @ {Fore.YELLOW}{price_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL}?{tp_sl_info} (yes/no): "
    if not await user_confirm_async(prompt):
        print_color("Order cancelled.", Fore.CYAN)
        return

    if config["PAPER_TRADING_ENABLED"]:
        global paper_orders
        order_id = f"paper-{uuid.uuid4().hex[:8]}"
        paper_order = {
            "id": order_id,
            "symbol": symbol,
            "type": "limit",
            "side": side,
            "amount": amount,
            "price": price,
            "status": "open",
            "filled": decimal.Decimal("0"),
            "remaining": amount,
            "timestamp": time.time() * 1000,
            "info": {
                "paper": True,
                "stopLoss": params.get("stopLoss"),
                "takeProfit": params.get("takeProfit"),
            },
        }
        paper_orders.append(paper_order)
        print_color(
            f"{Fore.MAGENTA}{Style.BRIGHT}# Paper:{Style.RESET_ALL} LIMIT {side.upper()} {Fore.YELLOW}{amount_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL} @ {Fore.YELLOW}{price_str}{Style.RESET_ALL} Placed (ID: {order_id})",
            Fore.CYAN,
        )
        termux_toast(f"Paper LIMIT {side.upper()} {amount_str} Placed", "long")
    # await check_paper_order_fills() # Let periodic task handle fills
    else:
        try:
            print_color(f"{Fore.CYAN}# Placing limit order...{Style.DIM}", end=" ")
            # Use validated amount/price and combined params
            order = await exchange_pro.create_limit_order(
                symbol, side, float(amount), float(price), params=order_params_ccxt
            )
            print_color(f"{Fore.GREEN}Done.{Style.RESET_ALL}")
            oid = order.get("id", "N/A") if isinstance(order, dict) else "N/A"
            print_color(
                f"{Fore.GREEN}{Style.BRIGHT}Limit Order {side.upper()} {amount_str} {symbol} Placed @ {price_str}. ID: {oid}{Style.RESET_ALL}",
                Fore.CYAN,
            )
            termux_toast(f"Limit Order {side.upper()} {amount_str} Placed", "long")
            verbose_print(f"Order details: {order}")
        except Exception as e:
            print_color(f"{Fore.RED}Failed. Error: {e}{Style.RESET_ALL}", Fore.RED)
            termux_toast(f"Limit Order Failed {e}", duration="long")
            traceback.print_exc()


# Modified interactive limit to accept amount first
async def place_limit_order_interactive_async(
    exchange_pro: ccxtpro.Exchange,
    symbol: str,
    side: str,
    amount_str: str,
    market_info: dict,
    config: dict,
    ask_map: dict[int, decimal.Decimal],
    bid_map: dict[int, decimal.Decimal],
) -> None:
    print_color(
        f"\n{Fore.BLUE}--- Interactive Limit Order ({side.upper()}) ---{Style.RESET_ALL}"
    )
    target_map, prompt_char = (ask_map, "A") if side == "buy" else (bid_map, "B")

    # Validate amount first (already done before calling, but double check format)
    try:
        amount = decimal.Decimal(amount_str)
        amount_str_fmt = format_decimal(
            amount, market_info["amount_precision"]
        )  # Use formatted string
        print_color(f"Amount set: {Fore.YELLOW}{amount_str_fmt}{Style.RESET_ALL}")
    except Exception as e:
        print_color(
            f"Invalid amount passed to interactive: {amount_str} ({e})", Fore.RED
        )
        return

    if not target_map:
        print_color(
            f"Order book side for '{side}' empty. Cannot select price.", Fore.YELLOW
        )
        return

    selected_price: decimal.Decimal | None = None
    while selected_price is None:
        try:
            # Display OB again for context if needed (or assume user sees it)
            # display_orderbook(latest_data.get("orderbook"), market_info, config) # Optional re-display
            idx_prompt = f"{Style.BRIGHT}Select OB Index ({prompt_char}1-{prompt_char}{len(target_map)}, 'm' manual price, 'c' cancel): {Style.RESET_ALL}"
            index_str = await asyncio.to_thread(input, idx_prompt)
            index_str = index_str.strip().upper()

            if index_str == "C":
                print_color("Cancelled.", Fore.YELLOW)
                return
            elif index_str == "M":
                manual_price_str = await get_price_from_user_async(
                    config, market_info, "Manual Limit Price"
                )
                if manual_price_str:
                    selected_price = decimal.Decimal(manual_price_str)
                else:
                    continue  # Re-prompt if manual price entry cancelled
            elif not index_str.startswith(prompt_char) or not index_str[1:].isdigit():
                print_color("Invalid format.", Fore.YELLOW)
                continue
            else:
                index = int(index_str[1:])
                selected_price = target_map.get(index)
                if selected_price is None:
                    print_color(
                        f"Index {index_str} not found in current OB.", Fore.YELLOW
                    )

        except (ValueError, IndexError):
            print_color("Invalid index.", Fore.YELLOW)
        except (EOFError, KeyboardInterrupt):
            print_color("\nCancelled.", Fore.YELLOW)
            return

    # Use the validated amount and selected price
    price_fmt = format_decimal(selected_price, market_info["price_precision"])
    print_color(f"Selected Price: {Fore.YELLOW}{price_fmt}{Style.RESET_ALL}")
    await place_limit_order_async(
        exchange_pro,
        symbol,
        side,
        amount_str_fmt,
        str(selected_price),
        market_info,
        config,
    )


async def place_trailing_stop_order_async(
    exchange_pro: ccxtpro.Exchange,
    symbol: str,
    side: str,
    market_info: dict,
    config: dict,
) -> None:
    print_color(
        f"{Fore.CYAN}# Preparing ASYNC {side.upper()} trailing stop market order...{Style.RESET_ALL}"
    )
    amount_str = await get_amount_from_user_async(
        config, market_info, "Trailing Stop Amount"
    )
    if not amount_str:
        return

    try:
        amount = decimal.Decimal(amount_str)
        amount_str = format_decimal(
            amount, market_info["amount_precision"]
        )  # Use validated string
    except:
        print_color("Invalid amount.", Fore.YELLOW)
        return

    try:
        dist_prompt = f"{Fore.CYAN}Enter trailing stop price distance (e.g., 100 for ${market_info['symbol'].split('/')[1].split(':')[0]} 100): {Style.RESET_ALL}"
        trailing_dist_input = await asyncio.to_thread(input, dist_prompt)
        trailing_dist = decimal.Decimal(trailing_dist_input)
        if trailing_dist <= 0:
            print_color("Trailing distance must be positive.", Fore.YELLOW)
            return
        trailing_dist_str = format_decimal(
            trailing_dist, market_info["price_precision"]
        )
    except (decimal.InvalidOperation, ValueError, EOFError, KeyboardInterrupt):
        print_color("Invalid distance input or cancelled.", Fore.YELLOW)
        return

    side_color = Fore.GREEN if side == "buy" else Fore.RED
    prompt = f"{Style.BRIGHT}Confirm TRAILING STOP MARKET {side_color}{side.upper()}{Style.RESET_ALL} {Fore.YELLOW}{amount_str}{Style.RESET_ALL} {Fore.CYAN}Dist:{Fore.YELLOW}{trailing_dist_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL}? (yes/no): "
    if not await user_confirm_async(prompt):
        print_color("Order cancelled.", Fore.CYAN)
        return

    if config["PAPER_TRADING_ENABLED"]:
        global paper_orders
        order_id = f"paper-ts-{uuid.uuid4().hex[:8]}"
        # Paper trail stops are tricky, represent them as open orders for now
        paper_order = {
            "id": order_id,
            "symbol": symbol,
            "type": "trailing_stop",
            "side": side,
            "amount": amount,
            "trailing_distance": trailing_dist,
            "status": "open",
            "filled": decimal.Decimal("0"),
            "remaining": amount,
            "timestamp": time.time() * 1000,
            "info": {"paper": True, "activationPrice": None},
        }  # Activation price might be needed
        paper_orders.append(paper_order)
        print_color(
            f"{Fore.MAGENTA}# Paper:{Style.RESET_ALL} TRAILING STOP {side.upper()} {Fore.YELLOW}{amount_str}{Style.RESET_ALL} {Fore.CYAN}Dist:{Fore.YELLOW}{trailing_dist_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL} Placed (ID: {order_id})",
            Fore.CYAN,
        )
        termux_toast(f"Paper TRAIL STOP {side.upper()} Placed", "long")
    else:
        try:
            print_color(
                f"{Fore.CYAN}# Placing trailing stop order...{Style.DIM}", end=" "
            )
            # Bybit V5 uses create_order with specific params for trailing stop market
            params = {
                "category": config["EXCHANGE_TYPE"],
                "orderType": "Market",  # Trailing stops are usually triggered as Market orders
                "positionIdx": config["POSITION_IDX"],
                "trailingStop": str(trailing_dist),  # Price distance
                # 'activePrice': 'xxx', # Optional activation price
                "orderLinkId": generate_order_link_id(),
            }
            order = await exchange_pro.create_order(
                symbol, "market", side, float(amount), params=params
            )
            print_color(f"{Fore.GREEN}Done.{Style.RESET_ALL}")
            oid = order.get("id", "N/A") if isinstance(order, dict) else "N/A"
            # Note: The order placed is conditional. It won't show in open orders until triggered.
            print_color(
                f"{Fore.GREEN}{Style.BRIGHT}Trailing Stop Market {side.upper()} {amount_str} {symbol} Condition Placed. Dist:{trailing_dist_str}. ID: {oid}{Style.RESET_ALL}",
                Fore.CYAN,
            )
            termux_toast(f"Trail Stop {side.upper()} {amount_str} Placed", "long")
            verbose_print(f"Order details: {order}")
        except Exception as e:
            print_color(f"{Fore.RED}Failed. Error: {e}{Style.RESET_ALL}", Fore.RED)
            termux_toast(f"Trail Stop Failed: {e}", duration="long")
            traceback.print_exc()


# --- Paper Trading Order Fill Simulation ---
async def check_paper_order_fills() -> None:
    global paper_orders, paper_positions, paper_balance, paper_trade_log
    if not CONFIG["PAPER_TRADING_ENABLED"]:
        return
    ticker = latest_data.get("ticker")
    if not ticker or not ticker.get("last"):
        return  # Need price to check fills
    if not latest_data.get("market_info"):
        return  # Need market info for formatting

    last_price = decimal.Decimal(str(ticker["last"]))
    market_info = latest_data["market_info"]
    orders_to_remove_ids = []
    filled_something = False

    # --- Process existing active paper trailing stops ---
    # This needs refinement - how to store/update the peak price? For now, basic trigger.
    # A more robust way would store peak price associated with the order ID.

    # --- Iterate through open paper orders ---
    for i in range(len(paper_orders) - 1, -1, -1):  # Iterate backwards for safe removal
        order = paper_orders[i]
        if order["status"] != "open":
            continue

        filled = False
        fill_price = None
        fill_reason = ""

        # --- Limit Order Check ---
        if order["type"] == "limit":
            if (order["side"] == "buy" and last_price <= order["price"]) or (
                order["side"] == "sell" and last_price >= order["price"]
            ):
                filled, fill_price, fill_reason = (
                    True,
                    order["price"],
                    "Limit Price Hit",
                )

        # --- TP/SL Check (Attached to paper order info) ---
        # This assumes TP/SL are attached to the original order. Bybit might create separate conditional orders.
        elif order.get("info", {}).get("takeProfit") and not filled:
            tp_price = decimal.Decimal(str(order["info"]["takeProfit"]))
            # TP triggers when price moves favorably THROUGH the TP level
            if (order["side"] == "buy" and last_price >= tp_price) or (
                order["side"] == "sell" and last_price <= tp_price
            ):
                filled, fill_price, fill_reason = (
                    True,
                    tp_price,
                    f"TP Hit ({tp_price})",
                )  # Fill at TP price
                verbose_print(
                    f"Paper TP Check: Order {order['id']} Side {order['side']} Last {last_price} TP {tp_price} -> FILL"
                )
        elif order.get("info", {}).get("stopLoss") and not filled:
            sl_price = decimal.Decimal(str(order["info"]["stopLoss"]))
            # SL triggers when price moves unfavorably THROUGH the SL level
            if (order["side"] == "buy" and last_price <= sl_price) or (
                order["side"] == "sell" and last_price >= sl_price
            ):
                filled, fill_price, fill_reason = (
                    True,
                    sl_price,
                    f"SL Hit ({sl_price})",
                )  # Fill at SL price (simplification)
                verbose_print(
                    f"Paper SL Check: Order {order['id']} Side {order['side']} Last {last_price} SL {sl_price} -> FILL"
                )

        # --- Trailing Stop Logic (Simplified Paper Simulation) ---
        elif order["type"] == "trailing_stop" and not filled:
            distance = order["trailing_distance"]
            activation_price = order["info"].get("activationPrice")

            # Activate if price moves beyond activation (if set) or immediately if not set
            activated = False
            if activation_price:
                if (order["side"] == "buy" and last_price >= activation_price) or (
                    order["side"] == "sell" and last_price <= activation_price
                ):
                    activated = True
                    if order["info"].get("_peak_price") is None:
                        order["info"]["_peak_price"] = (
                            last_price  # Initialize peak on activation
                        )
            else:  # Activate immediately
                activated = True
                if order["info"].get("_peak_price") is None:
                    order["info"]["_peak_price"] = last_price  # Initialize peak

            if activated:
                peak_price = order["info"]["_peak_price"]
                # Update peak
                if (
                    order["side"] == "buy"
                    and last_price > peak_price
                    or order["side"] == "sell"
                    and last_price < peak_price
                ):
                    order["info"]["_peak_price"] = last_price
                    peak_price = last_price

                # Check trigger
                trigger_price = (
                    peak_price - distance
                    if order["side"] == "buy"
                    else peak_price + distance
                )
                if (order["side"] == "buy" and last_price <= trigger_price) or (
                    order["side"] == "sell" and last_price >= trigger_price
                ):
                    # Simulate market fill
                    market_fill = simulate_order_fill(
                        "market", order["side"], None, order["amount"], order["symbol"]
                    )
                    if market_fill:
                        filled, fill_price = True, market_fill["price"]
                        fill_reason = f"Trailing Stop Hit (Trig:{format_decimal(trigger_price, market_info['price_precision'])})"
                        verbose_print(
                            f"Paper Trail Stop Check: Order {order['id']} Side {order['side']} Last {last_price} Peak {peak_price} Trig {trigger_price} -> FILL @ {fill_price}"
                        )
                    else:
                        verbose_print(
                            "Paper Trail Stop Check: Triggered but failed to get market fill price."
                        )

        # --- Process Fill ---
        if filled and fill_price:
            fill_amount = order["remaining"]  # Fill the remaining amount
            order["status"] = "filled"
            order["filled"] += fill_amount
            order["remaining"] = decimal.Decimal("0")
            update_paper_position(
                order["symbol"], order["side"], fill_amount, fill_price
            )  # Update using filled amount
            print_color(
                f"{Fore.MAGENTA}# Paper:{Style.RESET_ALL} Order {order['id']} ({order['type'].upper()} {order['side'].upper()}) Filled @ {Fore.YELLOW}{format_decimal(fill_price, market_info['price_precision'])}{Style.RESET_ALL} ({fill_reason})",
                Fore.CYAN,
            )
            termux_toast(f"Paper Order {order['id']} Filled", "long")
            orders_to_remove_ids.append(order["id"])  # Mark for removal
            filled_something = True

    # --- Remove filled orders ---
    if orders_to_remove_ids:
        paper_orders = [o for o in paper_orders if o["id"] not in orders_to_remove_ids]
    if filled_something and CONFIG["VERBOSE_DEBUG"]:
        verbose_print("Paper trade log:")
        for log in paper_trade_log[-5:]:
            print_color(f"  {log}", Fore.LIGHTBLACK_EX)  # Print last 5 log entries


# ==============================================================================
# Interactive User Input Functions
# ==============================================================================
async def get_amount_from_user_async(
    config: dict, market_info: dict, prompt_prefix: str = "Amount"
) -> str | None:
    """Async version to get validated amount string."""
    while True:
        amount_str = await asyncio.to_thread(
            input,
            f"{Fore.CYAN}{prompt_prefix} ({market_info['symbol'].split('/')[0]}): {Style.RESET_ALL}",
        )  # Show base asset
        amount_str = amount_str.strip()
        if not amount_str:
            print_color("Amount cannot be empty.", Fore.YELLOW)
            continue
        try:
            amount = decimal.Decimal(amount_str)
            min_amount, amount_prec, amount_step = (
                market_info.get("min_amount", decimal.Decimal("0")),
                market_info["amount_precision"],
                market_info.get("amount_step", decimal.Decimal("0")),
            )
            if amount <= 0:
                print_color("Amount must be positive.", Fore.YELLOW)
                continue
            if min_amount > 0 and amount < min_amount:
                print_color(
                    f"Amount < min ({format_decimal(min_amount, amount_prec)}).",
                    Fore.YELLOW,
                )
                continue
            # Check step but don't auto-round here, just return valid string
            if amount_step > 0 and (amount % amount_step) != 0:
                quantized_amount = (amount // amount_step) * amount_step
                print_color(
                    f"Warning: Amount ({amount}) doesn't meet step ({amount_step}). Nearest valid: {quantized_amount}. Exchange might reject or round.",
                    Fore.YELLOW,
                )
            return format_decimal(
                amount, amount_prec
            )  # Return formatted valid amount string
        except (decimal.InvalidOperation, ValueError):
            print_color("Invalid number format.", Fore.RED)
        except (EOFError, KeyboardInterrupt):
            print_color("\nCancelled.", Fore.YELLOW)
            return None


async def get_price_from_user_async(
    config: dict, market_info: dict, prompt_text: str = "Limit price"
) -> str | None:
    """Async version to get validated price string."""
    while True:
        price_str = await asyncio.to_thread(
            input,
            f"{Fore.CYAN}Enter {prompt_text} ({market_info['symbol'].split(':')[0].split('/')[1]}): {Style.RESET_ALL}",
        )  # Show quote asset
        price_str = price_str.strip()
        if not price_str:
            print_color("Price cannot be empty.", Fore.YELLOW)
            continue
        try:
            price = decimal.Decimal(price_str)
            price_prec, price_tick = (
                market_info["price_precision"],
                market_info["price_tick_size"],
            )
            if price <= 0:
                print_color("Price must be positive.", Fore.YELLOW)
                continue
            if price_tick > 0 and (price % price_tick) != 0:
                quantized_price = price.quantize(
                    price_tick, rounding=decimal.ROUND_HALF_UP
                )  # Suggest nearest valid
                print_color(
                    f"Warning: Price ({price}) may not meet tick size ({price_tick}). Nearest valid: {quantized_price}. Exchange might reject or round.",
                    Fore.YELLOW,
                )
            return format_decimal(
                price, price_prec
            )  # Return formatted valid price string
        except (decimal.InvalidOperation, ValueError):
            print_color("Invalid number format.", Fore.RED)
        except (EOFError, KeyboardInterrupt):
            print_color("\nCancelled.", Fore.YELLOW)
            return None


async def user_confirm_async(prompt: str) -> bool:
    """Async version to get user confirmation."""
    try:
        confirm = await asyncio.to_thread(input, prompt)
        return confirm.strip().lower() in ["yes", "y"]
    except (EOFError, KeyboardInterrupt):
        print_color("\nCancelled.", Fore.YELLOW)
        return False


# Make TP/SL input async as well
async def get_tp_sl_from_user_async(
    market_info: dict, side: str
) -> tuple[decimal.Decimal | None, decimal.Decimal | None]:
    tp_price, sl_price = None, None
    market_info["price_precision"]
    quote_asset = market_info["symbol"].split(":")[0].split("/")[1]

    try:
        tp_input = await asyncio.to_thread(
            input,
            f"{Fore.CYAN}Take Profit price ({quote_asset}, optional, 'n' skip): {Style.RESET_ALL}",
        )
        tp_input = tp_input.strip().lower()
        if tp_input not in ["n", "no", ""]:
            try:
                tp_price = decimal.Decimal(tp_input)
                assert tp_price > 0
            except:
                print_color("Invalid TP price format.", Fore.YELLOW)
                tp_price = None

        sl_input = await asyncio.to_thread(
            input,
            f"{Fore.CYAN}Stop Loss price ({quote_asset}, optional, 'n' skip): {Style.RESET_ALL}",
        )
        sl_input = sl_input.strip().lower()
        if sl_input not in ["n", "no", ""]:
            try:
                sl_price = decimal.Decimal(sl_input)
                assert sl_price > 0
            except:
                print_color("Invalid SL price format.", Fore.YELLOW)
                sl_price = None

        # Basic validation: TP > SL for buy, TP < SL for sell
        if tp_price and sl_price:
            ticker = latest_data.get("ticker")
            last_p = (
                decimal.Decimal(str(ticker["last"]))
                if ticker and ticker.get("last")
                else None
            )

            valid = True
            if side == "buy":
                if tp_price <= sl_price:
                    print_color("TP must be > SL for Buy.", Fore.YELLOW)
                    valid = False
                if last_p and (tp_price <= last_p):
                    print_color(
                        f"Warning: Buy TP ({tp_price}) <= Last Price ({last_p}).",
                        Fore.YELLOW,
                    )
                if last_p and (sl_price >= last_p):
                    print_color(
                        f"Warning: Buy SL ({sl_price}) >= Last Price ({last_p}).",
                        Fore.YELLOW,
                    )
            elif side == "sell":
                if tp_price >= sl_price:
                    print_color("TP must be < SL for Sell.", Fore.YELLOW)
                    valid = False
                if last_p and (tp_price >= last_p):
                    print_color(
                        f"Warning: Sell TP ({tp_price}) >= Last Price ({last_p}).",
                        Fore.YELLOW,
                    )
                if last_p and (sl_price <= last_p):
                    print_color(
                        f"Warning: Sell SL ({sl_price}) <= Last Price ({last_p}).",
                        Fore.YELLOW,
                    )

            if not valid:
                tp_price = sl_price = None  # Invalidate if basic logic fails

    except (EOFError, KeyboardInterrupt):
        print_color("\nTP/SL entry cancelled.", Fore.YELLOW)
        return None, None

    return tp_price, sl_price


async def cancel_order_async(
    exchange_pro: ccxtpro.Exchange, symbol: str, order_id: str
) -> None:
    if CONFIG["PAPER_TRADING_ENABLED"]:
        global paper_orders
        initial_len = len(paper_orders)
        paper_orders = [o for o in paper_orders if o["id"] != order_id]
        if len(paper_orders) < initial_len:
            print_color(
                f"{Fore.MAGENTA}# Paper:{Style.RESET_ALL} Order {order_id} cancelled.",
                Fore.CYAN,
            )
            termux_toast(f"Paper Order {order_id} Cancelled")
        else:
            print_color(
                f"{Fore.YELLOW}# Paper: Order {order_id} not found.", Fore.YELLOW
            )
    else:
        try:
            print_color(
                f"{Fore.CYAN}# Cancelling order {order_id}...{Style.DIM}", end=" "
            )
            # --- Add category for Bybit V5 ---
            cancel_params = {"category": CONFIG["EXCHANGE_TYPE"]}
            await exchange_pro.cancel_order(order_id, symbol, params=cancel_params)
            print_color(f"{Fore.GREEN}Done.{Style.RESET_ALL}")
            print_color(f"{Fore.GREEN}Cancel request sent for {order_id}.", Fore.CYAN)
            termux_toast(f"Cancel Order {order_id} Sent")
        except ccxt.OrderNotFound:
            print_color(
                f"{Fore.RED}Failed. Order {order_id} not found.{Style.RESET_ALL}",
                Fore.YELLOW,
            )
        except Exception as e:
            print_color(f"{Fore.RED}Failed. Error: {e}{Style.RESET_ALL}", Fore.RED)
            termux_toast(f"Cancel Order Failed: {e}", "long")
            traceback.print_exc()


async def cancel_all_orders_async(exchange_pro: ccxtpro.Exchange, symbol: str) -> None:
    if CONFIG["PAPER_TRADING_ENABLED"]:
        global paper_orders
        removed_count = len(paper_orders)
        if removed_count > 0:
            paper_orders = []
            print_color(
                f"{Fore.MAGENTA}# Paper:{Style.RESET_ALL} All {removed_count} open orders for {symbol} cancelled.",
                Fore.CYAN,
            )
            termux_toast("Paper All Orders Cancelled")
        else:
            print_color(
                f"{Fore.YELLOW}# Paper: No open orders for {symbol} to cancel.",
                Fore.YELLOW,
            )
    else:
        try:
            print_color(
                f"{Fore.CYAN}# Cancelling all orders for {symbol}...{Style.DIM}",
                end=" ",
            )
            # --- Add category for Bybit V5 ---
            cancel_params = {"category": CONFIG["EXCHANGE_TYPE"]}
            await exchange_pro.cancel_all_orders(symbol, params=cancel_params)
            print_color(f"{Fore.GREEN}Done.{Style.RESET_ALL}")
            print_color(f"{Fore.GREEN}Cancel ALL request sent for {symbol}.", Fore.CYAN)
            termux_toast(f"Cancel ALL Sent for {symbol}")
        except Exception as e:
            print_color(f"{Fore.RED}Failed. Error: {e}{Style.RESET_ALL}", Fore.RED)
            termux_toast(f"Cancel All Failed: {e}", "long")
            traceback.print_exc()


# ==============================================================================
# Main Menu and Application Logic
# ==============================================================================
async def main_menu(
    exchange_pro: ccxtpro.Exchange,
    market_info: dict,
    config: dict,
    ask_map: dict,
    bid_map: dict,
) -> str:
    global paper_balance  # Allow modification if needed (e.g., add funds feature)
    while True:
        # No need to check paper fills here, dedicated task handles it
        # if CONFIG["PAPER_TRADING_ENABLED"]: await check_paper_order_fills()
        print_color("\n" + "=" * 85, Fore.CYAN)
        print_color(
            f"{Fore.MAGENTA}{Style.BRIGHT}🔮 Pyrmethus Bybit Terminal v3.3 🔮 {Style.NORMAL}{Fore.CYAN} - {market_info['symbol']} - {exchange_pro.name} {exchange_pro.options.get('defaultType', '').upper()} {Fore.YELLOW}{Style.BRIGHT}{'(PAPER TRADING)' if config['PAPER_TRADING_ENABLED'] else ''}{Style.RESET_ALL}",
            Fore.CYAN,
        )
        print_color("=" * 85 + "\n", Fore.CYAN)
        ask_map, bid_map = display_combined_analysis_async(
            latest_data, market_info, config
        )
        print_color(f"{Fore.CYAN}Choose Action:{Style.RESET_ALL}")
        menu_options = [
            "1: Force Refresh Data",
            "2: Market Order",
            "3: Limit Order",
            "4: Interactive Limit",
            "5: Trailing Stop",
            "6: Cancel Order",
            "7: Cancel All",
            "8: Set Symbol",
            "9: Toggle Paper Mode",
            "0: Exit",
        ]
        for option in menu_options:
            print_color(f"  {Fore.YELLOW}{option}{Style.RESET_ALL}")
        choice = await asyncio.to_thread(
            input, f"{Fore.WHITE}Enter option: {Style.RESET_ALL}"
        )
        choice = choice.strip()
        print_color(f"{Fore.CYAN}# Processing...{Style.RESET_ALL}", end="\r")
        await asyncio.sleep(0.1)
        sys.stdout.write("\033[K")

        if choice == "1":
            print_color(f"{Fore.CYAN}# Forcing REST data fetch...{Style.RESET_ALL}")
            # Trigger a one-off fetch by resetting the OHLCV timer and calling directly
            # Note: This is a simplified trigger; task runs on its own schedule anyway
            # A more direct approach would involve signalling the task.
            latest_data["last_update_times"]["indicator_ohlcv"] = (
                0  # Force OHLCV fetch next cycle
            )
            latest_data["last_update_times"]["pivot_ohlcv"] = 0
            # Manually trigger fetch now (will run alongside the scheduled one)
            asyncio.create_task(
                fetch_periodic_data(exchange_pro, market_info["symbol"])
            )
            await asyncio.sleep(1)  # Give it a moment to start

        elif choice in ["2", "3", "4", "5"]:
            side_input = await asyncio.to_thread(
                input, f"{Fore.CYAN}Buy or Sell (b/s): {Style.RESET_ALL}"
            )
            side = (
                "buy"
                if side_input.strip().lower() == "b"
                else "sell"
                if side_input.strip().lower() == "s"
                else None
            )
            if not side:
                print_color("Invalid side.", Fore.YELLOW)
                continue

            if choice == "2":  # Market
                amount_str = await get_amount_from_user_async(
                    config, market_info, "Market Amount"
                )
                if amount_str:
                    await place_market_order_async(
                        exchange_pro,
                        market_info["symbol"],
                        side,
                        amount_str,
                        market_info,
                        config,
                    )
            elif choice == "3":  # Limit
                amount_str = await get_amount_from_user_async(
                    config, market_info, "Limit Amount"
                )
                if amount_str:
                    price_str = await get_price_from_user_async(
                        config, market_info, "Limit Price"
                    )
                    if price_str:
                        await place_limit_order_async(
                            exchange_pro,
                            market_info["symbol"],
                            side,
                            amount_str,
                            price_str,
                            market_info,
                            config,
                        )
            elif choice == "4":  # Interactive Limit
                amount_str = await get_amount_from_user_async(
                    config, market_info, "Interactive Limit Amount"
                )
                if amount_str:
                    # Get current OB maps directly from latest_data
                    latest_ob = latest_data.get("orderbook")
                    current_ask_map, current_bid_map = {}, {}
                    if latest_ob and latest_ob.get("asks") and latest_ob.get("bids"):
                        # Rebuild maps with correct numbering (A1 = best ask, B1 = best bid)
                        display_asks_reversed = list(reversed(latest_ob["asks"]))
                        current_ask_map = {
                            idx + 1: ask["price"]
                            for idx, ask in enumerate(
                                display_asks_reversed[
                                    : config["MAX_ORDERBOOK_DEPTH_DISPLAY"]
                                ]
                            )
                        }
                        current_bid_map = {
                            idx + 1: bid["price"]
                            for idx, bid in enumerate(
                                latest_ob["bids"][
                                    : config["MAX_ORDERBOOK_DEPTH_DISPLAY"]
                                ]
                            )
                        }
                    else:
                        print_color(
                            "Orderbook data not available for interactive selection.",
                            Fore.YELLOW,
                        )
                        continue
                    await place_limit_order_interactive_async(
                        exchange_pro,
                        market_info["symbol"],
                        side,
                        amount_str,
                        market_info,
                        config,
                        current_ask_map,
                        current_bid_map,
                    )
            elif choice == "5":  # Trailing Stop
                await place_trailing_stop_order_async(
                    exchange_pro, market_info["symbol"], side, market_info, config
                )

        elif choice == "6":
            order_id = await asyncio.to_thread(
                input, f"{Fore.CYAN}Enter Order ID to cancel: {Style.RESET_ALL}"
            )
            if order_id.strip():
                await cancel_order_async(
                    exchange_pro, market_info["symbol"], order_id.strip()
                )
            else:
                print_color("Order ID cannot be empty.", Fore.YELLOW)
        elif choice == "7":
            if await user_confirm_async(
                f"{Style.BRIGHT}{Fore.YELLOW}Cancel ALL open orders for {market_info['symbol']}? (yes/no): {Style.RESET_ALL}"
            ):
                await cancel_all_orders_async(exchange_pro, market_info["symbol"])
            else:
                print_color("Cancellation aborted.", Fore.CYAN)
        elif choice == "8":
            new_symbol_input = await asyncio.to_thread(
                input, f"{Fore.CYAN}New symbol (e.g., ETH/USDT:USDT): {Style.RESET_ALL}"
            )
            new_symbol = new_symbol_input.strip().upper()
            if new_symbol:
                # Basic validation (presence of '/', ':')
                if "/" in new_symbol and ":" in new_symbol:
                    if new_symbol != CONFIG["SYMBOL"]:
                        CONFIG["SYMBOL"] = new_symbol
                        print_color(
                            f"Switching to {Fore.MAGENTA}{CONFIG['SYMBOL']}{Style.RESET_ALL}. Restarting watchers...",
                            Fore.CYAN,
                        )
                        return "symbol_change"
                    else:
                        print_color("Symbol already set to this value.", Fore.YELLOW)
                else:
                    print_color(
                        "Invalid symbol format (e.g., BASE/QUOTE:SETTLE).", Fore.YELLOW
                    )
            else:
                print_color("Symbol cannot be empty.", Fore.YELLOW)
        elif choice == "9":
            CONFIG["PAPER_TRADING_ENABLED"] = not CONFIG["PAPER_TRADING_ENABLED"]
            mode_str = (
                f"{Fore.GREEN}ENABLED{Style.RESET_ALL}"
                if CONFIG["PAPER_TRADING_ENABLED"]
                else f"{Fore.YELLOW}DISABLED{Style.RESET_ALL}"
            )
            print_color(
                f"Paper Trading Mode toggled to: {mode_str}. Restarting...", Fore.CYAN
            )
            return "mode_change"
        elif choice == "0":
            print_color("Exiting...", Fore.MAGENTA)
            return "exit"
        else:
            print_color("Invalid option.", Fore.RED)
        await asyncio.sleep(0.5)  # Short pause after action before redisplaying


# ==============================================================================
# Main Function - Initialize and Run the Terminal
# ==============================================================================
async def main() -> None:
    global exchange, latest_data, paper_balance, paper_positions, paper_orders, paper_trade_log  # Added paper_trade_log
    active_tasks = []

    while True:  # Main loop for handling restarts (symbol/mode change)
        # --- Cleanup previous session ---
        print_color(
            f"{Fore.CYAN}# Cleaning up previous tasks...{Style.RESET_ALL}", Fore.CYAN
        )
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(
                *active_tasks, return_exceptions=True
            )  # Wait for cancellations
            verbose_print("Previous tasks cancelled.")
        active_tasks = []
        if exchange:
            try:
                verbose_print("Closing previous exchange connection...")
                await exchange.close()
                verbose_print("Previous exchange connection closed.")
            except Exception as e:
                verbose_print(f"Error closing prev exchange: {e}")
        exchange = None
        # Reset shared data relevant to the symbol/mode
        latest_data = {
            "ticker": None,
            "orderbook": None,
            "balance": None,
            "positions": [],
            "open_orders": [],
            "indicator_ohlcv": None,
            "pivot_ohlcv": None,
            "indicators": {},
            "pivots": None,
            "market_info": None,
            "last_update_times": {},
            "connection_status": {"ws_ticker": "init", "ws_ob": "init", "rest": "init"},
        }

        # --- Initialize new session ---
        symbol = CONFIG["SYMBOL"]
        print_color(
            f"\n{Fore.CYAN}{Style.BRIGHT}# Initializing Pyrmethus Terminal v3.3 for {Fore.MAGENTA}{Style.BRIGHT}{symbol}{Style.RESET_ALL}...{Style.RESET_ALL}",
            Fore.CYAN,
        )
        print_color(
            f"{Fore.YELLOW}# Paper Trading: {'ENABLED' if CONFIG['PAPER_TRADING_ENABLED'] else 'DISABLED'}{Style.RESET_ALL}"
        )

        # --- Connect Exchange ---
        try:
            exchange_config = {
                "apiKey": CONFIG["API_KEY"],
                "secret": CONFIG["API_SECRET"],
                "options": {
                    "defaultType": CONFIG["EXCHANGE_TYPE"],  # 'linear', 'inverse'
                    "adjustForTimeDifference": True,
                    "recvWindow": 10000,  # Increase recvWindow potentially
                    # V5 Specific options if needed, category handled per-call mostly
                    # 'brokerId': 'YOUR_BROKER_ID', # If applicable
                },
                "enableRateLimit": True,
                "userAgent": "PyrmethusTerminal/3.3",
                "timeout": CONFIG["CONNECT_TIMEOUT"],
            }
            exchange = ccxtpro.bybit(exchange_config)

            # --- Set Sandbox Mode ---
            if CONFIG["PAPER_TRADING_ENABLED"]:
                # Check if sandbox mode is supported and set it
                if hasattr(exchange, "set_sandbox_mode"):
                    exchange.set_sandbox_mode(True)
                    print_color(f"{Fore.CYAN}# Sandbox Mode ENABLED.{Style.RESET_ALL}")
                else:
                    print_color(
                        f"{Fore.YELLOW}# Warning: exchange.set_sandbox_mode not available in this ccxt version for Bybit.{Style.RESET_ALL}"
                    )
                    # Rely on testnet API keys if set_sandbox_mode is unavailable
                    if "test" not in CONFIG.get("API_KEY", "").lower():
                        print_color(
                            f"{Fore.RED}# Ensure using TESTNET API Keys for Paper Trading if set_sandbox_mode fails.{Style.RESET_ALL}"
                        )

            # --- Load Markets ---
            print_color(
                f"{Fore.CYAN}# Loading market data...{Style.RESET_ALL}", end="\r"
            )
            await exchange.load_markets()
            sys.stdout.write("\033[K")  # Clear line
            print_color(f"{Fore.CYAN}# Market data loaded.{Style.RESET_ALL}")

        except Exception as e:
            print_color(
                f"{Fore.RED}{Style.BRIGHT}FATAL: Failed exchange initialization: {e}{Style.RESET_ALL}"
            )
            traceback.print_exc()
            print_color("# Retrying in 15 seconds...", Fore.YELLOW)
            await asyncio.sleep(15)
            continue  # Restart the main initialization loop

        # --- Fetch Initial Market Info ---
        latest_data["market_info"] = await get_market_info(exchange, symbol)
        if not latest_data["market_info"]:
            print_color(
                f"{Fore.RED}FATAL: Failed to get market info for {symbol}. Check symbol/network & restart.{Style.RESET_ALL}",
                Fore.RED,
            )
            if exchange:
                await exchange.close()
            return  # Exit script if essential market info fails

        # --- Initialize Paper/Live State ---
        if CONFIG["PAPER_TRADING_ENABLED"]:
            paper_balance = decimal.Decimal(str(CONFIG["PAPER_INITIAL_BALANCE"]))
            paper_positions = {}
            paper_orders = []
            paper_trade_log = []
            latest_data.update(
                {"balance": paper_balance, "positions": [], "open_orders": []}
            )
            print_color(
                f"# Paper Balance Initialized: {paper_balance} {CONFIG['FETCH_BALANCE_ASSET']}",
                Fore.CYAN,
            )
        else:  # Reset paper state if switching to live
            paper_positions, paper_orders, paper_trade_log = {}, [], []

        # --- Start Background Tasks ---
        print_color(
            f"{Fore.CYAN}# Summoning watchers & fetchers...{Style.RESET_ALL}", Fore.CYAN
        )
        try:
            task_ticker = asyncio.create_task(watch_ticker(exchange, symbol))
            task_orderbook = asyncio.create_task(watch_orderbook(exchange, symbol))
            task_periodic = asyncio.create_task(fetch_periodic_data(exchange, symbol))
            task_paper_fill = asyncio.create_task(
                paper_fill_checker_task()
            )  # Runs regardless, but only acts if paper mode enabled
            active_tasks = [task_ticker, task_orderbook, task_periodic, task_paper_fill]
            print_color(f"{Fore.GREEN}# Watchers deployed.{Style.RESET_ALL}")
            await asyncio.sleep(3)  # Allow connections & initial data fetch
        except Exception as e:
            print_color(
                f"{Fore.RED}{Style.BRIGHT}FATAL: Failed to start background tasks: {e}{Style.RESET_ALL}"
            )
            traceback.print_exc()
            # Cleanup before exiting/retrying
            for task in active_tasks:
                task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            if exchange:
                await exchange.close()
            await asyncio.sleep(5)
            continue  # Retry initialization

        # --- Run Main Menu ---
        menu_result = "continue"
        try:
            # Initial display before menu loop
            display_combined_analysis_async(
                latest_data, latest_data["market_info"], CONFIG
            )
            menu_result = await main_menu(
                exchange, latest_data["market_info"], CONFIG, {}, {}
            )
        except KeyboardInterrupt:
            menu_result = "exit"
        except Exception as e:
            print_color(
                f"\n{Fore.RED}{Style.BRIGHT}--- CRITICAL MENU FAILURE ---{Style.RESET_ALL}",
                Fore.RED,
            )
            print_color(f"{traceback.format_exc()}", Fore.RED)
            print_color(f"Error: {e}", Fore.RED)
            print_color("--- State at failure ---", Fore.YELLOW)
            verbose_print(f"Latest Data: {latest_data}")  # Print state if verbose
            menu_result = "exit"  # Exit on unhandled menu errors

        # --- Process Menu Result ---
        if menu_result == "exit":
            print_color(
                f"\n{Fore.MAGENTA}{Style.BRIGHT}# Initiating shutdown...{Style.RESET_ALL}"
            )
            # Cleanup handled at the start of the loop, break now
            break
        elif menu_result in ["symbol_change", "mode_change"]:
            print_color(
                f"{Fore.YELLOW}# Re-initializing for {menu_result}...{Style.RESET_ALL}"
            )
            # Loop will continue and cleanup/re-init
            await asyncio.sleep(1)
        else:  # Should not happen, but catch unexpected results
            print_color(
                f"{Fore.RED}# Unexpected state ({menu_result}). Exiting.{Style.RESET_ALL}"
            )
            break  # Exit

    # Final cleanup after loop breaks
    print_color(f"{Fore.CYAN}# Final cleanup...{Style.RESET_ALL}", Fore.CYAN)
    for task in active_tasks:
        task.cancel()
    if active_tasks:
        await asyncio.gather(*active_tasks, return_exceptions=True)
    if exchange:
        try:
            await exchange.close()
        except Exception as e:
            verbose_print(f"Error during final exchange close: {e}")
    print_color(f"{Fore.MAGENTA}{Style.BRIGHT}# Terminal Closed.{Style.RESET_ALL}")


async def paper_fill_checker_task() -> None:
    """Dedicated task to periodically check paper order fills."""
    while True:
        await asyncio.sleep(CONFIG["PAPER_FILL_CHECK_INTERVAL"])  # Check interval first
        if CONFIG["PAPER_TRADING_ENABLED"]:
            try:
                # Ensure necessary data is available before checking fills
                if latest_data.get("ticker") and latest_data.get("market_info"):
                    await check_paper_order_fills()
                # else:
                #     verbose_print("Skipping paper fill check - ticker or market_info missing.")
            except Exception as e:
                print_color(f"# Paper Fill Check Error: {e}", Fore.YELLOW)
                # traceback.print_exc() # Optionally print stack trace for debugging


if __name__ == "__main__":
    print_color(
        f"{Style.BRIGHT}{Fore.CYAN}--- Pyrmethus Bybit Terminal ---{Style.RESET_ALL}"
    )
    # Validate API keys only if NOT in paper trading mode
    if not CONFIG["PAPER_TRADING_ENABLED"] and (
        not CONFIG["API_KEY"] or not CONFIG["API_SECRET"]
    ):
        print_color(
            f"{Fore.RED}{Style.BRIGHT}ERROR: API_KEY/SECRET missing in .env for LIVE trading.{Style.RESET_ALL}"
        )
        print_color(
            f"{Fore.YELLOW}Set 'PAPER_TRADING_ENABLED=true' in .env or add API keys to run.{Style.RESET_ALL}"
        )
        sys.exit(1)
    # Check Exchange Type validity
    if CONFIG["EXCHANGE_TYPE"] not in [
        "linear",
        "inverse",
        "spot",
        "option",
    ]:  # Add 'spot' if you intend to support it
        print_color(
            f"{Fore.RED}{Style.BRIGHT}ERROR: Invalid BYBIT_EXCHANGE_TYPE '{CONFIG['EXCHANGE_TYPE']}' in .env. Use 'linear' or 'inverse'.{Style.RESET_ALL}"
        )
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_color(
            f"\n{Fore.CYAN}# Shutdown signal received. Farewell!{Style.RESET_ALL}"
        )
    except Exception as e:
        print_color(
            f"\n{Fore.RED}{Style.BRIGHT}# Unhandled Top-Level Error: {e}{Style.RESET_ALL}"
        )
        traceback.print_exc()
    finally:
        pass  # Final reset
