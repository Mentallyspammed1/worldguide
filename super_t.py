#!/usr/bin/env python

"""High-Frequency Trading Bot (Scalping) using Dual Supertrend, ATR, Volume, and Order Book Analysis.

Disclaimer:
- HIGH RISK: Scalping bots are extremely high-risk due to market noise, latency,
  slippage, fees, and the need for precise execution.
- Parameter Sensitivity: Requires significant tuning and backtesting. Defaults are illustrative.
- Rate Limits: Frequent API calls (esp. order book) can hit limits. Monitor usage.
- Slippage: Market orders are prone to slippage.
- Test Thoroughly: DO NOT RUN LIVE WITHOUT EXTENSIVE TESTNET/DEMO TESTING.
"""

import logging
import os
import sys
import time
import traceback
from typing import Any

import ccxt
import pandas as pd
from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv()

# API Credentials
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# --- Scalping Specific Configuration ---
DEFAULT_SYMBOL = os.getenv("SYMBOL", "BTC/USDT:USDT")  # CCXT unified format
DEFAULT_INTERVAL = os.getenv("INTERVAL", "1m")  # Scalping timeframe (e.g., 1m, 3m)
DEFAULT_LEVERAGE = int(os.getenv("LEVERAGE", 10))  # Caution with high leverage
DEFAULT_SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", 10))  # Check frequency

# Risk Management (Critical for Scalping - TUNE CAREFULLY!)
RISK_PER_TRADE_PERCENTAGE = float(os.getenv("RISK_PER_TRADE_PERCENTAGE", 0.005))  # 0.5% risk
ATR_STOP_LOSS_MULTIPLIER = float(os.getenv("ATR_STOP_LOSS_MULTIPLIER", 1.5))  # SL = 1.5 * ATR
ATR_TAKE_PROFIT_MULTIPLIER = float(os.getenv("ATR_TAKE_PROFIT_MULTIPLIER", 2.0))  # TP = 2.0 * ATR
MAX_ORDER_USDT_AMOUNT = float(os.getenv("MAX_ORDER_USDT_AMOUNT", 500.0))  # Cap exposure

# Supertrend Indicator Parameters (Adjust for Scalping)
DEFAULT_ST_ATR_LENGTH = int(os.getenv("ST_ATR_LENGTH", 7))  # Shorter period
DEFAULT_ST_MULTIPLIER = float(os.getenv("ST_MULTIPLIER", 2.5))
CONFIRM_ST_ATR_LENGTH = int(os.getenv("CONFIRM_ST_ATR_LENGTH", 5))  # Shorter period
CONFIRM_ST_MULTIPLIER = float(os.getenv("CONFIRM_ST_MULTIPLIER", 2.0))

# Volume Analysis Parameters
VOLUME_MA_PERIOD = int(os.getenv("VOLUME_MA_PERIOD", 20))
VOLUME_SPIKE_THRESHOLD = float(os.getenv("VOLUME_SPIKE_THRESHOLD", 1.5))  # Vol > 1.5x MA
REQUIRE_VOLUME_SPIKE_FOR_ENTRY = os.getenv("REQUIRE_VOLUME_SPIKE_FOR_ENTRY", "true").lower() == "true"

# Order Book Analysis Parameters
ORDER_BOOK_DEPTH = int(os.getenv("ORDER_BOOK_DEPTH", 10))  # Analysis depth
ORDER_BOOK_RATIO_THRESHOLD_LONG = float(os.getenv("ORDER_BOOK_RATIO_THRESHOLD_LONG", 1.2))  # Bid > 1.2x Ask
ORDER_BOOK_RATIO_THRESHOLD_SHORT = float(os.getenv("ORDER_BOOK_RATIO_THRESHOLD_SHORT", 0.8))  # Ask > 1.2x Bid
FETCH_ORDER_BOOK_PER_CYCLE = os.getenv("FETCH_ORDER_BOOK_PER_CYCLE", "false").lower() == "true"  # <<< OPTIMIZATION

# ATR Calculation Parameter
ATR_CALCULATION_PERIOD = int(os.getenv("ATR_CALCULATION_PERIOD", 14))

# CCXT / API Parameters
DEFAULT_RECV_WINDOW = 10000
ORDER_BOOK_FETCH_LIMIT = max(25, ORDER_BOOK_DEPTH)  # Min levels often required by API
SHALLOW_OB_FETCH_DEPTH = 5  # For quick best bid/ask check

# --- Constants ---
SIDE_BUY = "buy"
SIDE_SELL = "sell"
POSITION_SIDE_LONG = "Long"
POSITION_SIDE_SHORT = "Short"
POSITION_SIDE_NONE = "None"
USDT_SYMBOL = "USDT"
RETRY_COUNT = 3
RETRY_DELAY_SECONDS = 1  # Short delay for scalping

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for max verbosity
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def log_success(self, message, *args, **kwargs) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = log_success
logging.addLevelName(logging.INFO, f"\033[96m{logging.getLevelName(logging.INFO)}\033[0m")
logging.addLevelName(logging.WARNING, f"\033[93m{logging.getLevelName(logging.WARNING)}\033[0m")
logging.addLevelName(logging.ERROR, f"\033[91m{logging.getLevelName(logging.ERROR)}\033[0m")
logging.addLevelName(logging.CRITICAL, f"\033[91m\033[1m{logging.getLevelName(logging.CRITICAL)}\033[0m")
logging.addLevelName(SUCCESS_LEVEL, f"\033[92m{logging.getLevelName(SUCCESS_LEVEL)}\033[0m")


# logging.getLogger().setLevel(logging.DEBUG) # Uncomment for extreme detail
# logging.addLevelName(logging.DEBUG, f"\033[90m{logging.getLevelName(logging.DEBUG)}\033[0m")


# --- Exchange Initialization ---
def initialize_exchange() -> ccxt.Exchange | None:
    """Initializes and returns the CCXT Bybit exchange instance."""
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        logger.critical("CRITICAL: API keys missing in .env file.")
        return None
    try:
        exchange = ccxt.bybit(
            {
                "apiKey": BYBIT_API_KEY,
                "secret": BYBIT_API_SECRET,
                "enableRateLimit": True,
                "enableTimeSync": True,
                "options": {"defaultType": "swap", "recvWindow": DEFAULT_RECV_WINDOW},
            }
        )
        exchange.load_markets()
        exchange.fetch_balance()  # Test authentication
        logger.info("CCXT Bybit Session Initialized (LIVE FUTURES SCALPING MODE - EXTREME CAUTION!).")
        return exchange
    except ccxt.AuthenticationError as e:
        logger.critical(f"Authentication failed: {e}")
    except ccxt.NetworkError as e:
        logger.critical(f"Network error during init: {e}")
    except ccxt.ExchangeError as e:
        logger.critical(f"Exchange error during init: {e}")
    except Exception as e:
        logger.critical(f"Failed to initialize exchange: {e}")
        logger.debug(traceback.format_exc())
    return None


# --- Indicator Calculation ---
def calculate_supertrend(df: pd.DataFrame, length: int, multiplier: float, prefix: str = "") -> pd.DataFrame:
    """Calculates Supertrend indicator using pandas_ta."""
    col_prefix = f"{prefix}" if prefix else ""
    st_col_name = f"SUPERT_{length}_{multiplier}"
    st_trend_col = f"SUPERTd_{length}_{multiplier}"
    target_cols = [f"{col_prefix}supertrend", f"{col_prefix}trend", f"{col_prefix}st_long", f"{col_prefix}st_short"]

    if df is None or df.empty or not all(c in df.columns for c in ["high", "low", "close"]):
        logger.warning(f"Insufficient data/columns for {col_prefix}Supertrend.")
        for col in target_cols:
            df[col] = pd.NA
        return df

    try:
        df.ta.supertrend(length=length, multiplier=multiplier, append=True)
        if st_col_name not in df.columns or st_trend_col not in df.columns:
            raise KeyError(f"pandas_ta did not create expected columns: {st_col_name}, {st_trend_col}")

        df[f"{col_prefix}supertrend"] = df[st_col_name]
        df[f"{col_prefix}trend"] = df[st_trend_col] == 1
        prev_trend_direction = df[st_trend_col].shift(1)
        df[f"{col_prefix}st_long"] = (prev_trend_direction == -1) & (df[st_trend_col] == 1)
        df[f"{col_prefix}st_short"] = (prev_trend_direction == 1) & (df[st_trend_col] == -1)
        # Clean up intermediate columns
        cols_to_drop = [c for c in df.columns if c.startswith("SUPERT_") and c not in target_cols]
        df.drop(columns=cols_to_drop, errors="ignore", inplace=True)
        logger.debug(f"Calculated {col_prefix}Supertrend. Last trend: {df[f'{col_prefix}trend'].iloc[-1]}")

    except Exception as e:
        logger.error(f"Error calculating {col_prefix}Supertrend: {e}")
        logger.debug(traceback.format_exc())
        for col in target_cols:
            df[col] = pd.NA
    return df


# --- Volume and ATR Analysis (Corrected Logging) ---
def analyze_volume_atr(df: pd.DataFrame, atr_len: int, vol_ma_len: int) -> dict[str, float | None]:
    """Calculates ATR and Volume MA, checks for volume spikes."""
    results: dict[str, float | None] = {"atr": None, "volume_ma": None, "last_volume": None, "volume_ratio": None}
    if df is None or df.empty or "volume" not in df.columns or len(df) < max(atr_len, vol_ma_len):
        logger.warning("Insufficient data for Volume/ATR analysis.")
        return results

    try:
        # Calculate ATR
        df.ta.atr(length=atr_len, append=True)
        atr_col = f"ATRr_{atr_len}"
        if atr_col in df.columns and not df[atr_col].isnull().all():
            results["atr"] = df[atr_col].iloc[-1]
        else:
            logger.warning(f"Failed to calculate ATR({atr_len}).")
        df.drop(columns=[atr_col], errors="ignore", inplace=True)  # Clean up

        # Calculate Volume MA
        df["volume_ma"] = df["volume"].rolling(window=vol_ma_len, min_periods=vol_ma_len // 2).mean()
        if not df["volume_ma"].isnull().all():
            results["volume_ma"] = df["volume_ma"].iloc[-1]
            results["last_volume"] = df["volume"].iloc[-1]
            if results["volume_ma"] is not None and results["volume_ma"] > 1e-9 and results["last_volume"] is not None:
                results["volume_ratio"] = results["last_volume"] / results["volume_ma"]
            else:
                results["volume_ratio"] = None
        df.drop(columns=["volume_ma"], errors="ignore", inplace=True)  # Clean up

        # Corrected Logging
        atr_val = results.get("atr")
        atr_str = f"{atr_val:.5f}" if atr_val is not None else "N/A"
        logger.debug(f"ATR({atr_len}): {atr_str}")

        last_vol_val = results.get("last_volume")
        vol_ma_val = results.get("volume_ma")
        vol_ratio_val = results.get("volume_ratio")
        last_vol_str = f"{last_vol_val:.2f}" if last_vol_val is not None else "N/A"
        vol_ma_str = f"{vol_ma_val:.2f}" if vol_ma_val is not None else "N/A"
        vol_ratio_str = f"{vol_ratio_val:.2f}" if vol_ratio_val is not None else "N/A"
        logger.debug(f"Volume Analysis: Last={last_vol_str}, MA({vol_ma_len})={vol_ma_str}, Ratio={vol_ratio_str}")
        # End Corrected Logging

    except Exception as e:
        logger.error(f"Error in Volume/ATR analysis: {e}")
        logger.debug(traceback.format_exc())
    return results


# --- Order Book Analysis ---
def analyze_order_book(exchange: ccxt.Exchange, symbol: str, depth: int) -> dict[str, float | None]:
    """Fetches L2 order book and analyzes bid/ask pressure and spread."""
    results: dict[str, float | None] = {"bid_ask_ratio": None, "spread": None, "best_bid": None, "best_ask": None}
    fetch_limit = max(25, depth)  # Exchange minimums often apply
    logger.debug(f"Fetching order book for {symbol} (Depth: {depth}, Fetch Limit: {fetch_limit})...")
    if not exchange.has["fetchL2OrderBook"]:
        logger.warning(f"Exchange {exchange.id} does not support fetchL2OrderBook. Skipping analysis.")
        return results
    try:
        # Fetch order book data
        order_book = exchange.fetch_l2_order_book(symbol, limit=fetch_limit)
        if not order_book or not order_book.get("bids") or not order_book.get("asks"):
            logger.warning(f"Incomplete order book data received for {symbol}.")
            return results

        bids = order_book["bids"]
        asks = order_book["asks"]

        # Get best bid/ask and spread
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        results["best_bid"] = best_bid
        results["best_ask"] = best_ask
        if best_bid > 0 and best_ask > 0:
            results["spread"] = best_ask - best_bid
        logger.debug(
            f"Order Book: Best Bid={best_bid:.4f}, Best Ask={best_ask:.4f}, "
            f"Spread={results.get('spread'):.4f if results.get('spread') else 'N/A'}"
        )

        # Calculate cumulative volume within the specified depth
        bid_volume_sum = sum(bid[1] for bid in bids[:depth])
        ask_volume_sum = sum(ask[1] for ask in asks[:depth])
        logger.debug(
            f"Order Book Analysis (Depth {depth}): "
            f"Total Bid Vol={bid_volume_sum:.4f}, Total Ask Vol={ask_volume_sum:.4f}"
        )

        # Calculate Bid/Ask Ratio
        if ask_volume_sum > 1e-9:  # Avoid division by zero
            results["bid_ask_ratio"] = bid_volume_sum / ask_volume_sum
            logger.debug(f"Order Book Bid/Ask Ratio: {results['bid_ask_ratio']:.3f}")
        else:
            logger.debug("Order Book Bid/Ask Ratio: Undefined (Ask volume is zero)")
    except ccxt.NetworkError as e:
        logger.warning(f"Network error fetching order book: {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Exchange error fetching order book: {e}")
    except Exception as e:
        logger.error(f"Error analyzing order book: {e}")
        logger.debug(traceback.format_exc())
    return results


# --- Data Fetching ---
def get_market_data(exchange: ccxt.Exchange, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame | None:
    """Fetches OHLCV data and returns it as a pandas DataFrame."""
    if not exchange.has["fetchOHLCV"]:
        logger.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return None
    try:
        logger.debug(f"Fetching {limit} OHLCV candles for {symbol} ({interval})...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        if not ohlcv:
            logger.warning(f"No OHLCV data returned for {symbol} ({interval}).")
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        logger.debug(f"Successfully fetched {len(df)} OHLCV candles.")
        return df
    except ccxt.NetworkError as e:
        logger.warning(f"Network error fetching OHLCV: {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Exchange error fetching OHLCV: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching market data: {e}")
        logger.debug(traceback.format_exc())
    return None


# --- Position & Order Management ---
def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    """Fetches the current position details for a given symbol."""
    default_pos = {"side": POSITION_SIDE_NONE, "qty": 0.0, "entry_price": 0.0}
    try:
        logger.debug(f"Fetching position for {symbol}...")
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            # Check for non-zero contracts, handle potential float inaccuracies
            contracts_val = pos.get("contracts")
            if pos.get("symbol") == symbol and contracts_val is not None and abs(float(contracts_val)) > 1e-9:
                side = POSITION_SIDE_LONG if pos.get("side") == "long" else POSITION_SIDE_SHORT
                qty = float(contracts_val)
                entry = float(pos.get("entryPrice") or 0.0)
                logger.info(f"Found active position: {side} {qty:.5f} {symbol} @ Entry={entry:.4f}")
                return {"side": side, "qty": qty, "entry_price": entry}
        logger.info(f"No active position found for {symbol}.")
        return default_pos
    except ccxt.NetworkError as e:
        logger.warning(f"Network error fetching position: {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Exchange error fetching position: {e}")
    except Exception as e:
        logger.error(f"Error fetching position: {e}")
        logger.debug(traceback.format_exc())
    return default_pos


# --- Leverage Setting (Corrected) ---
def set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> bool:
    """Sets leverage for a futures symbol with retries and correct handling."""
    logger.info(f"Attempting to set leverage to {leverage}x for {symbol}...")
    try:
        market = exchange.market(symbol)
        if not market.get("swap", False) and not market.get("future", False):
            logger.error(f"Cannot set leverage for non-futures market: {symbol}")
            return False
    except Exception as e:
        logger.error(f"Failed to get market info for {symbol} during leverage check: {e}")
        return False

    for attempt in range(RETRY_COUNT):
        try:
            exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage successfully set to {leverage}x for {symbol}")
            return True
        except ccxt.ExchangeError as e:
            error_message_lower = str(e).lower()
            # Correctly handle "leverage not modified" case-insensitively
            if "leverage not modified" in error_message_lower:
                logger.info(f"Leverage already set to {leverage}x for {symbol} (Exchange confirmed no change needed).")
                return True  # Success - it's already set correctly

            # Handle other exchange errors
            logger.warning(f"Leverage set attempt {attempt + 1}/{RETRY_COUNT} failed: {e}")
            if attempt < RETRY_COUNT - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"Failed to set leverage for {symbol} after {RETRY_COUNT} attempts due to errors.")
        except Exception as e:
            logger.error(f"Unexpected error setting leverage for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return False  # Exit loop on unexpected error
    return False  # Failed all retries


# --- Close Position ---
def close_position(exchange: ccxt.Exchange, symbol: str, position: dict[str, Any]) -> dict[str, Any] | None:
    """Closes the current position using a market order with reduce_only."""
    if position["side"] == POSITION_SIDE_NONE or position["qty"] <= 0:
        logger.info(f"No position to close for {symbol}.")
        return None

    side_to_close = SIDE_SELL if position["side"] == POSITION_SIDE_LONG else SIDE_BUY
    amount_to_close = position["qty"]
    try:
        amount_str = exchange.amount_to_precision(symbol, amount_to_close)
        amount_float = float(amount_str)
        if amount_float <= 0:
            logger.error(f"Calculated closing amount is zero or negative: {amount_str}")
            return None

        logger.warning(
            f"Attempting CLOSE {position['side']} position: {amount_str} {symbol} via {side_to_close.upper()} MARKET order (reduce_only)..."
        )
        params = {"reduce_only": True}
        order = exchange.create_market_order(symbol, side_to_close, amount_float, params=params)
        logger.success(f"Position CLOSE order placed for {symbol}. Order ID: {order['id']}")
        return order
    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds to close position: {e}")
    except ccxt.NetworkError as e:
        logger.error(f"Network error closing position: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error closing position: {e}")
    except Exception as e:
        logger.error(f"Unexpected error closing position: {e}")
        logger.debug(traceback.format_exc())
    return None


# --- Risk Calculation ---
def calculate_position_size(
    equity: float, risk_per_trade: float, entry_price: float, stop_loss_price: float, leverage: int
) -> tuple[float | None, float | None]:
    """Calculates position size based on risk, entry, and SL price."""
    logger.debug(
        f"Risk Calc Input: Equity={equity:.2f}, Risk%={risk_per_trade:.2%}, "
        f"Entry={entry_price:.4f}, SL={stop_loss_price:.4f}, Lev={leverage}x"
    )
    if entry_price <= 0 or stop_loss_price <= 0 or abs(entry_price - stop_loss_price) < 1e-9:
        logger.error("Invalid prices for position size calculation.")
        return None, None

    risk_amount_usdt = equity * risk_per_trade
    price_difference = abs(entry_price - stop_loss_price)
    quantity = risk_amount_usdt / price_difference
    if quantity <= 0:
        logger.warning("Calculated quantity <= 0.")
        return None, None

    position_value_usdt = quantity * entry_price
    required_margin = position_value_usdt / leverage
    logger.debug(
        f"Risk Calc Output: RiskAmt={risk_amount_usdt:.2f}, PriceDiff={price_difference:.4f} "
        f"=> Qty={quantity:.5f}, Value={position_value_usdt:.2f}, ReqMargin={required_margin:.2f}"
    )
    return quantity, required_margin


# --- Place Order (Improved Entry Price Estimation) ---
def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    risk_percentage: float,
    current_atr: float,
    sl_atr_multiplier: float,
    leverage: int,
) -> dict[str, Any] | None:
    """Calculates position size based on risk/ATR-StopLoss and places market order."""
    logger.info(f"--- Attempting to place {side.upper()} order for {symbol} ---")
    if current_atr is None or current_atr <= 0:
        logger.error("Invalid ATR value provided for risk calculation. Cannot place order.")
        return None
    try:
        # 1. Get Balance and Equity
        balance = exchange.fetch_balance()
        usdt_equity = balance.get("total", {}).get(USDT_SYMBOL, 0.0)
        usdt_free = balance.get("free", {}).get(USDT_SYMBOL, 0.0)
        if usdt_equity <= 0:
            usdt_equity = usdt_free
        if usdt_equity <= 0:
            logger.error("Cannot determine equity.")
            return None
        logger.debug(f"Account Status: Equity={usdt_equity:.2f} USDT, Free={usdt_free:.2f} USDT")

        # 2. Get Market Info & Estimate Entry Price from Fresh OB
        market = exchange.market(symbol)
        logger.debug("Fetching shallow order book for entry price estimate...")
        # Fetch shallow, fresh OB for best bid/ask just before ordering
        ob_data = analyze_order_book(exchange, symbol, SHALLOW_OB_FETCH_DEPTH)
        entry_price_estimate = 0.0
        if side == SIDE_BUY and ob_data.get("best_ask"):
            entry_price_estimate = ob_data["best_ask"]  # Estimate BUY entry at best ask
            logger.debug(f"Using best ASK {entry_price_estimate:.4f} as BUY entry estimate.")
        elif side == SIDE_SELL and ob_data.get("best_bid"):
            entry_price_estimate = ob_data["best_bid"]  # Estimate SELL entry at best bid
            logger.debug(f"Using best BID {entry_price_estimate:.4f} as SELL entry estimate.")
        else:
            # Fallback to ticker if OB fetch failed
            logger.warning("Order book fetch for entry estimate failed, using ticker 'last' price.")
            ticker = exchange.fetch_ticker(symbol)
            entry_price_estimate = ticker.get("last")

        if entry_price_estimate is None or entry_price_estimate <= 0:
            logger.error(f"Invalid entry price estimate for {symbol}: {entry_price_estimate}")
            return None

        # 3. Calculate Stop Loss Price based on Estimated Entry and ATR
        stop_loss_distance = current_atr * sl_atr_multiplier
        stop_loss_price = 0.0
        if side == SIDE_BUY:
            stop_loss_price = entry_price_estimate - stop_loss_distance
        elif side == SIDE_SELL:
            stop_loss_price = entry_price_estimate + stop_loss_distance
        logger.info(
            f"Calculated SL based on EntryEst={entry_price_estimate:.4f}, "
            f"ATR({current_atr:.5f})*{sl_atr_multiplier}: "
            f"Price={stop_loss_price:.4f} (Distance={stop_loss_distance:.4f})"
        )

        # 4. Calculate Position Size and Required Margin
        quantity, required_margin = calculate_position_size(
            usdt_equity, risk_percentage, entry_price_estimate, stop_loss_price, leverage
        )
        if quantity is None or required_margin is None:
            return None  # Error logged in calc func

        # 5. Apply Precision and Check Minimums/Maximums
        precise_qty_str = exchange.amount_to_precision(symbol, quantity)
        precise_qty = float(precise_qty_str)
        min_qty = market.get("limits", {}).get("amount", {}).get("min")

        if min_qty is not None and precise_qty < min_qty:
            logger.warning(f"Calculated Qty {precise_qty_str} < Min Qty {min_qty}. Skipping order.")
            return None

        # Optional: Check against MAX_ORDER_USDT_AMOUNT cap
        order_value_usdt = precise_qty * entry_price_estimate
        if order_value_usdt > MAX_ORDER_USDT_AMOUNT:
            logger.warning(
                f"Risk-calculated order value ({order_value_usdt:.2f}) exceeds MAX Cap "
                f"({MAX_ORDER_USDT_AMOUNT:.2f}). Capping Qty."
            )
            capped_qty = MAX_ORDER_USDT_AMOUNT / entry_price_estimate
            precise_qty_str = exchange.amount_to_precision(symbol, capped_qty)
            precise_qty = float(precise_qty_str)
            required_margin = (precise_qty * entry_price_estimate) / leverage  # Recalc margin

            if min_qty is not None and precise_qty < min_qty:
                logger.warning(f"Capped Qty {precise_qty_str} < Min Qty {min_qty}. Skipping.")
                return None

        if precise_qty <= 0:
            logger.error("Final order Qty <= 0.")
            return None
        logger.info(
            f"Final Order Details: Qty={precise_qty_str}, "
            f"Est. Value={precise_qty * entry_price_estimate:.2f}, "
            f"Req. Margin~={required_margin:.2f}"
        )

        # 6. Check Available Margin (with buffer)
        margin_buffer = 1.05  # 5% buffer
        if usdt_free < required_margin * margin_buffer:
            logger.error(
                f"Insufficient free margin. Required ~{required_margin * margin_buffer:.2f} USDT, "
                f"Available: {usdt_free:.2f} USDT."
            )
            return None

        # 7. Place Order
        logger.info(f"*** PLACING {side.upper()} MARKET ORDER: {precise_qty_str} {symbol}... ***")
        order = exchange.create_market_order(symbol, side, precise_qty)
        # Note: Actual fill price might differ due to slippage.
        logger.success(
            f"Order Placement Success! Size: {order.get('amount', '?')}, "
            f"Avg Fill Price: {order.get('average', 'N/A')}, Order ID: {order['id']}"
        )

        # IMPORTANT TODO: Implement placing exchange-native SL/TP orders here.
        return order

    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds placing order: {e}")
    except ccxt.NetworkError as e:
        logger.error(f"Network error placing order: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error placing order: {e}")
    except Exception as e:
        logger.error(f"Unexpected error placing order: {e}")
        logger.debug(traceback.format_exc())
    return None


# --- Trading Logic (Corrected Logging Again) ---
def trade_logic(
    exchange: ccxt.Exchange,
    symbol: str,
    df: pd.DataFrame,
    st_len: int,
    st_mult: float,
    cf_st_len: int,
    cf_st_mult: float,
    atr_len: int,
    vol_ma_len: int,
    risk_pct: float,
    sl_atr_mult: float,
    tp_atr_mult: float,
    leverage: int,
) -> None:
    """Main trading logic with optimized OB fetching and detailed logging."""
    logger.info(f"========== New Check Cycle: {symbol} ==========")
    required_rows = max(2, st_len, cf_st_len, atr_len, vol_ma_len)
    if df is None or len(df) < required_rows:
        logger.warning(f"Insufficient data ({len(df) if df is not None else 0} < {required_rows}). Skipping.")
        return

    ob_data: dict[str, float | None] | None = None  # Initialize OB data

    try:
        # 1. Calculate Indicators
        logger.debug("Calculating indicators (Supertrends, ATR, Volume)...")
        df = calculate_supertrend(df, st_len, st_mult)
        df = calculate_supertrend(df, cf_st_len, cf_st_mult, prefix="confirm_")
        vol_atr_data = analyze_volume_atr(df, atr_len, vol_ma_len)
        current_atr = vol_atr_data.get("atr")

        # Check primary calculations
        required_cols = ["close", "supertrend", "trend", "st_long", "st_short", "confirm_supertrend", "confirm_trend"]
        if (
            not all(col in df.columns for col in required_cols)
            or df[required_cols].iloc[-1].isnull().any()
            or current_atr is None
        ):
            logger.warning("Indicator calculation failed/NA. Skipping logic.")
            return

        last = df.iloc[-1]
        current_price = last["close"]
        primary_long_signal = last["st_long"]
        primary_short_signal = last["st_short"]

        # 2. Analyze Order Book (Conditionally Fetched)
        if FETCH_ORDER_BOOK_PER_CYCLE or primary_long_signal or primary_short_signal:
            logger.debug(
                f"Fetching/Analyzing order book (FETCH_PER_CYCLE={FETCH_ORDER_BOOK_PER_CYCLE}, Signal={primary_long_signal or primary_short_signal})..."
            )
            ob_data = analyze_order_book(exchange, symbol, ORDER_BOOK_DEPTH)
        else:
            logger.debug("Order book fetch skipped this cycle (no signal and FETCH_PER_CYCLE=False).")

        bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None

        # 3. Get Current Position
        position = get_current_position(exchange, symbol)
        position_side = position["side"]
        position_qty = position["qty"]
        entry_price = position["entry_price"]

        # --- Pre-computation & State Logging (Corrected) ---
        primary_trend = "Up" if last["trend"] else "Down"
        confirm_trend = "Up" if last["confirm_trend"] else "Down"
        volume_ratio = vol_atr_data.get("volume_ratio")
        volume_spike = volume_ratio is not None and volume_ratio > VOLUME_SPIKE_THRESHOLD

        # Corrected Logging
        atr_str = f"{current_atr:.5f}" if current_atr is not None else "N/A"
        vol_ratio_str = f"{volume_ratio:.2f}" if volume_ratio is not None else "N/A"
        logger.info(f"Market State: Price={current_price:.4f}, ATR={atr_str}, V.Ratio={vol_ratio_str}")

        logger.info(
            f"Indicators: ST={last['supertrend']:.4f} ({primary_trend}), ConfirmST={last['confirm_supertrend']:.4f} ({confirm_trend})"
        )

        if ob_data:  # Only log if fetched
            bid_ask_ratio_str = f"{bid_ask_ratio:.3f}" if bid_ask_ratio is not None else "N/A"
            spread_val = ob_data.get("spread")
            spread_str = f"{spread_val:.4f}" if spread_val is not None else "N/A"
            logger.info(f"Order Book: Ratio={bid_ask_ratio_str}, Spread={spread_str}")
        # End Corrected Logging

        logger.info(f"Position State: Side={position_side}, Qty={position_qty}, Entry={entry_price:.4f}")

        # 4. Check Stop-Loss / Take-Profit (Loop-based - INACCURATE FOR SCALPING)
        # !!! WARNING: Loop-based check is inaccurate. Price can move past levels between checks. !!!
        if position_side != POSITION_SIDE_NONE and entry_price > 0 and current_atr > 0:
            logger.debug("Checking active position SL/TP (Loop-based check)...")
            sl_triggered, tp_triggered = False, False
            stop_price, profit_price = 0.0, 0.0
            sl_distance = current_atr * sl_atr_mult
            tp_distance = current_atr * tp_atr_mult

            if position_side == POSITION_SIDE_LONG:
                stop_price = entry_price - sl_distance
                profit_price = entry_price + tp_distance
                logger.debug(
                    f"Long Pos Check: Entry={entry_price:.4f}, Curr={current_price:.4f}, SL={stop_price:.4f}, TP={profit_price:.4f}"
                )
                if current_price <= stop_price:
                    sl_triggered = True
                if current_price >= profit_price:
                    tp_triggered = True
            elif position_side == POSITION_SIDE_SHORT:
                stop_price = entry_price + sl_distance
                profit_price = entry_price - tp_distance
                logger.debug(
                    f"Short Pos Check: Entry={entry_price:.4f}, Curr={current_price:.4f}, SL={stop_price:.4f}, TP={profit_price:.4f}"
                )
                if current_price >= stop_price:
                    sl_triggered = True
                if current_price <= profit_price:
                    tp_triggered = True

            if sl_triggered:
                logger.warning(
                    f"*** LOOP-BASED STOP-LOSS TRIGGERED *** at {current_price:.4f} (SL Level: {stop_price:.4f}). Closing {position_side}."
                )
                close_position(exchange, symbol, position)
                return  # Exit cycle
            elif tp_triggered:
                logger.success(
                    f"*** LOOP-BASED TAKE-PROFIT TRIGGERED *** at {current_price:.4f} (TP Level: {profit_price:.4f}). Closing {position_side}."
                )
                close_position(exchange, symbol, position)
                return  # Exit cycle
            else:
                logger.debug("Position SL/TP not triggered (Loop-based check).")
        elif position_side != POSITION_SIDE_NONE:
            logger.warning("Position exists but entry price/ATR invalid for SL/TP check.")

        # 5. Check for Entry Signals
        logger.debug("Checking entry signals...")
        confirm_trend_up = last["confirm_trend"]
        ob_available = ob_data is not None and bid_ask_ratio is not None

        # --- Create safe string for bid_ask_ratio for logging ---
        bid_ask_ratio_log_str = f"{bid_ask_ratio:.3f}" if ob_available else "N/A"
        # --- End safe string creation ---

        # --- Long Entry Conditions ---
        passes_long_ob = ob_available and bid_ask_ratio >= ORDER_BOOK_RATIO_THRESHOLD_LONG
        passes_volume = not REQUIRE_VOLUME_SPIKE_FOR_ENTRY or volume_spike
        enter_long = primary_long_signal and confirm_trend_up and passes_long_ob and passes_volume
        # Use the safe string here:
        logger.debug(
            f"Long Entry Check: PrimarySig={primary_long_signal}, ConfirmTrendUp={confirm_trend_up}, "
            f"OB OK={passes_long_ob} (Ratio={bid_ask_ratio_log_str}), Vol OK={passes_volume} (Spike={volume_spike})"
        )

        # --- Short Entry Conditions ---
        passes_short_ob = ob_available and bid_ask_ratio <= ORDER_BOOK_RATIO_THRESHOLD_SHORT
        enter_short = primary_short_signal and not confirm_trend_up and passes_short_ob and passes_volume
        # Use the safe string here:
        logger.debug(
            f"Short Entry Check: PrimarySig={primary_short_signal}, ConfirmTrendDown={not confirm_trend_up}, "
            f"OB OK={passes_short_ob} (Ratio={bid_ask_ratio_log_str}), Vol OK={passes_volume} (Spike={volume_spike})"
        )

        # --- Execute Actions ---
        if enter_long:
            logger.success(f"*** CONFIRMED LONG ENTRY SIGNAL *** for {symbol} at {current_price:.4f}")
            if position_side == POSITION_SIDE_SHORT:
                logger.warning("Signal LONG, but currently SHORT. Closing short first.")
                close_position(exchange, symbol, position)
                time.sleep(RETRY_DELAY_SECONDS)  # Allow time for closure
                place_risked_market_order(exchange, symbol, SIDE_BUY, risk_pct, current_atr, sl_atr_mult, leverage)
            elif position_side == POSITION_SIDE_NONE:
                logger.info("No position, entering LONG.")
                place_risked_market_order(exchange, symbol, SIDE_BUY, risk_pct, current_atr, sl_atr_mult, leverage)
            else:
                logger.info("Already LONG, maintaining position (no action).")

        elif enter_short:
            logger.success(f"*** CONFIRMED SHORT ENTRY SIGNAL *** for {symbol} at {current_price:.4f}")
            if position_side == POSITION_SIDE_LONG:
                logger.warning("Signal SHORT, but currently LONG. Closing long first.")
                close_position(exchange, symbol, position)
                time.sleep(RETRY_DELAY_SECONDS)  # Allow time for closure
                place_risked_market_order(exchange, symbol, SIDE_SELL, risk_pct, current_atr, sl_atr_mult, leverage)
            elif position_side == POSITION_SIDE_NONE:
                logger.info("No position, entering SHORT.")
                place_risked_market_order(exchange, symbol, SIDE_SELL, risk_pct, current_atr, sl_atr_mult, leverage)
            else:
                logger.info("Already SHORT, maintaining position (no action).")

        else:
            logger.info("No confirmed entry signal this cycle.")

    except Exception as e:
        logger.error(f"Critical error in trade_logic: {e}")
        logger.debug(traceback.format_exc())  # Log traceback for the error
    finally:
        logger.info(f"========== Cycle Check End: {symbol} ==========\n")


# --- Main Execution ---
def main() -> None:
    """Main function to run the scalping bot."""
    logger.info("--- Live Scalping Bot Initializing ---")
    logger.warning("--- !!! LIVE FUTURES SCALPING MODE - EXTREME RISK !!! ---")
    logger.warning("--- VERIFY ALL PARAMETERS & TEST THOROUGHLY BEFORE USE ---")

    exchange = initialize_exchange()
    if not exchange:
        return

    # Symbol setup
    symbol_input = input(f"Enter symbol (default: {DEFAULT_SYMBOL}): ").strip() or DEFAULT_SYMBOL
    try:
        market = exchange.market(symbol_input)
        symbol = market["symbol"]  # Use validated symbol format
        if not market.get("swap", False) and not market.get("future", False):
            logger.critical(f"{symbol} is not a swap/futures market.")
            return
        logger.info(f"Using symbol: {symbol}")
        if not set_leverage(exchange, symbol, DEFAULT_LEVERAGE):
            logger.critical("Leverage setup failed. Exiting.")
            return
    except (ccxt.BadSymbol, KeyError) as e:
        logger.critical(f"Invalid/unsupported symbol: {symbol_input}. Error: {e}")
        return
    except Exception as e:
        logger.critical(f"Symbol/Leverage setup failed: {e}")
        return

    # --- Configuration Summary ---
    logger.info("--- Scalping Bot Configuration ---")
    logger.info(f"Symbol: {symbol}, Interval: {DEFAULT_INTERVAL}, Leverage: {DEFAULT_LEVERAGE}x")
    logger.info(f"Risk/Trade: {RISK_PER_TRADE_PERCENTAGE:.3%}, Max Order Cap: {MAX_ORDER_USDT_AMOUNT:.2f} USDT")
    logger.info(
        f"SL: {ATR_STOP_LOSS_MULTIPLIER} * ATR({ATR_CALCULATION_PERIOD}), TP: {ATR_TAKE_PROFIT_MULTIPLIER} * ATR({ATR_CALCULATION_PERIOD})"
    )
    logger.info(f"ST: Len={DEFAULT_ST_ATR_LENGTH}, Mult={DEFAULT_ST_MULTIPLIER}")
    logger.info(f"Confirm ST: Len={CONFIRM_ST_ATR_LENGTH}, Mult={CONFIRM_ST_MULTIPLIER}")
    logger.info(
        f"Volume: MA={VOLUME_MA_PERIOD}, Spike > {VOLUME_SPIKE_THRESHOLD}x MA (Required: {REQUIRE_VOLUME_SPIKE_FOR_ENTRY})"
    )
    logger.info(
        f"Order Book: Depth={ORDER_BOOK_DEPTH}, Long Ratio > {ORDER_BOOK_RATIO_THRESHOLD_LONG}, Short Ratio < {ORDER_BOOK_RATIO_THRESHOLD_SHORT}"
    )
    logger.info(f"Fetch OB Every Cycle: {FETCH_ORDER_BOOK_PER_CYCLE} (IMPORTANT: Affects Rate Limits)")
    logger.info(f"Check Interval: {DEFAULT_SLEEP_SECONDS} seconds")
    logger.warning("Ensure configuration aligns with your risk tolerance and strategy.")
    logger.warning("Loop-based SL/TP check is inaccurate for scalping - use caution.")
    logger.info("------------------------------------")

    # Main Loop
    while True:
        start_time = time.time()
        try:
            logger.debug(f"--- Starting check cycle at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            # Fetch data
            data_limit = max(
                100,
                DEFAULT_ST_ATR_LENGTH * 2,
                CONFIRM_ST_ATR_LENGTH * 2,
                ATR_CALCULATION_PERIOD * 2,
                VOLUME_MA_PERIOD * 2,
            )
            df_market = get_market_data(exchange, symbol, DEFAULT_INTERVAL, limit=data_limit)

            if df_market is not None and not df_market.empty:
                trade_logic(
                    exchange=exchange,
                    symbol=symbol,
                    df=df_market.copy(),  # Pass copy
                    st_len=DEFAULT_ST_ATR_LENGTH,
                    st_mult=DEFAULT_ST_MULTIPLIER,
                    cf_st_len=CONFIRM_ST_ATR_LENGTH,
                    cf_st_mult=CONFIRM_ST_MULTIPLIER,
                    atr_len=ATR_CALCULATION_PERIOD,
                    vol_ma_len=VOLUME_MA_PERIOD,
                    risk_pct=RISK_PER_TRADE_PERCENTAGE,
                    sl_atr_mult=ATR_STOP_LOSS_MULTIPLIER,
                    tp_atr_mult=ATR_TAKE_PROFIT_MULTIPLIER,
                    leverage=DEFAULT_LEVERAGE,
                )
            else:
                logger.warning("No market data received, skipping this logic cycle.")

        except KeyboardInterrupt:
            logger.warning("Shutdown requested via KeyboardInterrupt.")
            # TODO: Implement graceful shutdown (close positions)
            break  # Exit loop
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"RATE LIMIT EXCEEDED: {e}. Sleeping longer...")
            time.sleep(DEFAULT_SLEEP_SECONDS * 5)  # Significantly longer sleep
        except ccxt.NetworkError as e:
            logger.warning(f"Network error in main loop: {e}. Retrying after delay...")
            time.sleep(DEFAULT_SLEEP_SECONDS)
        except ccxt.ExchangeNotAvailable as e:
            logger.error(f"Exchange not available ({e}). Sleeping longer...")
            time.sleep(DEFAULT_SLEEP_SECONDS * 5)  # Long sleep
        except ccxt.ExchangeError as e:
            logger.error(f"Unhandled Exchange error in main loop: {e}.")
            logger.debug(traceback.format_exc())
            time.sleep(DEFAULT_SLEEP_SECONDS)
        except Exception as e:
            logger.exception(f"!!! Unexpected critical error in main loop: {e} !!!")
            logger.info(f"Attempting to continue after {DEFAULT_SLEEP_SECONDS}s sleep...")
            time.sleep(DEFAULT_SLEEP_SECONDS)

        # Maintain check interval approximately
        elapsed_time = time.time() - start_time
        sleep_duration = max(0, DEFAULT_SLEEP_SECONDS - elapsed_time)
        logger.debug(f"Cycle execution time: {elapsed_time:.2f}s. Sleeping for {sleep_duration:.2f}s.")
        if sleep_duration > 0:
            time.sleep(sleep_duration)

    logger.info("--- Scalping Bot Shutdown Complete ---")


if __name__ == "__main__":
    main()
