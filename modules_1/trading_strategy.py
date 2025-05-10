# File: trading_strategy.py
import logging
import asyncio
import time  # Keep for time.monotonic() used in polling/timeouts
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation
from typing import Any, Dict, Optional, Union  # Added List, Callable
import sys
import traceback
import pandas as pd

# Import the enhanced API client class
try:
    from exchange_api import BybitAPI  # Import the class
except ImportError as e:
    _NEON_RED = "\033[1;91m"
    _RESET_ALL_STYLE = "\033[0m"
    print(
        f"{_NEON_RED}CRITICAL ERROR: Failed import BybitAPI from exchange_api.py: {e}{_RESET_ALL_STYLE}",
        file=sys.stderr,
    )
    if "traceback" not in sys.modules:
        import traceback
    traceback.print_exc(file=sys.stderr)
    raise

# Import other necessary modules
try:
    import utils  # Keep for constants/colors etc.
    from analysis import TradingAnalyzer
    from risk_manager import calculate_position_size

    # Import constants directly from utils
    from utils import (
        CCXT_INTERVAL_MAP,
        DEFAULT_INDICATOR_PERIODS,
        NEON_GREEN,
        NEON_PURPLE,
        NEON_RED,
        NEON_YELLOW,
        RESET_ALL_STYLE,
        NEON_BLUE,
        NEON_CYAN,
    )
except ImportError as e:
    _NEON_RED = "\033[1;91m"
    _RESET_ALL_STYLE = "\033[0m"
    print(
        f"{_NEON_RED}CRITICAL ERROR: Failed import modules in trading_strategy.py: {e}{_RESET_ALL_STYLE}",
        file=sys.stderr,
    )
    if "traceback" not in sys.modules:
        import traceback
    traceback.print_exc(file=sys.stderr)
    raise

# Define color constants within this module
NEON_GREEN = utils.NEON_GREEN
NEON_BLUE = utils.NEON_BLUE
NEON_PURPLE = utils.NEON_PURPLE
NEON_YELLOW = utils.NEON_YELLOW
NEON_RED = utils.NEON_RED
NEON_CYAN = utils.NEON_CYAN
RESET = utils.RESET_ALL_STYLE


# --- Formatting Helpers ---
def _format_signal(signal_text: Any) -> str:
    signal_str = str(signal_text).upper()
    if signal_str == "BUY":
        return f"{NEON_GREEN}{signal_str}{RESET}"
    if signal_str == "SELL":
        return f"{NEON_RED}{signal_str}{RESET}"
    if signal_str == "HOLD":
        return f"{NEON_YELLOW}{signal_str}{RESET}"
    return f"{signal_str}{RESET}"


def _format_side(side_text: Optional[str]) -> str:
    if side_text is None:
        return f"{NEON_YELLOW}UNKNOWN{RESET}"
    side_upper = side_text.upper()
    if side_upper == "LONG":
        return f"{NEON_GREEN}{side_upper}{RESET}"
    if side_upper == "SHORT":
        return f"{NEON_RED}{side_upper}{RESET}"
    return f"{NEON_YELLOW}{side_upper}{RESET}"


def _format_price_or_na(
    price_val: Optional[Union[Decimal, str]], precision_places: int, label: str = ""
) -> str:
    color = NEON_CYAN
    na_color = NEON_YELLOW
    if price_val is None or str(price_val).strip() == "":
        return f"{na_color}N/A{RESET}"
    try:
        d_val = Decimal(str(price_val))
        if d_val == Decimal(0) and label != "ATR":  # Allow ATR to be 0.00000
            # For prices, 0.0 is often a placeholder or error, but for ATR small values are fine
            if precision_places > 0:  # Display 0.0 if precision is expected
                return f"{na_color}{d_val:.{precision_places}f}{RESET}"
            return f"{na_color}0.0{RESET}"  # Fallback for 0 precision
        # Display with specified precision
        format_str = f":.{precision_places}f"
        return f"{color}{d_val:{format_str}}{RESET}"
    except (InvalidOperation, TypeError, ValueError):  # Catch more formatting errors
        return f"{na_color}{price_val} (invalid){RESET}"


# --- Core Trading Logic ---


async def _set_position_protection_logic(
    api_client: BybitAPI,  # Use API client instance
    symbol: str,
    market_info: Dict[str, Any],
    confirmed_pos: Dict[str, Any],
    config: Dict[str, Any],
    lg: logging.Logger,
    analysis_results: Dict[str, Any],
    price_precision: int,
    signal: str,
) -> bool:
    """Internal helper to calculate and set SL/TP/TSL after position confirmation."""
    lg.info(f"Proceeding with protection setup for confirmed {symbol} position...")
    try:
        entry_px_actual = confirmed_pos.get("entryPriceDecimal")
        analyzer: TradingAnalyzer = analysis_results[
            "analyzer"
        ]  # Get analyzer instance
        if not (
            entry_px_actual
            and isinstance(entry_px_actual, Decimal)
            and entry_px_actual > 0
        ):
            lg.warning(
                f"{NEON_YELLOW}Actual entry price missing/invalid. Using analysis close price estimate.{RESET}"
            )
            entry_px_actual = analyzer.indicator_values.get("Close")  # Fallback
            if not (
                entry_px_actual
                and isinstance(entry_px_actual, Decimal)
                and entry_px_actual > 0
            ):
                lg.error(
                    f"{NEON_RED}Cannot determine entry price for protection {symbol}. Aborting.{RESET}"
                )
                return False

        # Recalculate TP/SL using actual/estimated entry price
        _, tp_f, sl_f = analyzer.calculate_entry_tp_sl(
            entry_px_actual, signal
        )  # signal here is entry signal ('BUY' or 'SELL')
        lg.info(
            f"Protection targets based on Entry {_format_price_or_na(entry_px_actual, price_precision)}: SL={_format_price_or_na(sl_f, price_precision)}, TP={_format_price_or_na(tp_f, price_precision)}"
        )

        tsl_enabled_config = analysis_results.get("tsl_enabled_config", False)
        prot_ok = False
        if tsl_enabled_config:
            lg.info(f"Setting Trailing Stop Loss for {symbol}...")
            # Pass TP target (Decimal, 0 Decimal to remove, or None to leave unchanged)
            tp_param = tp_f if isinstance(tp_f, Decimal) else None
            prot_ok = await api_client.set_trailing_stop_loss(
                symbol, confirmed_pos, config, tp_param
            )
        # Check if valid fixed SL or TP exists (allow 0 to remove)
        elif (sl_f is not None and isinstance(sl_f, Decimal) and sl_f >= 0) or (
            tp_f is not None and isinstance(tp_f, Decimal) and tp_f >= 0
        ):
            lg.info(f"Setting Fixed SL/TP for {symbol}...")
            # Use the internal method directly, passing Decimals (0 means remove)
            prot_ok = await api_client._set_position_protection(
                symbol, market_info, confirmed_pos, sl_f, tp_f
            )
        else:
            lg.info(
                f"No valid protection targets and TSL disabled. No action taken for {symbol}."
            )
            prot_ok = True  # Success as no action needed

        if prot_ok:
            lg.info(
                f"{NEON_GREEN}Protection setup successful/not required for {symbol}.{RESET}"
            )
        else:
            lg.error(
                f"{NEON_RED}Protection setup FAILED for {symbol}. Manual check advised.{RESET}"
            )
        return prot_ok

    except Exception as protect_err:
        lg.error(
            f"{NEON_RED}Error during protection setup {symbol}: {protect_err}{RESET}",
            exc_info=True,
        )
        return False


async def _execute_close_position(
    api_client: BybitAPI,
    symbol: str,
    market_info: Dict[str, Any],
    open_position: Dict[str, Any],
    config: Dict[str, Any],  # Added config for close_confirm_delay_seconds
    lg: logging.Logger,
    reason: str = "exit signal",
) -> bool:
    """Executes a market order to close an existing position using the API client."""
    pos_side = open_position.get("side")
    pos_size = open_position.get("contractsDecimal")
    amount_precision = market_info.get("amountPrecision", 8)

    if not (
        pos_side in ["long", "short"] and isinstance(pos_size, Decimal) and pos_size > 0
    ):
        lg.warning(
            f"Close {symbol} ({reason}) skipped: Invalid pos data. Side='{pos_side}', Size='{pos_size}'."
        )
        return False

    try:
        close_signal = "SELL" if pos_side == "long" else "BUY"
        lg.warning(
            f"{NEON_YELLOW}==> Closing {_format_side(pos_side)} position {symbol} ({reason}) | Size: {pos_size:.{amount_precision}f} <=="
        )

        close_order = await api_client.place_trade(  # Use API client method
            symbol=symbol,
            trade_signal=close_signal,
            position_size=pos_size,
            order_type="market",
            reduce_only=True,
        )

        if close_order and close_order.get("id"):
            lg.info(
                f"{NEON_GREEN}CLOSE order placed {symbol}. ID: {close_order['id']}, Status: {close_order.get('status', 'N/A')}{RESET}"
            )
            # Optional: Confirm position closure
            await asyncio.sleep(
                config.get("close_confirm_delay_seconds", 2.0)
            )  # Configurable delay
            final_pos = await api_client.get_open_position(symbol)
            if final_pos is None:
                lg.info(f"{NEON_GREEN}Position closure {symbol} confirmed.{RESET}")
            else:
                lg.warning(
                    f"{NEON_YELLOW}Position closure {symbol} NOT confirmed. Pos size: {final_pos.get('contractsDecimal')}{RESET}"
                )
            return True
        else:
            lg.error(
                f"{NEON_RED}Failed to place CLOSE order {symbol}. API call failed/returned None.{RESET}"
            )
            return False
    except Exception as e:
        lg.error(
            f"{NEON_RED}Error closing position {symbol}: {e}{RESET}", exc_info=True
        )
        return False


async def _fetch_and_prepare_market_data(
    api_client: BybitAPI, symbol: str, config: Dict[str, Any], lg: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Fetches market info, klines, price, orderbook using the API client."""
    market_info = await api_client.get_market_info(symbol)
    if not market_info:
        lg.error(f"Failed get market info {symbol}.")
        return None

    # Get derived values from processed market_info
    price_precision = market_info.get("pricePrecisionPlaces", 4)
    min_tick_size = market_info.get("minTickSizeDecimal", Decimal("1e-4"))
    amount_precision = market_info.get("amountPrecision", 8)
    min_qty = Decimal(
        str(market_info.get("limits", {}).get("amount", {}).get("min", "0"))
    )

    interval_str = str(config.get("interval", "5"))
    ccxt_interval = CCXT_INTERVAL_MAP.get(interval_str)
    if not ccxt_interval:
        lg.error(f"Invalid interval '{interval_str}' {symbol}.")
        return None

    kline_limit = max(1, config.get("kline_limit", 500))
    klines_df = await api_client.fetch_klines(
        symbol, ccxt_interval, limit=kline_limit
    )  # Use API client
    min_kline_len = max(1, config.get("min_kline_length", 100))
    if klines_df is None or klines_df.empty or len(klines_df) < min_kline_len:
        lg.error(
            f"Insufficient klines {symbol} ({len(klines_df) if klines_df is not None else 0}/{min_kline_len})."
        )
        return None

    current_price = await api_client.fetch_current_price(symbol)  # Use API client
    if not (current_price and current_price > 0):
        last_close = klines_df["close"].iloc[-1]
        if isinstance(last_close, Decimal) and last_close > 0:
            current_price = last_close
            lg.warning(f"Using last kline close price as current price for {symbol}.")
        else:
            lg.error(f"Cannot get valid current price {symbol}.")
            return None
    lg.debug(
        f"Current price {symbol}: {_format_price_or_na(current_price, price_precision)}"
    )

    orderbook_data = None
    orderbook_enabled = config.get("indicators", {}).get("orderbook", False)
    active_weights = config.get("weight_sets", {}).get(
        config.get("active_weight_set", "default"), {}
    )
    try:
        weight = Decimal(str(active_weights.get("orderbook", 0)))
    except (InvalidOperation, TypeError):
        weight = Decimal(0)
    if orderbook_enabled and weight != 0:
        ob_limit = max(1, config.get("orderbook_limit", 25))
        orderbook_data = await api_client.fetch_orderbook(
            symbol, limit=ob_limit
        )  # Use API client
        if not orderbook_data:
            lg.warning(f"Failed fetch orderbook {symbol}.")
    elif orderbook_enabled:
        lg.debug(f"OB indicator enabled but weight zero {symbol}.")
    else:
        lg.debug(f"OB indicator disabled {symbol}.")

    return {
        "market_info": market_info,
        "klines_df": klines_df,
        "current_price_decimal": current_price,
        "orderbook_data": orderbook_data,
        "price_precision": price_precision,
        "min_tick_size": min_tick_size,
        "amount_precision": amount_precision,
        "min_qty": min_qty,
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
    """Performs trading analysis using TradingAnalyzer."""
    try:
        analyzer = TradingAnalyzer(klines_df.copy(), lg, config, market_info)
    except Exception as e:
        lg.error(
            f"Failed init Analyzer {market_info.get('symbol', '?')}: {e}", exc_info=True
        )
        return None
    if (
        not analyzer.indicator_values
    ):  # Check if indicators were successfully calculated
        lg.error(f"Indicator calc failed {market_info.get('symbol', '?')}.")
        return None

    signal = analyzer.generate_trading_signal(current_price_decimal, orderbook_data)
    _, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price_decimal, signal)
    current_atr = analyzer.indicator_values.get("ATR")

    lg.info(
        f"--- {NEON_PURPLE}Analysis Summary ({market_info.get('symbol')}){RESET} ---"
    )
    lg.info(f"  Price: {_format_price_or_na(current_price_decimal, price_precision)}")
    atr_period = config.get(
        "atr_period", DEFAULT_INDICATOR_PERIODS.get("atr_period", 14)
    )
    lg.info(
        f"  ATR ({atr_period}): {_format_price_or_na(current_atr, price_precision + 2, 'ATR')}"
    )  # ATR often needs more precision
    lg.info(
        f"  Init SL (sizing): {_format_price_or_na(sl_calc, price_precision, 'SL Calc')}"
    )
    lg.info(
        f"  Init TP (target): {_format_price_or_na(tp_calc, price_precision, 'TP Calc')}"
    )
    tsl_en = config.get("enable_trailing_stop", False)
    be_en = config.get("enable_break_even", False)
    t_exit_val = config.get("time_based_exit_minutes")
    t_exit_str = (
        f"{t_exit_val:.1f}m"
        if isinstance(t_exit_val, (int, float)) and t_exit_val > 0
        else "Disabled"
    )
    lg.info(f"  Config: TSL={tsl_en}, BE={be_en}, TimeExit={t_exit_str}")
    lg.info(f"  Signal: {_format_signal(signal)}")
    lg.info("-----------------------------")

    return {
        "signal": signal,
        "tp_calc": tp_calc,
        "sl_calc": sl_calc,
        "analyzer": analyzer,
        "tsl_enabled_config": tsl_en,
        "be_enabled_config": be_en,
        "time_exit_minutes_config": t_exit_val,
        "current_atr": current_atr,
    }


async def _handle_no_open_position(
    api_client: BybitAPI,
    symbol: str,
    config: Dict[str, Any],
    lg: logging.Logger,
    market_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
):
    """Handles entry logic: validation, calculation, order placement, confirmation, protection."""
    signal = analysis_results["signal"]
    if signal not in ["BUY", "SELL"]:
        lg.info(f"Signal {_format_signal(signal)}, no pos {symbol}. No entry.")
        return
    lg.info(
        f"{NEON_PURPLE}*** {_format_signal(signal)} Signal & No Position: Evaluating Entry {symbol} ***{RESET}"
    )
    loop = asyncio.get_event_loop()

    # --- Pre-checks ---
    if api_client.circuit_breaker_tripped:
        lg.error(f"{NEON_RED}CB tripped. Skip entry {symbol}.{RESET}")
        return
    # Optional health check: if not (await api_client.health_check())['checks'].get('connection'): return

    # --- Fetch Balance & Calculate Size ---
    quote_currency = config.get("quote_currency", "USDT")
    balance = await api_client.fetch_balance(quote_currency)
    if not (balance and isinstance(balance, Decimal) and balance > 0):
        lg.error(
            f"Entry Abort {symbol}: Invalid balance or zero balance for {quote_currency}."
        )
        return
    risk_pct = Decimal(
        str(config.get("risk_per_trade", "0.01"))
    )  # Ensure config value is string for Decimal
    if not (0 < risk_pct <= 1):  # risk_pct is a percentage of balance, e.g. 0.01 for 1%
        lg.error(
            f"Entry Abort {symbol}: Invalid risk_per_trade '{risk_pct}'. Must be > 0 and <= 1."
        )
        return
    sl_calc = analysis_results["sl_calc"]  # Initial SL for sizing
    if not (sl_calc and isinstance(sl_calc, Decimal) and sl_calc > 0):
        lg.error(f"Entry Abort {symbol}: Invalid SL price for sizing: {sl_calc}.")
        return

    market_info = market_data["market_info"]
    current_price = market_data["current_price_decimal"]
    price_precision = market_data["price_precision"]
    amount_precision = market_data["amount_precision"]
    min_tick_size = market_data["min_tick_size"]
    min_qty = market_data["min_qty"]

    # --- Set Leverage ---
    if market_info.get(
        "is_contract"
    ):  # Check if it's a contract market (futures/swaps)
        lev = max(1, int(config.get("leverage", 1)))
        if not await api_client.set_leverage(symbol, lev):
            lg.error(f"Entry Abort {symbol}: Failed set leverage to {lev}x.")
            return

    # --- Calculate Position Size ---
    pos_size_dec = calculate_position_size(
        balance, risk_pct, sl_calc, current_price, market_info, api_client.exchange, lg
    )
    if not (pos_size_dec and pos_size_dec > 0):
        lg.error(
            f"Entry Abort {symbol}: Pos size calc invalid or zero ({pos_size_dec})."
        )
        return

    # --- Pre-placement Validations ---
    entry_type = config.get("entry_order_type", "market").lower()
    limit_price_for_api: Optional[Decimal] = None
    try:
        if min_qty > 0 and pos_size_dec < min_qty:
            raise ValueError(f"Calculated size {pos_size_dec} < MinQty {min_qty}.")

        if entry_type == "limit":
            use_ob_price = config.get("adjust_price_to_orderbook", False)
            if (
                use_ob_price
                and market_data["orderbook_data"]
                and market_data["orderbook_data"].get("bids")
                and market_data["orderbook_data"].get("asks")
            ):
                ob = market_data["orderbook_data"]
                top_ask = Decimal(str(ob["asks"][0][0]))
                top_bid = Decimal(str(ob["bids"][0][0]))
                limit_px_candidate = top_bid if signal == "BUY" else top_ask
            else:
                offset_key = f"limit_order_offset_{signal.lower()}"
                offset_pct_str = str(config.get(offset_key, "0.0005"))  # e.g. 0.05%
                offset_pct = Decimal(offset_pct_str)
                limit_px_candidate = current_price * (
                    Decimal("1") - offset_pct
                    if signal == "BUY"
                    else Decimal("1") + offset_pct
                )

            if min_tick_size > 0:
                limit_px = (limit_px_candidate / min_tick_size).quantize(
                    Decimal("1"), ROUND_DOWN if signal == "BUY" else ROUND_UP
                ) * min_tick_size
            else:  # Should not happen for valid market_info
                limit_px = limit_px_candidate.quantize(
                    Decimal("1e-8")
                )  # Generic fallback

            if not (limit_px and limit_px > 0):
                raise ValueError(f"Limit price calculation invalid: {limit_px}.")
            limit_price_for_api = limit_px

            price_limits = market_info.get("limits", {}).get("price", {})
            min_p_str = str(price_limits.get("min", "0"))
            max_p_str = str(price_limits.get("max", "inf"))
            min_p = Decimal(min_p_str) if min_p_str else Decimal("0")
            max_p = (
                Decimal(max_p_str)
                if max_p_str and max_p_str != "inf"
                else Decimal("Infinity")
            )

            if not (min_p <= limit_px <= max_p):
                raise ValueError(
                    f"Limit price {limit_px} outside exchange limits [{min_p},{max_p}]"
                )

            cost = pos_size_dec * limit_px
            min_cost_str = str(
                market_info.get("limits", {}).get("cost", {}).get("min", "0")
            )
            min_cost = Decimal(min_cost_str) if min_cost_str else Decimal("0")
            if min_cost > 0 and cost < min_cost:
                raise ValueError(
                    f"Order cost {cost:.{price_precision + amount_precision}f} < min cost {min_cost}."
                )

    except ValueError as val_err:
        lg.error(f"Entry Aborted {symbol}: Validation Error - {val_err}")
        return
    except Exception as e:
        lg.error(
            f"Entry Aborted {symbol}: Pre-placement validation error - {e}",
            exc_info=True,
        )
        return

    # --- Determine Final Order Parameters ---
    conditional_trigger_price: Optional[Decimal] = None
    final_params: Dict[str, Any] = {}  # For exchange-specific params
    if entry_type == "conditional":
        # Example: trigger_offset_pct = Decimal(str(config.get("conditional_trigger_offset_pct", "0.001")))
        # if signal == "BUY": # Stop Buy (buy when price rises to trigger)
        #     conditional_trigger_price = current_price * (Decimal("1") + trigger_offset_pct)
        # else: # Stop Sell (sell when price falls to trigger)
        #     conditional_trigger_price = current_price * (Decimal("1") - trigger_offset_pct)
        # if min_tick_size > 0:
        #    conditional_trigger_price = (conditional_trigger_price / min_tick_size).quantize(Decimal("1"), ROUND_UP if signal == "BUY" else ROUND_DOWN) * min_tick_size
        # if not (conditional_trigger_price and conditional_trigger_price > 0):
        #    lg.error(f"Conditional trigger price calc invalid for {symbol}. Aborting entry.")
        #    return
        # limit_price_for_api might be None for conditional market, or set for conditional limit
        lg.warning(
            f"Conditional order logic for {symbol} is a placeholder. Ensure trigger price is correctly set."
        )

    time_in_force = config.get("time_in_force")
    post_only = config.get("post_only", False) if entry_type == "limit" else False
    client_order_id = (
        f"{symbol.replace('/', '')[:10]}_{signal[:1]}_{int(time.monotonic() * 1000)}"[
            -60:
        ]
    )
    trigger_by = config.get(
        "trigger_by"
    )  # e.g., 'LastPrice', 'MarkPrice', 'IndexPrice' for conditional/stop

    # --- Cancel Conflicting Open Orders ---
    try:
        open_orders = await api_client.fetch_open_orders(symbol)
        for order in open_orders:
            order_side = order.get("side", "").lower()
            # Cancel open orders that oppose the new signal (e.g. if new signal is BUY, cancel existing SELL limit orders)
            if (signal == "BUY" and order_side == "sell") or (
                signal == "SELL" and order_side == "buy"
            ):
                # Be cautious not to cancel legitimate TP orders of an existing position if this logic runs unexpectedly
                # This function assumes no existing position, so any opposing order is likely a remnant or manual.
                lg.warning(
                    f"{NEON_YELLOW}Canceling conflicting open {order_side} order {order['id']} for {symbol} before placing new {signal} order.{RESET}"
                )
                await api_client.cancel_order(order["id"], symbol)
    except Exception as cancel_err:
        lg.error(
            f"Failed to cancel conflicting orders for {symbol}: {cancel_err}. Proceeding with caution."
        )

    # --- Place Trade ---
    trade_order = await api_client.place_trade(
        symbol=symbol,
        trade_signal=signal,
        position_size=pos_size_dec,
        order_type=entry_type,
        limit_price=limit_price_for_api,
        reduce_only=False,
        time_in_force=time_in_force,
        post_only=post_only,
        trigger_price=conditional_trigger_price,
        trigger_by=trigger_by,
        client_order_id=client_order_id,
        params=final_params,
    )

    if not (trade_order and trade_order.get("id")):
        lg.error(
            f"{NEON_RED}=== TRADE ENTRY FAILED {symbol} {signal}. Order placement failed. ==={RESET}"
        )
        return

    order_id = trade_order["id"]
    order_status = trade_order.get("status", "unknown").lower()
    # --- Post-Placement: Confirmation & Protection ---
    confirmed_pos: Optional[Dict[str, Any]] = None
    protection_set = False

    if entry_type == "market" or (
        entry_type in ["limit", "conditional"] and order_status == "closed"
    ):  # Order filled immediately
        confirm_delay = config.get("order_confirmation_delay_seconds", 0.75)
        max_retries = config.get("position_confirm_retries", 3)
        timeout_seconds = config.get(
            "protection_setup_timeout_seconds", 30
        )  # Overall timeout for this block
        lg.info(
            f"Waiting {confirm_delay}s (up to {max_retries} tries / {timeout_seconds}s total) for position confirmation {symbol}..."
        )
        start_time = loop.time()
        for attempt in range(max_retries):
            if loop.time() - start_time > timeout_seconds:
                lg.error(
                    f"{NEON_RED}Position confirmation timeout for {symbol} after order {order_id}.{RESET}"
                )
                break
            await asyncio.sleep(confirm_delay * (attempt + 1))  # Increasing delay
            order_check = await api_client.fetch_order(order_id, symbol)
            if order_check and order_check.get("status") == "closed":
                confirmed_pos = await api_client.get_open_position(symbol)
                if confirmed_pos:
                    filled_size_str = str(order_check.get("filled", "0"))
                    filled_size = (
                        Decimal(filled_size_str) if filled_size_str else Decimal("0")
                    )
                    expected_size = pos_size_dec
                    # Check for significant partial fill (e.g., more than 1% difference)
                    if filled_size > 0 and abs(filled_size - expected_size) > (
                        expected_size * Decimal("0.01")
                    ):
                        lg.warning(
                            f"{NEON_YELLOW}Partial Fill! Expected {expected_size}, Filled {filled_size} for order {order_id}. Using filled size for protection.{RESET}"
                        )
                        # Update position object if it's fetched and used by protection logic
                        confirmed_pos["contractsDecimal"] = filled_size
                        confirmed_pos["size"] = filled_size  # common alternative key
                    break  # Position confirmed as open, or order confirmed as filled
            elif order_check is None:
                lg.warning(
                    f"Order {order_id} not found during confirmation poll for {symbol}."
                )
                # Potentially break if order should exist
                # break
            else:  # Order not closed yet or status unknown
                lg.warning(
                    f"Position confirm attempt {attempt + 1} for {symbol}: Order {order_id} status: '{order_check.get('status') if order_check else 'Not Found'}'. Retrying..."
                )
        else:  # Loop finished without break (max_retries reached)
            lg.error(
                f"{NEON_RED}FAILED TO CONFIRM open position for {symbol} after order {order_id}! Manual check required!{RESET}"
            )
            # Consider canceling the order if it's still open and unfilled, depending on strategy
            return

        if confirmed_pos:
            lg.info(
                f"{NEON_GREEN}Position Confirmed for {symbol}! Setting protection...{RESET}"
            )
            protection_set = await _set_position_protection_logic(
                api_client,
                symbol,
                market_info,
                confirmed_pos,
                config,
                lg,
                analysis_results,
                price_precision,
                signal,
            )
        elif order_status == "closed" and not confirmed_pos:
            lg.error(
                f"{NEON_RED}Order {order_id} for {symbol} is 'closed' but no open position found. Possible immediate fill & exit or error.{RESET}"
            )
            # This state might occur if the order fills and closes immediately due to market conditions or other orders.
            # Or it could be an issue with get_open_position consistency.

    elif entry_type in ["limit", "conditional"] and order_status in [
        "open",
        "partially_filled",
    ]:
        lg.info(
            f"{NEON_YELLOW}{entry_type.capitalize()} order {order_id} is '{order_status}' for {symbol} @ price {_format_price_or_na(limit_price_for_api or conditional_trigger_price, price_precision)}. Monitoring...{RESET}"
        )
        max_wait_seconds = config.get("limit_order_timeout_seconds", 300)
        poll_interval_seconds = config.get("limit_order_poll_interval_seconds", 5)
        # Stale timeout: if order hasn't been touched (created/updated) in X seconds, consider it stale
        stale_timeout_seconds = config.get("limit_order_stale_timeout_seconds", 600)
        order_timestamp_ms = trade_order.get("timestamp", int(loop.time() * 1000))  # ms

        start_time = loop.time()
        while loop.time() - start_time < max_wait_seconds:
            await asyncio.sleep(poll_interval_seconds)

            current_order_state = await api_client.fetch_order(order_id, symbol)
            if not current_order_state:
                lg.warning(
                    f"{entry_type.capitalize()} order {order_id} for {symbol} disappeared during monitoring."
                )
                break  # Order no longer exists

            current_status = current_order_state.get("status", "").lower()
            last_update_ts_ms = current_order_state.get(
                "lastUpdateTimestamp"
            ) or current_order_state.get("timestamp", order_timestamp_ms)

            # Stale Check (optional, based on last update time if available)
            if (
                stale_timeout_seconds > 0
                and last_update_ts_ms
                and (loop.time() * 1000 - last_update_ts_ms) / 1000
                > stale_timeout_seconds
            ):
                lg.warning(
                    f"{NEON_YELLOW}{entry_type.capitalize()} order {order_id} for {symbol} considered stale (no update for > {stale_timeout_seconds}s). Canceling.{RESET}"
                )
                await api_client.cancel_order(order_id, symbol)
                break

            if current_status == "closed":
                lg.info(
                    f"{NEON_GREEN}{entry_type.capitalize()} order {order_id} for {symbol} filled.{RESET}"
                )
                filled_size_str = str(current_order_state.get("filled", "0"))
                filled_size = (
                    Decimal(filled_size_str) if filled_size_str else Decimal("0")
                )

                if filled_size <= Decimal(0):
                    lg.error(
                        f"{NEON_RED}Order {order_id} for {symbol} 'closed' but filled size is {filled_size}. Aborting protection.{RESET}"
                    )
                    break

                # Handle partial fills if necessary
                if filled_size < pos_size_dec * Decimal(
                    "0.99"
                ):  # If filled less than 99% of intended
                    lg.warning(
                        f"{NEON_YELLOW}Partial Fill for {entry_type} order {order_id} on {symbol}! Expected {pos_size_dec}, Filled {filled_size}.{RESET}"
                    )
                    # Strategy decision: continue with partial, or cancel and reassess? For now, continue.

                confirmed_pos = await api_client.get_open_position(symbol)
                if confirmed_pos:
                    # Ensure the position size reflects the actual filled amount for protection logic
                    confirmed_pos["contractsDecimal"] = filled_size
                    confirmed_pos["size"] = filled_size
                    lg.info(
                        f"{NEON_GREEN}Position Confirmed for {symbol} after {entry_type} fill! Setting protection...{RESET}"
                    )
                    protection_set = await _set_position_protection_logic(
                        api_client,
                        symbol,
                        market_info,
                        confirmed_pos,
                        config,
                        lg,
                        analysis_results,
                        price_precision,
                        signal,
                    )
                else:
                    lg.error(
                        f"{NEON_RED}{entry_type.capitalize()} order {order_id} filled for {symbol}, but FAILED to confirm open position! Manual check!{RESET}"
                    )
                break  # Exit polling loop
            elif current_status == "canceled":
                lg.info(
                    f"{NEON_RED}{entry_type.capitalize()} order {order_id} for {symbol} was canceled (manually or by stale check).{RESET}"
                )
                break
            elif current_status not in [
                "open",
                "partially_filled",
            ]:  # new, accepted, etc. are usually transient
                lg.error(
                    f"{NEON_RED}{entry_type.capitalize()} order {order_id} for {symbol} has unexpected status '{current_status}'. Stopping monitor.{RESET}"
                )
                break
            # else: still open/partially_filled, continue polling
        else:  # Polling loop timed out
            lg.warning(
                f"{NEON_YELLOW}{entry_type.capitalize()} order {order_id} for {symbol} not filled within {max_wait_seconds}s timeout. Canceling.{RESET}"
            )
            await api_client.cancel_order(order_id, symbol)
    else:  # e.g. status rejected, expired
        lg.error(
            f"{NEON_RED}Order {order_id} for {symbol} has initial status '{order_status}' which is not 'closed', 'open', or 'partially_filled'. Manual check required!{RESET}"
        )

    if protection_set:
        lg.info(
            f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION COMPLETE ({symbol} {signal}) ==={RESET}"
        )
    elif confirmed_pos:  # Position exists but protection failed or was skipped
        lg.warning(
            f"{NEON_YELLOW}=== TRADE ENTRY COMPLETE ({symbol} {signal}) BUT PROTECTION FAILED/SKIPPED ==={RESET}"
        )
    # If no confirmed_pos, the earlier logs for failure/cancellation cover it.


async def _manage_existing_open_position(
    api_client: BybitAPI,
    symbol: str,
    config: Dict[str, Any],
    lg: logging.Logger,
    market_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
    open_position: Dict[str, Any],
):
    """Manages existing position: checks exits, BE, TSL, cancels conflicting orders."""
    loop = asyncio.get_event_loop()
    pos_side = open_position.get("side")  # 'long' or 'short'
    pos_size = open_position.get("contractsDecimal")
    entry_px = open_position.get("entryPriceDecimal")
    pos_ts_ms = open_position.get(
        "timestamp_ms"
    )  # Timestamp of position opening/last update

    market_info = market_data["market_info"]
    current_price = market_data["current_price_decimal"]
    price_prec = market_data["price_precision"]
    amt_prec = market_data["amount_precision"]
    min_tick = market_data["min_tick_size"]
    current_atr = analysis_results.get("current_atr")  # From analysis_results

    if not (
        pos_side in ["long", "short"]
        and isinstance(pos_size, Decimal)
        and pos_size > 0
        and isinstance(entry_px, Decimal)
        and entry_px > 0
    ):
        lg.error(
            f"Cannot manage {symbol}: Invalid/incomplete position details: Side={pos_side}, Size={pos_size}, EntryPx={entry_px}."
        )
        return

    # --- Log Current Position State ---
    lg.info(f"{NEON_BLUE}--- Managing Position ({symbol}) ---{RESET}")
    lg.info(
        f"  Side: {_format_side(pos_side)}, Size: {_format_price_or_na(pos_size, amt_prec)}, Entry: {_format_price_or_na(entry_px, price_prec)}"
    )
    cur_sl = open_position.get("stopLossPriceDecimal")
    cur_tp = open_position.get("takeProfitPriceDecimal")
    lg.info(
        f"  SL: {_format_price_or_na(cur_sl, price_prec)}, TP: {_format_price_or_na(cur_tp, price_prec)}"
    )

    tsl_dist_val = open_position.get(
        "trailingStopDistanceDecimal"
    )  # Bybit might provide this if TSL is active on exchange
    tsl_act_px_val = open_position.get("trailingStopActivationPriceDecimal")
    is_tsl_active_on_exchange = bool(tsl_dist_val and tsl_dist_val > 0)
    lg.info(
        f"  Exch TSL: Active={is_tsl_active_on_exchange}, Value={_format_price_or_na(tsl_dist_val, price_prec)}, ActPx={_format_price_or_na(tsl_act_px_val, price_prec)}"
    )

    # --- Cancel Conflicting Open Orders (e.g. old limit orders not related to current TP/SL) ---
    signal = analysis_results["signal"]  # Current signal from analysis
    try:
        open_orders = await api_client.fetch_open_orders(symbol)
        for order in open_orders:
            order_id_to_check = order.get("id")
            order_side = order.get("side", "").lower()  # 'buy' or 'sell'
            order_price_str = str(order.get("price", "0"))
            order_price = Decimal(order_price_str) if order_price_str else None
            order_type = order.get("type", "").lower()

            # Determine if order is likely the current position's TP or SL
            is_current_tp = False
            if (
                cur_tp
                and order_price
                and abs(order_price - cur_tp) < min_tick * Decimal("0.5")
            ):
                is_current_tp = (pos_side == "long" and order_side == "sell") or (
                    pos_side == "short" and order_side == "buy"
                )

            is_current_sl = False  # Harder to match SL as it might be a stop market order without explicit price in `fetch_open_orders`
            # For Bybit, stop orders might appear differently. This check is basic.
            if (
                cur_sl
                and order_price
                and abs(order_price - cur_sl) < min_tick * Decimal("0.5")
            ):
                is_current_sl = (
                    pos_side == "long"
                    and order_side == "sell"
                    and order_type == "stop_market"
                ) or (
                    pos_side == "short"
                    and order_side == "buy"
                    and order_type == "stop_market"
                )

            # If the order opposes the *current position* and is NOT its TP/SL, consider canceling
            # Example: Position is LONG. A SELL limit order that is not the TP might be a remnant.
            conflicting = False
            if (
                pos_side == "long"
                and order_side == "sell"
                and not is_current_tp
                and not is_current_sl
            ):
                conflicting = True
            elif (
                pos_side == "short"
                and order_side == "buy"
                and not is_current_tp
                and not is_current_sl
            ):
                conflicting = True

            if conflicting:
                lg.warning(
                    f"{NEON_YELLOW}Canceling potentially conflicting open {order_side} order {order_id_to_check} for {symbol} position.{RESET}"
                )
                await api_client.cancel_order(order_id_to_check, symbol)
            elif is_current_tp or is_current_sl:
                lg.debug(
                    f"Keeping order {order_id_to_check} as it appears to be TP/SL for {symbol}."
                )

    except Exception as cancel_err:
        lg.error(
            f"Error during conflicting order cancellation for {symbol}: {cancel_err}",
            exc_info=True,
        )

    # --- Check Exit Signal ---
    # If current signal opposes the open position side
    if (pos_side == "long" and signal == "SELL") or (
        pos_side == "short" and signal == "BUY"
    ):
        lg.warning(
            f"{NEON_YELLOW}*** EXIT Signal ({_format_signal(signal)}) opposes current {_format_side(pos_side)} position for {symbol}. Closing... ***{RESET}"
        )
        if await _execute_close_position(
            api_client,
            symbol,
            market_info,
            open_position,
            config,
            lg,
            "opposing signal",
        ):
            return  # Position closed, no further management needed in this cycle

    # --- Time Exit ---
    time_exit_min_config = analysis_results.get("time_exit_minutes_config")
    if (
        isinstance(time_exit_min_config, (int, float))
        and time_exit_min_config > 0
        and pos_ts_ms
    ):
        try:
            current_time_ms = int(loop.time() * 1000)
            elapsed_min = (current_time_ms - pos_ts_ms) / 60000.0
            if elapsed_min >= time_exit_min_config:
                lg.warning(
                    f"{NEON_YELLOW}*** TIME EXIT for {symbol} ({elapsed_min:.1f}m >= configured {time_exit_min_config}m). Closing... ***{RESET}"
                )
                if await _execute_close_position(
                    api_client,
                    symbol,
                    market_info,
                    open_position,
                    config,
                    lg,
                    "time exit",
                ):
                    return  # Position closed
        except Exception as terr:
            lg.error(f"Time exit check error for {symbol}: {terr}", exc_info=True)

    # --- Break-Even --- (Only if not using exchange TSL, as they might conflict)
    be_enabled_config = analysis_results.get("be_enabled_config", False)
    if (
        be_enabled_config and not is_tsl_active_on_exchange
    ):  # Don't manage BE if exchange TSL is on
        lg.debug(f"--- Break-Even Check for {symbol} ---")
        try:
            if not (
                isinstance(current_atr, Decimal)
                and current_atr > 0
                and isinstance(min_tick, Decimal)
                and min_tick > 0
            ):
                raise ValueError(
                    f"Invalid ATR ({current_atr}) or MinTick ({min_tick}) for BE calculation."
                )

            be_trigger_atr_multiple = Decimal(
                str(config.get("break_even_trigger_atr_multiple", "1.0"))
            )
            be_offset_ticks = int(config.get("break_even_offset_ticks", 2))

            if (
                be_trigger_atr_multiple <= 0 or be_offset_ticks < 0
            ):  # Trigger multiple must be positive
                raise ValueError(
                    f"Invalid BE config: TriggerATR={be_trigger_atr_multiple}, OffsetTicks={be_offset_ticks}"
                )

            price_diff_from_entry = (
                (current_price - entry_px)
                if pos_side == "long"
                else (entry_px - current_price)
            )
            profit_in_atr = (
                price_diff_from_entry / current_atr
                if current_atr > 0
                else Decimal("Infinity")
            )  # Avoid div by zero

            if profit_in_atr >= be_trigger_atr_multiple:
                tick_offset_value = min_tick * Decimal(be_offset_ticks)
                raw_be_price = (
                    entry_px + tick_offset_value
                    if pos_side == "long"
                    else entry_px - tick_offset_value
                )
                # Round BE price to nearest tick in a favorable direction (or at entry)
                rounding_direction = ROUND_UP if pos_side == "long" else ROUND_DOWN
                be_stop_price = (raw_be_price / min_tick).quantize(
                    Decimal("1"), rounding=rounding_direction
                ) * min_tick

                # Ensure BE price is sensible (e.g., not worse than entry for a profit move)
                if (pos_side == "long" and be_stop_price < entry_px) or (
                    pos_side == "short" and be_stop_price > entry_px
                ):
                    be_stop_price = (entry_px / min_tick).quantize(
                        Decimal("1"), rounding=rounding_direction
                    ) * min_tick

                if be_stop_price <= 0:  # Should not happen with valid inputs
                    raise ValueError(
                        f"Calculated BE stop price is zero or negative: {be_stop_price}"
                    )

                current_sl_price = open_position.get("stopLossPriceDecimal")
                update_sl_needed = False
                if not current_sl_price:  # No SL set, so set BE SL
                    update_sl_needed = True
                elif (
                    pos_side == "long" and be_stop_price > current_sl_price
                ):  # Move SL up to BE
                    update_sl_needed = True
                elif (
                    pos_side == "short" and be_stop_price < current_sl_price
                ):  # Move SL down to BE
                    update_sl_needed = True

                if update_sl_needed:
                    lg.warning(
                        f"{NEON_YELLOW}*** Moving SL to Break-Even for {symbol} @ {_format_price_or_na(be_stop_price, price_prec)} (Profit ATR: {profit_in_atr:.2f} >= Trigger: {be_trigger_atr_multiple:.2f}) ***{RESET}"
                    )
                    # Keep existing TP if any, otherwise TP remains None (or 0 to remove if API requires)
                    tp_for_be_update = open_position.get("takeProfitPriceDecimal")
                    if await api_client._set_position_protection(
                        symbol,
                        market_info,
                        open_position,
                        stop_loss_price=be_stop_price,
                        take_profit_price=tp_for_be_update,
                    ):
                        lg.info(f"Break-Even SL update successful for {symbol}.")
                        open_position["stopLossPriceDecimal"] = (
                            be_stop_price  # Update local cache
                        )
                    else:
                        lg.error(
                            f"{NEON_RED}Break-Even SL update FAILED for {symbol}.{RESET}"
                        )
            else:
                lg.debug(
                    f"BE Target not reached for {symbol} (Profit ATR: {profit_in_atr:.2f} < Trigger: {be_trigger_atr_multiple:.2f})"
                )
        except Exception as be_err:
            lg.error(f"Break-Even check error for {symbol}: {be_err}", exc_info=True)

    # --- Trailing Stop Loss (TSL) Activation (if not already active on exchange) ---
    # This logic assumes TSL is to be activated once, then managed by the exchange.
    # If custom TSL logic is needed (bot manages TSL adjustments), this would be different.
    tsl_enabled_config = analysis_results.get("tsl_enabled_config", False)
    if tsl_enabled_config and not is_tsl_active_on_exchange:
        # Check if BE has already moved SL to entry or better. If so, TSL might take over or be redundant.
        # For simplicity, this attempts to set TSL if config says so and exchange doesn't show one.
        lg.debug(
            f"--- TSL Activation Check for {symbol} (Exchange TSL not detected) ---"
        )
        try:
            analyzer: TradingAnalyzer = analysis_results["analyzer"]
            # TP target for TSL might be based on initial entry or dynamically calculated
            # Here, using entry_px (actual position entry) and current position side.
            _, tsl_tp_target, _ = analyzer.calculate_entry_tp_sl(entry_px, pos_side)

            lg.info(
                f"Attempting to activate Trailing Stop Loss for {symbol}. Associated TP Target (if any): {_format_price_or_na(tsl_tp_target, price_prec)}"
            )
            if await api_client.set_trailing_stop_loss(
                symbol, open_position, config, tsl_tp_target
            ):
                lg.info(f"Trailing Stop Loss activation initiated for {symbol}.")
                # After this, is_tsl_active_on_exchange should become true on next cycle's fetch
            else:
                lg.warning(
                    f"{NEON_YELLOW}Trailing Stop Loss activation attempt failed for {symbol}.{RESET}"
                )
        except Exception as tsl_err:
            lg.error(f"TSL activation error for {symbol}: {tsl_err}", exc_info=True)

    lg.info(f"{NEON_CYAN}------------------------------------{RESET}")


# --- Main Analysis Function ---
async def analyze_and_trade_symbol(
    api_client: BybitAPI,
    symbol: str,
    config: Dict[str, Any],
    logger: logging.Logger,
    enable_trading: bool,
) -> None:
    """Main analysis and trading logic function for a single symbol per cycle."""
    lg = logger  # Use passed logger instance
    loop = asyncio.get_event_loop()
    cycle_start_time = loop.time()
    lg.debug(f"--- Start Cycle: {symbol} ---")

    # Fetch Market Data
    market_data = await _fetch_and_prepare_market_data(api_client, symbol, config, lg)
    if not market_data:
        lg.info(
            f"{NEON_BLUE}--- Cycle End ({symbol}, Market Data Fetch Failed) ---{RESET}\n"
        )
        return

    # Perform Analysis
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
            f"{NEON_BLUE}--- Cycle End ({symbol}, Trade Analysis Failed) ---{RESET}\n"
        )
        return

    # Trading Logic (if enabled)
    if not enable_trading:
        lg.info(
            f"{NEON_YELLOW}Trading is disabled by global flag. Skipping trade execution for {symbol}.{RESET}"
        )
        lg.info(f"{NEON_BLUE}--- Cycle End ({symbol}, Trading Disabled) ---{RESET}\n")
        return

    try:
        # Check Current Position
        open_position = await api_client.get_open_position(symbol)

        if open_position is None:  # No open position for this symbol
            await _handle_no_open_position(
                api_client, symbol, config, lg, market_data, analysis_results
            )
        else:  # There is an open position for this symbol
            await _manage_existing_open_position(
                api_client,
                symbol,
                config,
                lg,
                market_data,
                analysis_results,
                open_position,
            )
    except Exception as trade_err:
        lg.critical(
            f"{NEON_RED}!!! CRITICAL ERROR during trading logic execution for {symbol}: {trade_err} !!!{RESET}",
            exc_info=True,
        )
        # Potentially trigger circuit breaker or other master error handling here
        # e.g., api_client.trip_circuit_breaker(reason=f"Critical trading error: {symbol}")

    cycle_duration = loop.time() - cycle_start_time
    lg.info(
        f"{NEON_BLUE}---== Cycle End ({symbol}, Duration: {cycle_duration:.2f}s) ==---{RESET}\n"
    )
