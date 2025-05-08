```python
# File: trading_strategy.py
"""
Core trading strategy logic for analyzing market data, generating trading signals,
and managing positions on the Bybit exchange.
"""

# Standard library imports
import asyncio
import logging
import sys
import time  # Keep for time.monotonic() used in polling/timeouts
import traceback
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Union, Tuple

# Third-party imports
import pandas as pd

# Local application/library specific imports
try:
    from exchange_api import BybitAPI  # Import the class
except ImportError as e:
    _NEON_RED_FALLBACK = "\033[1;91m"
    _RESET_ALL_STYLE_FALLBACK = "\033[0m"
    print(
        f"{_NEON_RED_FALLBACK}CRITICAL ERROR: Failed import BybitAPI from exchange_api.py: {e}{_RESET_ALL_STYLE_FALLBACK}",
        file=sys.stderr,
    )
    traceback.print_exc(file=sys.stderr)
    raise

try:
    from analysis import TradingAnalyzer
    from risk_manager import calculate_position_size
    from utils import (
        CCXT_INTERVAL_MAP,
        DEFAULT_INDICATOR_PERIODS,
        NEON_BLUE,
        NEON_CYAN,
        NEON_GREEN,
        NEON_PURPLE,
        NEON_RED,
        NEON_YELLOW,
        RESET_ALL_STYLE,
    )
except ImportError as e:
    _NEON_RED_FALLBACK = "\033[1;91m"
    _RESET_ALL_STYLE_FALLBACK = "\033[0m"
    print(
        f"{_NEON_RED_FALLBACK}CRITICAL ERROR: Failed import modules in trading_strategy.py: {e}{_RESET_ALL_STYLE_FALLBACK}",
        file=sys.stderr,
    )
    traceback.print_exc(file=sys.stderr)
    raise

# --- Formatting Helpers ---
def _format_signal(signal_text: Any) -> str:
    """Formats a trading signal string with appropriate colors."""
    signal_str = str(signal_text).upper()
    if signal_str == "BUY":
        return f"{NEON_GREEN}{signal_str}{RESET_ALL_STYLE}"
    if signal_str == "SELL":
        return f"{NEON_RED}{signal_str}{RESET_ALL_STYLE}"
    if signal_str == "HOLD":
        return f"{NEON_YELLOW}{signal_str}{RESET_ALL_STYLE}"
    return f"{signal_str}{RESET_ALL_STYLE}"


def _format_side(side_text: Optional[str]) -> str:
    """Formats a position side string (long/short) with appropriate colors."""
    if side_text is None:
        return f"{NEON_YELLOW}UNKNOWN{RESET_ALL_STYLE}"
    side_upper = side_text.upper()
    if side_upper == "LONG":
        return f"{NEON_GREEN}{side_upper}{RESET_ALL_STYLE}"
    if side_upper == "SHORT":
        return f"{NEON_RED}{side_upper}{RESET_ALL_STYLE}"
    return f"{NEON_YELLOW}{side_upper}{RESET_ALL_STYLE}"


def _format_price_or_na(
    price_val: Optional[Union[Decimal, str, float]],
    precision_places: int,
    label: str = "",
) -> str:
    """Formats a price Decimal to a string with specified precision, or 'N/A'."""
    color = NEON_CYAN
    na_color = NEON_YELLOW
    if price_val is None or (isinstance(price_val, str) and not price_val.strip()):
        return f"{na_color}N/A{RESET_ALL_STYLE}"
    try:
        d_val = Decimal(str(price_val))
        # Allow specific labels (like ATR) to display as 0.00000 if that's their value.
        # For other prices, 0.0 might be an error or uninitialized.
        if d_val == Decimal(0) and label.upper() != "ATR":
            if precision_places > 0:
                return f"{na_color}{d_val:.{precision_places}f}{RESET_ALL_STYLE}"
            return f"{na_color}0.0{RESET_ALL_STYLE}"

        format_str = f":.{precision_places}f"
        return f"{color}{d_val:{format_str}}{RESET_ALL_STYLE}"
    except (InvalidOperation, TypeError, ValueError):
        return f"{na_color}{price_val} (invalid){RESET_ALL_STYLE}"


# --- Core Trading Logic Helpers ---
async def _set_position_protection_logic(
    api_client: BybitAPI,
    symbol: str,
    market_info: Dict[str, Any],
    confirmed_pos: Dict[str, Any],
    config: Dict[str, Any],
    lg: logging.Logger,
    analysis_results: Dict[str, Any],
    price_precision: int,
    entry_signal: str,  # Original 'BUY' or 'SELL' signal for the entry
) -> bool:
    """
    Calculates and sets Stop Loss (SL), Take Profit (TP), or Trailing Stop Loss (TSL)
    for a confirmed position.
    Returns True if protection was set successfully or not required, False otherwise.
    """
    lg.info(f"Proceeding with protection setup for confirmed {symbol} position...")
    try:
        entry_px_actual = confirmed_pos.get("entryPriceDecimal")
        analyzer: TradingAnalyzer = analysis_results["analyzer"]

        if not (entry_px_actual and isinstance(entry_px_actual, Decimal) and entry_px_actual > Decimal(0)):
            lg.warning(
                f"{NEON_YELLOW}Actual entry price for {symbol} missing/invalid. "
                f"Using analysis reference price estimate ({analyzer.indicator_values.get('Close')}).{RESET_ALL_STYLE}"
            )
            entry_px_actual = analyzer.indicator_values.get("Close")  # Fallback to kline close price used in analysis
            if not (entry_px_actual and isinstance(entry_px_actual, Decimal) and entry_px_actual > Decimal(0)):
                lg.error(f"{NEON_RED}Cannot determine a valid entry price for protection setup on {symbol}. Aborting.{RESET_ALL_STYLE}")
                return False

        # Recalculate TP/SL targets using the actual or best-estimated entry price
        _, tp_target_decimal, sl_target_decimal = analyzer.calculate_entry_tp_sl(entry_px_actual, entry_signal)
        lg.info(
            f"Protection targets for {symbol} based on Entry {_format_price_or_na(entry_px_actual, price_precision)}: "
            f"SL={_format_price_or_na(sl_target_decimal, price_precision)}, "
            f"TP={_format_price_or_na(tp_target_decimal, price_precision)}"
        )

        tsl_enabled_config = analysis_results.get("tsl_enabled_config", False)
        protection_set_successfully = False

        if tsl_enabled_config:
            lg.info(f"Setting Trailing Stop Loss for {symbol}...")
            # Pass TP target (Decimal, 0 Decimal to remove, or None to leave unchanged)
            tp_param_for_tsl = tp_target_decimal if isinstance(tp_target_decimal, Decimal) else None
            protection_set_successfully = await api_client.set_trailing_stop_loss(
                symbol, confirmed_pos, config, tp_param_for_tsl
            )
        # Check if valid fixed SL or TP exists (allow Decimal(0) to remove protection)
        elif (sl_target_decimal is not None and isinstance(sl_target_decimal, Decimal) and sl_target_decimal >= Decimal(0)) or \
             (tp_target_decimal is not None and isinstance(tp_target_decimal, Decimal) and tp_target_decimal >= Decimal(0)):
            lg.info(f"Setting Fixed SL/TP for {symbol}...")
            protection_set_successfully = await api_client.set_position_protection( # Use the public method if available
                symbol, market_info, confirmed_pos, sl_target_decimal, tp_target_decimal
            )
        else:
            lg.info(f"No valid protection targets (SL/TP are None or invalid) and TSL disabled. No protection action taken for {symbol}.")
            protection_set_successfully = True  # Considered success as no action was needed

        if protection_set_successfully:
            lg.info(f"{NEON_GREEN}Protection setup successful or not required for {symbol}.{RESET_ALL_STYLE}")
        else:
            lg.error(f"{NEON_RED}Protection setup FAILED for {symbol}. Manual check advised.{RESET_ALL_STYLE}")
        return protection_set_successfully

    except Exception as protect_err:
        lg.error(f"{NEON_RED}Error during protection setup for {symbol}: {protect_err}{RESET_ALL_STYLE}", exc_info=True)
        return False


async def _execute_close_position(
    api_client: BybitAPI,
    symbol: str,
    market_info: Dict[str, Any],
    open_position: Dict[str, Any],
    config: Dict[str, Any],
    lg: logging.Logger,
    reason: str = "exit signal",
) -> bool:
    """
    Executes a market order to close an existing position.
    Returns True if the close order was successfully placed and confirmed (or likely closed), False otherwise.
    """
    pos_side = open_position.get("side")  # 'long' or 'short'
    pos_size = open_position.get("contractsDecimal")
    amount_precision = market_info.get("amountPrecisionPlaces", 8) # Use 'amountPrecisionPlaces' from get_market_info

    if not (pos_side in ["long", "short"] and isinstance(pos_size, Decimal) and pos_size > Decimal(0)):
        lg.warning(
            f"Close attempt for {symbol} ({reason}) skipped: Invalid position data. "
            f"Side='{pos_side}', Size='{pos_size}'.{RESET_ALL_STYLE}"
        )
        return False

    try:
        # Determine the signal needed to close the position
        close_signal = "SELL" if pos_side == "long" else "BUY"
        lg.warning(
            f"{NEON_YELLOW}==> Initiating CLOSE for {_format_side(pos_side)} position on {symbol} "
            f"(Reason: {reason}) | Size: {pos_size:.{amount_precision}f} <=="
            f"{RESET_ALL_STYLE}"
        )

        close_order = await api_client.place_trade(
            symbol=symbol,
            trade_signal=close_signal,
            position_size=pos_size,
            order_type="market",
            reduce_only=True, # Ensure this order only reduces/closes the position
        )

        if close_order and close_order.get("id"):
            lg.info(
                f"{NEON_GREEN}CLOSE order for {symbol} placed successfully. Order ID: {close_order['id']}, "
                f"Status: {close_order.get('status', 'N/A')}{RESET_ALL_STYLE}"
            )
            # Optional: Wait and confirm position closure
            close_confirm_delay = config.get("close_confirm_delay_seconds", 2.0)
            await asyncio.sleep(close_confirm_delay)
            final_pos_check = await api_client.get_open_position(symbol)
            if final_pos_check is None:
                lg.info(f"{NEON_GREEN}Position closure for {symbol} confirmed (no open position found).{RESET_ALL_STYLE}")
            else:
                lg.warning(
                    f"{NEON_YELLOW}Position closure for {symbol} NOT fully confirmed. "
                    f"Current size: {final_pos_check.get('contractsDecimal')}{RESET_ALL_STYLE}"
                )
            return True # Assume success if order placed, confirmation is best-effort
        else:
            lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. API call failed or returned invalid data.{RESET_ALL_STYLE}")
            return False
    except Exception as e:
        lg.error(f"{NEON_RED}Error during position closing for {symbol}: {e}{RESET_ALL_STYLE}", exc_info=True)
        return False


async def _fetch_and_prepare_market_data(
    api_client: BybitAPI, symbol: str, config: Dict[str, Any], lg: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Fetches and prepares market information, klines, current price, and orderbook data."""
    lg.debug(f"Fetching market data for {symbol}...")
    market_info = await api_client.get_market_info(symbol)
    if not market_info:
        lg.error(f"Failed to get market info for {symbol}. Cannot proceed.")
        return None

    price_precision = market_info.get("pricePrecisionPlaces", 4)
    amount_precision = market_info.get("amountPrecisionPlaces", 8)
    min_tick_size = market_info.get("minTickSizeDecimal", Decimal("1e-4")) # Ensure this is Decimal
    min_qty = market_info.get("minOrderQtyDecimal", Decimal("0")) # Ensure this is Decimal

    interval_str = str(config.get("interval", "5")) # e.g., "1", "5", "15", "60", "D"
    ccxt_interval = CCXT_INTERVAL_MAP.get(interval_str)
    if not ccxt_interval:
        lg.error(f"Invalid interval '{interval_str}' specified for {symbol}.")
        return None

    kline_limit = max(1, int(config.get("kline_limit", 500)))
    klines_df = await api_client.fetch_klines(symbol, ccxt_interval, limit=kline_limit)
    min_kline_len = max(1, int(config.get("min_kline_length", 100)))

    if klines_df is None or klines_df.empty or len(klines_df) < min_kline_len:
        lg.error(
            f"Insufficient klines data for {symbol}. "
            f"Fetched {len(klines_df) if klines_df is not None else 0}, required {min_kline_len}."
        )
        return None

    current_price = await api_client.fetch_current_price(symbol)
    if not (current_price and isinstance(current_price, Decimal) and current_price > Decimal(0)):
        last_close_price = klines_df["close"].iloc[-1]
        if isinstance(last_close_price, Decimal) and last_close_price > Decimal(0):
            current_price = last_close_price
            lg.warning(f"Using last kline close price as current price for {symbol}: {_format_price_or_na(current_price, price_precision)}")
        else:
            lg.error(f"Cannot obtain a valid current price for {symbol}.")
            return None
    lg.debug(f"Current price for {symbol}: {_format_price_or_na(current_price, price_precision)}")

    orderbook_data = None
    orderbook_enabled_in_config = config.get("indicators", {}).get("orderbook", False)
    active_weight_set_name = config.get("active_weight_set", "default")
    active_weights = config.get("weight_sets", {}).get(active_weight_set_name, {})
    try:
        orderbook_weight = Decimal(str(active_weights.get("orderbook", "0")))
    except (InvalidOperation, TypeError):
        orderbook_weight = Decimal("0")

    if orderbook_enabled_in_config and orderbook_weight != Decimal("0"):
        ob_limit = max(1, int(config.get("orderbook_limit", 25)))
        orderbook_data = await api_client.fetch_orderbook(symbol, limit=ob_limit)
        if not orderbook_data:
            lg.warning(f"Failed to fetch orderbook for {symbol}, proceeding without it.")
    elif orderbook_enabled_in_config:
        lg.debug(f"Orderbook indicator is enabled for {symbol} but its weight is zero in set '{active_weight_set_name}'.")
    else:
        lg.debug(f"Orderbook indicator is disabled for {symbol} in configuration.")

    return {
        "market_info": market_info,
        "klines_df": klines_df,
        "current_price_decimal": current_price,
        "orderbook_data": orderbook_data,
        "price_precision": price_precision,
        "amount_precision": amount_precision,
        "min_tick_size": min_tick_size,
        "min_qty": min_qty,
    }


def _perform_trade_analysis(
    klines_df: pd.DataFrame,
    current_price_decimal: Decimal,
    orderbook_data: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    market_info: Dict[str, Any], # Contains symbol, precisions, etc.
    lg: logging.Logger,
    price_precision: int, # Passed explicitly for formatting
) -> Optional[Dict[str, Any]]:
    """
    Performs trading analysis using TradingAnalyzer and returns key results.
    """
    symbol = market_info.get('symbol', 'UnknownSymbol')
    try:
        analyzer = TradingAnalyzer(klines_df.copy(), lg, config, market_info)
    except Exception as e:
        lg.error(f"Failed to initialize TradingAnalyzer for {symbol}: {e}", exc_info=True)
        return None

    if not analyzer.indicator_values: # Check if indicators were successfully calculated
        lg.error(f"Indicator calculation failed for {symbol}. Cannot generate signal.")
        return None

    # Generate trading signal based on current state
    signal = analyzer.generate_trading_signal(current_price_decimal, orderbook_data)
    # Calculate initial TP/SL targets based on the current price and generated signal
    _, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price_decimal, signal)
    current_atr = analyzer.indicator_values.get("ATR")

    lg.info(f"--- {NEON_PURPLE}Analysis Summary ({symbol}){RESET_ALL_STYLE} ---")
    lg.info(f"  Current Price: {_format_price_or_na(current_price_decimal, price_precision)}")
    atr_period_cfg = config.get("atr_period", DEFAULT_INDICATOR_PERIODS.get("atr_period", 14))
    # ATR often benefits from slightly higher display precision
    lg.info(f"  ATR ({atr_period_cfg}): {_format_price_or_na(current_atr, price_precision + 2, 'ATR')}")
    lg.info(f"  Initial SL (for sizing): {_format_price_or_na(sl_calc, price_precision, 'SL Calc')}")
    lg.info(f"  Initial TP (potential target): {_format_price_or_na(tp_calc, price_precision, 'TP Calc')}")

    tsl_enabled = config.get("enable_trailing_stop", False)
    be_enabled = config.get("enable_break_even", False)
    time_exit_val = config.get("time_based_exit_minutes")
    time_exit_str = f"{time_exit_val:.1f} min" if isinstance(time_exit_val, (int, float)) and time_exit_val > 0 else "Disabled"
    lg.info(f"  Risk Mgt Config: TSL={tsl_enabled}, BreakEven={be_enabled}, TimeExit={time_exit_str}")
    lg.info(f"  Generated Signal: {_format_signal(signal)}")
    lg.info("-----------------------------")

    return {
        "signal": signal, # "BUY", "SELL", or "HOLD"
        "tp_calc": tp_calc, # Calculated TP based on current_price_decimal
        "sl_calc": sl_calc, # Calculated SL based on current_price_decimal
        "analyzer": analyzer, # Instance of TradingAnalyzer
        "tsl_enabled_config": tsl_enabled,
        "be_enabled_config": be_enabled,
        "time_exit_minutes_config": time_exit_val,
        "current_atr": current_atr, # Current ATR value
    }

async def _await_entry_order_confirmation_and_protect(
    api_client: BybitAPI,
    symbol: str,
    order_id: str,
    initial_order_status: str,
    entry_type: str, # "market", "limit", "conditional"
    expected_size: Decimal,
    limit_price_for_api: Optional[Decimal], # For logging limit orders
    # conditional_trigger_price: Optional[Decimal], # For logging conditional orders (if needed)
    market_data: Dict[str, Any],
    config: Dict[str, Any],
    lg: logging.Logger,
    analysis_results: Dict[str, Any],
    entry_signal: str # Original "BUY" or "SELL" signal
) -> bool:
    """
    Waits for an entry order to fill (if not market), confirms the position,
    and sets protection (SL/TP/TSL).
    Returns True if entry and protection were successful, False otherwise.
    """
    loop = asyncio.get_event_loop()
    confirmed_pos: Optional[Dict[str, Any]] = None
    market_info = market_data["market_info"]
    price_precision = market_data["price_precision"]

    # Case 1: Market order or already filled limit/conditional order
    if entry_type == "market" or \
       (entry_type in ["limit", "conditional"] and initial_order_status == "closed"):
        confirm_delay = config.get("order_confirmation_delay_seconds", 0.75)
        max_retries = config.get("position_confirm_retries", 3)
        overall_timeout_sec = config.get("protection_setup_timeout_seconds", 30.0)
        lg.info(
            f"Order {order_id} ({entry_type}, status: {initial_order_status}) for {symbol}. "
            f"Waiting up to {max_retries} attempts / {overall_timeout_sec:.1f}s for position confirmation..."
        )

        start_time = loop.time()
        for attempt in range(max_retries):
            if loop.time() - start_time > overall_timeout_sec:
                lg.error(f"{NEON_RED}Position confirmation timeout for {symbol} after order {order_id}.{RESET_ALL_STYLE}")
                break
            
            await asyncio.sleep(confirm_delay * (attempt + 1)) # Increasing delay

            order_check = await api_client.fetch_order(order_id, symbol)
            if order_check and order_check.get("status") == "closed":
                confirmed_pos = await api_client.get_open_position(symbol)
                if confirmed_pos:
                    filled_size_str = str(order_check.get("filled", "0"))
                    filled_size = Decimal(filled_size_str) if filled_size_str else Decimal("0")
                    # Handle potential partial fills by updating position size if significantly different
                    if filled_size > Decimal(0) and abs(filled_size - expected_size) > (expected_size * Decimal("0.01")):
                        lg.warning(
                            f"{NEON_YELLOW}Partial Fill! Order {order_id} for {symbol}: "
                            f"Expected {expected_size}, Filled {filled_size}. Using filled size for protection.{RESET_ALL_STYLE}"
                        )
                        confirmed_pos["contractsDecimal"] = filled_size # Ensure protection logic uses actual filled size
                        # Also update common alternative keys if used by API client's position structure
                        if "size" in confirmed_pos: confirmed_pos["size"] = filled_size 
                    break # Position confirmed or order confirmed as filled
            elif order_check is None:
                lg.warning(f"Order {order_id} for {symbol} not found during confirmation poll (Attempt {attempt + 1}). May have been insta-filled & cleared.")
                # Check for position anyway, might have filled too fast for order to be queryable
                confirmed_pos = await api_client.get_open_position(symbol)
                if confirmed_pos: break
            else:
                lg.warning(
                    f"Position confirmation attempt {attempt + 1} for {symbol}: Order {order_id} status: "
                    f"'{order_check.get('status') if order_check else 'Not Found'}'. Retrying..."
                )
        else: # Loop finished without break (max_retries reached)
             lg.error(
                f"{NEON_RED}FAILED TO CONFIRM open position for {symbol} after order {order_id} was placed/filled! "
                f"Manual check required!{RESET_ALL_STYLE}"
            )
             return False # Failed to confirm

        if confirmed_pos:
            lg.info(f"{NEON_GREEN}Position Confirmed for {symbol}! Setting protection...{RESET_ALL_STYLE}")
            return await _set_position_protection_logic(
                api_client, symbol, market_info, confirmed_pos, config, lg,
                analysis_results, price_precision, entry_signal
            )
        elif initial_order_status == "closed" and not confirmed_pos: # Order filled but no position?
            lg.error(
                f"{NEON_RED}Order {order_id} for {symbol} is 'closed' but no open position found. "
                f"Possible immediate fill & exit, or API consistency issue.{RESET_ALL_STYLE}"
            )
            return False # Unclear state
        # If not confirmed_pos and order wasn't 'closed' initially, means it failed to confirm within retries.

    # Case 2: Limit or conditional order that is still open or partially filled
    elif entry_type in ["limit", "conditional"] and initial_order_status in ["open", "partially_filled"]:
        lg.info(
            f"{NEON_YELLOW}{entry_type.capitalize()} order {order_id} is '{initial_order_status}' for {symbol} "
            f"@ price {_format_price_or_na(limit_price_for_api, price_precision)}. Monitoring for fill...{RESET_ALL_STYLE}"
        )
        max_wait_fill_seconds = config.get("limit_order_timeout_seconds", 300.0)
        poll_interval = config.get("limit_order_poll_interval_seconds", 5.0)
        stale_timeout_seconds = config.get("limit_order_stale_timeout_seconds", 600.0) # If order not updated
        
        order_timestamp_ms = (await api_client.fetch_order(order_id, symbol) or {}).get("timestamp", int(loop.time() * 1000))

        start_monitoring_time = loop.time()
        while loop.time() - start_monitoring_time < max_wait_fill_seconds:
            await asyncio.sleep(poll_interval)
            current_order_state = await api_client.fetch_order(order_id, symbol)

            if not current_order_state:
                lg.warning(f"{entry_type.capitalize()} order {order_id} for {symbol} disappeared during monitoring (possibly canceled externally).")
                return False # Order gone

            current_status = current_order_state.get("status", "").lower()
            last_update_ts_ms = current_order_state.get("lastUpdateTimestamp") or \
                                current_order_state.get("timestamp", order_timestamp_ms)

            # Stale check (if configured)
            if stale_timeout_seconds > 0 and last_update_ts_ms and \
               (loop.time() * 1000 - last_update_ts_ms) / 1000 > stale_timeout_seconds:
                lg.warning(
                    f"{NEON_YELLOW}{entry_type.capitalize()} order {order_id} for {symbol} "
                    f"considered stale (no update for > {stale_timeout_seconds}s). Canceling.{RESET_ALL_STYLE}"
                )
                await api_client.cancel_order(order_id, symbol)
                return False # Canceled due to staleness

            if current_status == "closed": # Order filled
                lg.info(f"{NEON_GREEN}{entry_type.capitalize()} order {order_id} for {symbol} filled.{RESET_ALL_STYLE}")
                filled_size_str = str(current_order_state.get("filled", "0"))
                filled_size = Decimal(filled_size_str) if filled_size_str else Decimal("0")

                if filled_size <= Decimal(0):
                    lg.error(
                        f"{NEON_RED}Order {order_id} for {symbol} 'closed' but filled size is {filled_size}. "
                        f"Aborting protection setup.{RESET_ALL_STYLE}"
                    )
                    return False # No fill amount

                if filled_size < expected_size * Decimal("0.99"): # Significant partial fill
                    lg.warning(
                        f"{NEON_YELLOW}Partial Fill for {entry_type} order {order_id} on {symbol}! "
                        f"Expected {expected_size}, Filled {filled_size}.{RESET_ALL_STYLE}"
                    )
                    # Strategy may need to decide if to proceed with partial fill. Here, we do.

                confirmed_pos = await api_client.get_open_position(symbol)
                if confirmed_pos:
                    # Ensure position size reflects actual filled amount for protection logic
                    confirmed_pos["contractsDecimal"] = filled_size
                    if "size" in confirmed_pos: confirmed_pos["size"] = filled_size
                    lg.info(
                        f"{NEON_GREEN}Position Confirmed for {symbol} after {entry_type} fill! Setting protection...{RESET_ALL_STYLE}"
                    )
                    return await _set_position_protection_logic(
                        api_client, symbol, market_info, confirmed_pos, config, lg,
                        analysis_results, price_precision, entry_signal
                    )
                else:
                    lg.error(
                        f"{NEON_RED}{entry_type.capitalize()} order {order_id} filled for {symbol}, "
                        f"but FAILED to confirm an open position! Manual check!{RESET_ALL_STYLE}"
                    )
                return False # Filled but no position found
            
            elif current_status == "canceled":
                lg.info(f"{NEON_RED}{entry_type.capitalize()} order {order_id} for {symbol} was canceled.{RESET_ALL_STYLE}")
                return False # Canceled
            elif current_status not in ["open", "partially_filled"]:
                lg.error(
                    f"{NEON_RED}{entry_type.capitalize()} order {order_id} for {symbol} has unexpected status '{current_status}'. "
                    f"Stopping monitor.{RESET_ALL_STYLE}"
                )
                return False # Unexpected status
            # Else: still open/partially_filled, continue polling
        else: # Polling loop timed out
            lg.warning(
                f"{NEON_YELLOW}{entry_type.capitalize()} order {order_id} for {symbol} not filled within "
                f"{max_wait_fill_seconds:.1f}s timeout. Canceling.{RESET_ALL_STYLE}"
            )
            await api_client.cancel_order(order_id, symbol)
            return False # Timed out and canceled
    else: # e.g., initial status rejected, expired
        lg.error(
            f"{NEON_RED}Order {order_id} for {symbol} has initial status '{initial_order_status}' "
            f"which is not 'closed', 'open', or 'partially_filled'. Manual check required!{RESET_ALL_STYLE}"
        )
        return False # Unhandled initial status

    return False # Default to failure if no path leads to success

async def _handle_no_open_position(
    api_client: BybitAPI,
    symbol: str,
    config: Dict[str, Any],
    lg: logging.Logger,
    market_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
):
    """Handles logic for entering a new position if conditions are met."""
    entry_signal = analysis_results["signal"] # "BUY", "SELL", or "HOLD"
    if entry_signal not in ["BUY", "SELL"]:
        lg.info(f"Signal is {_format_signal(entry_signal)} for {symbol} with no open position. No entry action.")
        return

    lg.info(f"{NEON_PURPLE}*** {_format_signal(entry_signal)} Signal & No Position: Evaluating Entry for {symbol} ***{RESET_ALL_STYLE}")

    # --- Pre-entry Checks ---
    if api_client.circuit_breaker_tripped:
        lg.error(f"{NEON_RED}Circuit breaker is tripped. Skipping entry for {symbol}.{RESET_ALL_STYLE}")
        return

    quote_currency = config.get("quote_currency", "USDT")
    balance = await api_client.fetch_balance(quote_currency)
    if not (balance and isinstance(balance, Decimal) and balance > Decimal(0)):
        lg.error(f"Entry Aborted for {symbol}: Invalid or zero balance for {quote_currency} ({balance}).")
        return

    try:
        risk_pct_str = str(config.get("risk_per_trade", "0.01")) # Default 1% risk
        risk_pct = Decimal(risk_pct_str)
        if not (Decimal(0) < risk_pct <= Decimal(1)): # risk_pct is a fraction of balance, e.g. 0.01 for 1%
            lg.error(f"Entry Aborted for {symbol}: Invalid risk_per_trade '{risk_pct_str}'. Must be > 0 and <= 1.")
            return
    except (InvalidOperation, TypeError):
        lg.error(f"Entry Aborted for {symbol}: Invalid format for risk_per_trade in config.")
        return

    sl_for_sizing = analysis_results["sl_calc"] # Initial SL price calculated from analysis
    if not (sl_for_sizing and isinstance(sl_for_sizing, Decimal) and sl_for_sizing > Decimal(0)):
        lg.error(f"Entry Aborted for {symbol}: Invalid Stop Loss price for position sizing: {sl_for_sizing}.")
        return

    market_info = market_data["market_info"]
    current_price = market_data["current_price_decimal"]

    # --- Set Leverage (if applicable) ---
    if market_info.get("is_contract_market"): # Check if it's a futures/swaps market
        leverage_val = max(1, int(config.get("leverage", 1)))
        if not await api_client.set_leverage(symbol, leverage_val):
            lg.error(f"Entry Aborted for {symbol}: Failed to set leverage to {leverage_val}x.")
            return

    # --- Calculate Position Size ---
    position_size_decimal = calculate_position_size(
        balance, risk_pct, sl_for_sizing, current_price, market_info, api_client.exchange, lg
    )
    if not (position_size_decimal and position_size_decimal > Decimal(0)):
        lg.error(f"Entry Aborted for {symbol}: Position size calculation result is invalid or zero ({position_size_decimal}).")
        return
    
    # --- Pre-placement Validations ---
    entry_order_type = config.get("entry_order_type", "market").lower()
    limit_price_for_api: Optional[Decimal] = None
    min_qty = market_data["min_qty"]
    min_tick_size = market_data["min_tick_size"]

    try:
        if min_qty > Decimal(0) and position_size_decimal < min_qty:
            raise ValueError(f"Calculated size {position_size_decimal} is less than minimum order quantity {min_qty}.")

        if entry_order_type == "limit":
            use_ob_price_for_limit = config.get("adjust_price_to_orderbook", False)
            ob_data = market_data["orderbook_data"]
            if use_ob_price_for_limit and ob_data and ob_data.get("bids") and ob_data.get("asks"):
                top_ask_price = Decimal(str(ob_data["asks"][0][0]))
                top_bid_price = Decimal(str(ob_data["bids"][0][0]))
                limit_price_candidate = top_bid_price if entry_signal == "BUY" else top_ask_price
            else: # Use percentage offset from current price
                offset_config_key = f"limit_order_offset_{entry_signal.lower()}" # e.g., limit_order_offset_buy
                offset_pct_str = str(config.get(offset_config_key, "0.0005")) # Default 0.05%
                offset_percentage = Decimal(offset_pct_str)
                if entry_signal == "BUY":
                    limit_price_candidate = current_price * (Decimal("1") - offset_percentage)
                else: # SELL
                    limit_price_candidate = current_price * (Decimal("1") + offset_percentage)
            
            # Round to tick size
            if min_tick_size > Decimal(0):
                rounding_mode = ROUND_DOWN if entry_signal == "BUY" else ROUND_UP # Favorable rounding for limit orders
                limit_price_for_api = (limit_price_candidate / min_tick_size).quantize(Decimal("1"), rounding=rounding_mode) * min_tick_size
            else: # Fallback if min_tick_size is zero or invalid (should not happen for valid market_info)
                limit_price_for_api = limit_price_candidate.quantize(Decimal("1e-8")) # Generic precision

            if not (limit_price_for_api and limit_price_for_api > Decimal(0)):
                raise ValueError(f"Limit price calculation resulted in an invalid price: {limit_price_for_api}.")

            # Validate against exchange price limits
            price_limits_info = market_info.get("limits", {}).get("price", {})
            min_price_str = str(price_limits_info.get("min", "0"))
            max_price_str = str(price_limits_info.get("max", "inf"))
            min_allowable_price = Decimal(min_price_str) if min_price_str and min_price_str != "0" else Decimal("0")
            max_allowable_price = Decimal(max_price_str) if max_price_str and max_price_str.lower() != "inf" else Decimal("Infinity")
            
            if not (min_allowable_price <= limit_price_for_api <= max_allowable_price):
                raise ValueError(
                    f"Calculated limit price {limit_price_for_api} is outside exchange limits "
                    f"[{min_allowable_price}, {max_allowable_price}]."
                )
            
            # Validate order cost against minimum cost if applicable
            order_cost = position_size_decimal * limit_price_for_api
            min_cost_str = str(market_info.get("limits", {}).get("cost", {}).get("min", "0"))
            min_order_cost = Decimal(min_cost_str) if min_cost_str and min_cost_str != "0" else Decimal("0")
            if min_order_cost > Decimal(0) and order_cost < min_order_cost:
                formatted_cost = f"{order_cost:.{market_data['price_precision'] + market_data['amount_precision']}f}"
                raise ValueError(f"Order cost {formatted_cost} is less than minimum required cost {min_order_cost}.")

    except ValueError as val_err:
        lg.error(f"Entry Aborted for {symbol}: Validation Error - {val_err}")
        return
    except Exception as e: # Catch any other unexpected errors during validation
        lg.error(f"Entry Aborted for {symbol}: Pre-placement validation error - {e}", exc_info=True)
        return

    # --- Determine Final Order Parameters ---
    conditional_trigger_price: Optional[Decimal] = None
    # Placeholder for conditional order logic; ensure trigger_price is correctly set if entry_order_type == "conditional"
    if entry_order_type == "conditional":
        lg.warning(f"Conditional order logic for {symbol} is currently a placeholder. Ensure trigger price is set if used.")
        # Example: conditional_trigger_price = ... calculated based on config and current_price ...

    time_in_force_config = config.get("time_in_force") # e.g., "GTC", "IOC", "FOK"
    post_only_config = config.get("post_only", False) if entry_order_type == "limit" else False
    # Generate a unique client order ID (ensure it meets exchange length limits, e.g., Bybit typically 32-36 chars)
    client_order_id = f"{symbol.replace('/', '')[:10]}_{entry_signal[:1]}_{int(time.monotonic() * 1000)}"
    client_order_id = client_order_id[-32:] # Truncate to a common safe length

    trigger_by_config = config.get("trigger_by") # For conditional/stop orders: 'LastPrice', 'MarkPrice', 'IndexPrice'

    # --- Cancel Potentially Conflicting Open Orders ---
    # (e.g., old limit orders that might interfere with this new entry)
    try:
        open_orders = await api_client.fetch_open_orders(symbol)
        for order in open_orders:
            order_side = order.get("side", "").lower() # 'buy' or 'sell'
            # Cancel open orders that oppose the new signal
            # (e.g., if new signal is BUY, cancel existing SELL limit orders)
            if (entry_signal == "BUY" and order_side == "sell") or \
               (entry_signal == "SELL" and order_side == "buy"):
                # This assumes no existing position. If there was one, TP/SL orders might be caught here.
                # The function name _handle_no_open_position implies this is safe.
                lg.warning(
                    f"{NEON_YELLOW}Canceling conflicting open {order_side} order {order['id']} "
                    f"for {symbol} before placing new {entry_signal} order.{RESET_ALL_STYLE}"
                )
                await api_client.cancel_order(order["id"], symbol)
    except Exception as cancel_err:
        lg.error(f"Failed to cancel conflicting orders for {symbol}: {cancel_err}. Proceeding with caution.")

    # --- Place Trade ---
    trade_order_response = await api_client.place_trade(
        symbol=symbol,
        trade_signal=entry_signal,
        position_size=position_size_decimal,
        order_type=entry_order_type,
        limit_price=limit_price_for_api, # Will be None for market orders
        reduce_only=False, # This is an entry order
        time_in_force=time_in_force_config,
        post_only=post_only_config,
        trigger_price=conditional_trigger_price, # For conditional orders
        trigger_by=trigger_by_config,
        client_order_id=client_order_id,
        # params={}, # For any exchange-specific additional parameters
    )

    if not (trade_order_response and trade_order_response.get("id")):
        lg.error(f"{NEON_RED}=== TRADE ENTRY FAILED for {symbol} ({entry_signal}). Order placement API call failed. ==={RESET_ALL_STYLE}")
        return

    order_id = trade_order_response["id"]
    order_status_initial = trade_order_response.get("status", "unknown").lower()
    lg.info(f"Trade order {order_id} ({entry_order_type}) placed for {symbol}. Initial status: '{order_status_initial}'.")

    # --- Post-Placement: Confirmation & Protection Setup ---
    entry_and_protection_ok = await _await_entry_order_confirmation_and_protect(
        api_client, symbol, order_id, order_status_initial, entry_order_type,
        position_size_decimal, limit_price_for_api, # conditional_trigger_price,
        market_data, config, lg, analysis_results, entry_signal
    )

    if entry_and_protection_ok:
        lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE for {symbol} ({entry_signal}) ==={RESET_ALL_STYLE}")
    else:
        lg.warning(
            f"{NEON_YELLOW}=== TRADE ENTRY for {symbol} ({entry_signal}) COMPLETED, "
            f"but subsequent confirmation or protection setup FAILED/SKIPPED. ==={RESET_ALL_STYLE}"
        )
        # Further actions might be needed here: e.g., try to close the (partially) opened position if protection is critical.
        # For now, it logs a warning.


async def _cancel_conflicting_orders_for_active_position(
    api_client: BybitAPI,
    symbol: str,
    open_position: Dict[str, Any],
    min_tick_size: Decimal,
    lg: logging.Logger
) -> None:
    """Cancels open orders that might conflict with an existing position's management."""
    pos_side = open_position.get("side") # 'long' or 'short'
    current_sl_price = open_position.get("stopLossPriceDecimal")
    current_tp_price = open_position.get("takeProfitPriceDecimal")

    try:
        open_orders = await api_client.fetch_open_orders(symbol)
        for order in open_orders:
            order_id = order.get("id")
            order_side = order.get("side", "").lower() # API 'buy' or 'sell'
            order_price_str = str(order.get("price", "0")) # Price of limit/stop order
            order_price = Decimal(order_price_str) if order_price_str and order_price_str != "0" else None
            order_type = order.get("type", "").lower() # 'limit', 'stop', 'market', etc.

            # Check if this order is likely the current position's TP
            is_likely_tp_order = False
            if current_tp_price and order_price and abs(order_price - current_tp_price) < min_tick_size * Decimal("0.5"):
                # TP for long is a sell order; TP for short is a buy order
                if (pos_side == "long" and order_side == "sell") or \
                   (pos_side == "short" and order_side == "buy"):
                    is_likely_tp_order = True
            
            # Check if this order is likely the current position's SL
            # This is harder as SL might be a stop_market order without an explicit price in `fetch_open_orders` easily comparable,
            # or it might be a conditional order whose trigger price needs to be checked.
            # This basic check assumes SL orders might appear with a price (e.g. stop_limit).
            is_likely_sl_order = False
            if current_sl_price and order_price and abs(order_price - current_sl_price) < min_tick_size * Decimal("0.5"):
                # SL for long is a sell order (stop); SL for short is a buy order (stop)
                if (pos_side == "long" and order_side == "sell" and "stop" in order_type) or \
                   (pos_side == "short" and order_side == "buy" and "stop" in order_type):
                    is_likely_sl_order = True
            
            # If the order opposes the *current position's side* AND is NOT its TP or SL, consider it conflicting.
            # Example: Position is LONG. A pending SELL limit order that is not the TP might be a remnant or error.
            is_conflicting = False
            if pos_side == "long" and order_side == "sell" and not is_likely_tp_order and not is_likely_sl_order:
                is_conflicting = True
            elif pos_side == "short" and order_side == "buy" and not is_likely_tp_order and not is_likely_sl_order:
                is_conflicting = True
            
            if is_conflicting:
                lg.warning(
                    f"{NEON_YELLOW}Canceling potentially conflicting open {order_side} order {order_id} "
                    f"(Price: {order_price}, Type: {order_type}) for existing {pos_side} position on {symbol}.{RESET_ALL_STYLE}"
                )
                await api_client.cancel_order(order_id, symbol)
            elif is_likely_tp_order or is_likely_sl_order:
                lg.debug(f"Keeping order {order_id} as it appears to be a TP/SL for current {symbol} position.")

    except Exception as cancel_err:
        lg.error(f"Error during conflicting order cancellation for {symbol} position: {cancel_err}", exc_info=True)


async def _handle_break_even_adjustment(
    api_client: BybitAPI,
    symbol: str,
    open_position: Dict[str, Any], # Can be updated by this function
    current_price: Decimal,
    current_atr: Optional[Decimal],
    market_data: Dict[str, Any],
    config: Dict[str, Any],
    lg: logging.Logger
) -> bool:
    """Adjusts SL to break-even if conditions are met. Returns True if SL updated."""
    pos_side = open_position.get("side")
    entry_price = open_position.get("entryPriceDecimal")
    min_tick_size = market_data["min_tick_size"]
    price_precision = market_data["price_precision"]
    
    lg.debug(f"--- Break-Even Check for {symbol} ---")
    try:
        if not (isinstance(current_atr, Decimal) and current_atr > Decimal(0) and \
                isinstance(min_tick_size, Decimal) and min_tick_size > Decimal(0)):
            raise ValueError(f"Invalid ATR ({current_atr}) or MinTickSize ({min_tick_size}) for BE calculation on {symbol}.")

        be_trigger_atr_multiple_str = str(config.get("break_even_trigger_atr_multiple", "1.0"))
        be_offset_ticks_val = int(config.get("break_even_offset_ticks", 2))
        be_trigger_atr_multiple = Decimal(be_trigger_atr_multiple_str)

        if be_trigger_atr_multiple <= Decimal(0) or be_offset_ticks_val < 0:
            raise ValueError(
                f"Invalid BE config for {symbol}: TriggerATRMultiple={be_trigger_atr_multiple}, OffsetTicks={be_offset_ticks_val}"
            )

        price_diff_from_entry = (current_price - entry_price) if pos_side == "long" else (entry_price - current_price)
        profit_in_atr = (price_diff_from_entry / current_atr) if current_atr > Decimal(0) else Decimal("Infinity")

        if profit_in_atr >= be_trigger_atr_multiple:
            # Calculate BE stop price
            tick_offset_value = min_tick_size * Decimal(be_offset_ticks_val)
            raw_be_price_candidate = entry_price + tick_offset_value if pos_side == "long" else entry_price - tick_offset_value
            
            # Round BE price to nearest tick, ensuring it's at or better than entry
            rounding_direction = ROUND_UP if pos_side == "long" else ROUND_DOWN # Favorable rounding for SL
            be_stop_price = (raw_be_price_candidate / min_tick_size).quantize(Decimal("1"), rounding=rounding_direction) * min_tick_size

            # Sanity check: BE price should not be worse than entry price after a profit move
            if (pos_side == "long" and be_stop_price < entry_price) or \
               (pos_side == "short" and be_stop_price > entry_price):
                be_stop_price = (entry_price / min_tick_size).quantize(Decimal("1"), rounding=rounding_direction) * min_tick_size
            
            if be_stop_price <= Decimal(0): # Should not happen with valid inputs
                raise ValueError(f"Calculated BE stop price for {symbol} is zero or negative: {be_stop_price}")

            current_sl_price = open_position.get("stopLossPriceDecimal")
            needs_sl_update = False
            if not current_sl_price: # No SL currently set
                needs_sl_update = True
            elif pos_side == "long" and be_stop_price > current_sl_price: # Move SL up to BE
                needs_sl_update = True
            elif pos_side == "short" and be_stop_price < current_sl_price: # Move SL down to BE
                needs_sl_update = True
            
            if needs_sl_update:
                lg.warning(
                    f"{NEON_YELLOW}*** Moving SL to Break-Even for {symbol} @ "
                    f"{_format_price_or_na(be_stop_price, price_precision)} (Profit ATR: {profit_in_atr:.2f} >= "
                    f"Trigger: {be_trigger_atr_multiple:.2f}) ***{RESET_ALL_STYLE}"
                )
                # Keep existing TP if any, otherwise TP remains None (or Decimal(0) to remove if API needs that)
                tp_for_be_update = open_position.get("takeProfitPriceDecimal")
                if await api_client.set_position_protection(
                    symbol, market_data["market_info"], open_position,
                    stop_loss_price=be_stop_price, take_profit_price=tp_for_be_update
                ):
                    lg.info(f"Break-Even SL update successful for {symbol}.")
                    open_position["stopLossPriceDecimal"] = be_stop_price # Update local cache of position
                    return True
                else:
                    lg.error(f"{NEON_RED}Break-Even SL update FAILED for {symbol}.{RESET_ALL_STYLE}")
        else:
            lg.debug(
                f"BE Target not reached for {symbol}. Profit ATR: {profit_in_atr:.2f} "
                f"(Current: {_format_price_or_na(current_price, price_precision)}, "
                f"Entry: {_format_price_or_na(entry_price, price_precision)}) "
                f"< Trigger Multiple: {be_trigger_atr_multiple:.2f}"
            )
    except Exception as be_err:
        lg.error(f"Break-Even check/adjustment error for {symbol}: {be_err}", exc_info=True)
    return False


async def _handle_trailing_stop_loss_activation(
    api_client: BybitAPI,
    symbol: str,
    open_position: Dict[str, Any],
    analysis_results: Dict[str, Any],
    config: Dict[str, Any],
    lg: logging.Logger,
    price_precision: int
) -> bool:
    """Activates exchange-managed TSL if configured and not already active. Returns True if TSL activation initiated."""
    lg.debug(f"--- TSL Activation Check for {symbol} (Exchange TSL not detected active) ---")
    try:
        analyzer: TradingAnalyzer = analysis_results["analyzer"]
        pos_side = open_position.get("side")
        entry_price = open_position.get("entryPriceDecimal")

        # Calculate TP target to associate with TSL if any (based on original entry logic)
        _, tsl_associated_tp_target, _ = analyzer.calculate_entry_tp_sl(entry_price, pos_side)

        lg.info(
            f"Attempting to activate Trailing Stop Loss for {symbol}. "
            f"Associated TP Target (if any): {_format_price_or_na(tsl_associated_tp_target, price_precision)}"
        )
        if await api_client.set_trailing_stop_loss(symbol, open_position, config, tsl_associated_tp_target):
            lg.info(f"Trailing Stop Loss activation successfully initiated for {symbol}.")
            return True
        else:
            lg.warning(f"{NEON_YELLOW}Trailing Stop Loss activation attempt failed for {symbol}.{RESET_ALL_STYLE}")
    except Exception as tsl_err:
        lg.error(f"TSL activation error for {symbol}: {tsl_err}", exc_info=True)
    return False


async def _manage_existing_open_position(
    api_client: BybitAPI,
    symbol: str,
    config: Dict[str, Any],
    lg: logging.Logger,
    market_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
    open_position: Dict[str, Any],
):
    """Manages an existing open position: checks exits, break-even, TSL, and cancels conflicting orders."""
    loop = asyncio.get_event_loop() # For time-based exit
    pos_side = open_position.get("side")  # 'long' or 'short'
    pos_size = open_position.get("contractsDecimal")
    entry_price = open_position.get("entryPriceDecimal")
    pos_timestamp_ms = open_position.get("timestamp_ms") # Timestamp of position opening/last significant update

    current_price = market_data["current_price_decimal"]
    price_precision = market_data["price_precision"]
    amount_precision = market_data["amount_precision"]
    min_tick_size = market_data["min_tick_size"]
    current_atr = analysis_results.get("current_atr")

    if not (pos_side in ["long", "short"] and \
            isinstance(pos_size, Decimal) and pos_size > Decimal(0) and \
            isinstance(entry_price, Decimal) and entry_price > Decimal(0)):
        lg.error(
            f"Cannot manage existing position for {symbol}: Invalid/incomplete position details. "
            f"Side={pos_side}, Size={pos_size}, EntryPx={entry_price}."
        )
        return

    # --- Log Current Position State ---
    lg.info(f"{NEON_BLUE}--- Managing Existing Position ({symbol}) ---{RESET_ALL_STYLE}")
    lg.info(
        f"  Side: {_format_side(pos_side)}, Size: {_format_price_or_na(pos_size, amount_precision)}, "
        f"Entry: {_format_price_or_na(entry_price, price_precision)}"
    )
    current_sl = open_position.get("stopLossPriceDecimal")
    current_tp = open_position.get("takeProfitPriceDecimal")
    lg.info(f"  Current SL: {_format_price_or_na(current_sl, price_precision)}, Current TP: {_format_price_or_na(current_tp, price_precision)}")

    # Check if TSL is active on the exchange (e.g., Bybit might return trailingStop, activePrice)
    # This depends on how api_client.get_open_position structures TSL info.
    # Assuming 'trailingStopDistanceDecimal' > 0 implies TSL active.
    tsl_distance_val = open_position.get("trailingStopDistanceDecimal")
    tsl_activation_price_val = open_position.get("trailingStopActivationPriceDecimal")
    is_tsl_active_on_exchange = bool(tsl_distance_val and tsl_distance_val > Decimal(0))
    lg.info(
        f"  Exchange TSL Status: Active={is_tsl_active_on_exchange}, "
        f"Distance/Value={_format_price_or_na(tsl_distance_val, price_precision)}, "
        f"ActivationPx={_format_price_or_na(tsl_activation_price_val, price_precision)}"
    )

    # --- Cancel Conflicting Open Orders ---
    await _cancel_conflicting_orders_for_active_position(api_client, symbol, open_position, min_tick_size, lg)

    # --- Check Exit Signal ---
    current_signal = analysis_results["signal"] # Current signal from analysis ("BUY", "SELL", "HOLD")
    if (pos_side == "long" and current_signal == "SELL") or \
       (pos_side == "short" and current_signal == "BUY"):
        lg.warning(
            f"{NEON_YELLOW}*** EXIT Signal ({_format_signal(current_signal)}) opposes current "
            f"{_format_side(pos_side)} position for {symbol}. Initiating close... ***{RESET_ALL_STYLE}"
        )
        if await _execute_close_position(api_client, symbol, market_data["market_info"], open_position, config, lg, "opposing signal"):
            return  # Position closed, no further management needed in this cycle

    # --- Time-Based Exit ---
    time_exit_minutes_config = analysis_results.get("time_exit_minutes_config")
    if isinstance(time_exit_minutes_config, (int, float)) and time_exit_minutes_config > 0 and pos_timestamp_ms:
        try:
            current_time_ms = int(loop.time() * 1000)
            elapsed_minutes = (current_time_ms - pos_timestamp_ms) / 60000.0
            if elapsed_minutes >= time_exit_minutes_config:
                lg.warning(
                    f"{NEON_YELLOW}*** TIME EXIT for {symbol} "
                    f"(Elapsed: {elapsed_minutes:.1f}m >= Configured: {time_exit_minutes_config}m). "
                    f"Initiating close... ***{RESET_ALL_STYLE}"
                )
                if await _execute_close_position(api_client, symbol, market_data["market_info"], open_position, config, lg, "time exit"):
                    return  # Position closed
        except Exception as terr:
            lg.error(f"Time-based exit check error for {symbol}: {terr}", exc_info=True)

    # --- Break-Even Adjustment ---
    # Only manage BE if exchange TSL is not active, as they might conflict or TSL might handle it.
    be_enabled_config = analysis_results.get("be_enabled_config", False)
    if be_enabled_config and not is_tsl_active_on_exchange:
        await _handle_break_even_adjustment(
            api_client, symbol, open_position, current_price, current_atr, market_data, config, lg
        )
        # Note: open_position might be updated by _handle_break_even_adjustment if SL changes

    # --- Trailing Stop Loss (TSL) Activation ---
    # This logic attempts to activate TSL if configured and not already detected as active on the exchange.
    # Assumes that once activated, the exchange manages TSL adjustments.
    tsl_enabled_config = analysis_results.get("tsl_enabled_config", False)
    if tsl_enabled_config and not is_tsl_active_on_exchange:
        # It's possible BE has already moved SL. TSL activation might still be desired.
        # For Bybit, setting TSL might override or work with an existing fixed SL.
        await _handle_trailing_stop_loss_activation(
            api_client, symbol, open_position, analysis_results, config, lg, price_precision
        )
        # If TSL activated, on next cycle, is_tsl_active_on_exchange should reflect this.

    lg.info(f"{NEON_CYAN}--- End Position Management ({symbol}) ---{RESET_ALL_STYLE}")


# --- Main Analysis and Trading Function ---
async def analyze_and_trade_symbol(
    api_client: BybitAPI,
    symbol: str,
    config: Dict[str, Any],
    logger: logging.Logger, # Use the passed logger instance
    enable_trading: bool,
) -> None:
    """
    Main analysis and trading logic function for a single symbol per cycle.
    Fetches data, performs analysis, and executes trading decisions based on configuration.
    """
    lg = logger # Alias for convenience
    loop = asyncio.get_event_loop()
    cycle_start_time = loop.time()
    lg.debug(f"--- Start Cycle for Symbol: {symbol} ---")

    # Step 1: Fetch and Prepare Market Data
    market_data = await _fetch_and_prepare_market_data(api_client, symbol, config, lg)
    if not market_data:
        lg.info(f"{NEON_BLUE}--- Cycle End ({symbol}, Market Data Fetch Failed) ---{RESET_ALL_STYLE}\n")
        return

    # Step 2: Perform Trade Analysis
    analysis_results = _perform_trade_analysis(
        market_data["klines_df"],
        market_data["current_price_decimal"],
        market_data["orderbook_data"],
        config,
        market_data["market_info"], # Pass full market_info dictionary
        lg,
        market_data["price_precision"], # Pass price_precision explicitly
    )
    if not analysis_results:
        lg.info(f"{NEON_BLUE}--- Cycle End ({symbol}, Trade Analysis Failed) ---{RESET_ALL_STYLE}\n")
        return

    # Step 3: Trading Logic (if enabled)
    if not enable_trading:
        lg.info(f"{NEON_YELLOW}Trading is globally disabled. Skipping trade execution for {symbol}.{RESET_ALL_STYLE}")
        # Log intended action based on signal for dry-run observation
        signal = analysis_results.get("signal", "HOLD")
        if signal in ["BUY", "SELL"]:
             lg.info(f"Dry run: Signal for {symbol} is {_format_signal(signal)}. Would evaluate for entry/exit if trading enabled.")
        lg.info(f"{NEON_BLUE}--- Cycle End ({symbol}, Trading Disabled) ---{RESET_ALL_STYLE}\n")
        return

    try:
        # Check current position status for the symbol
        open_position = await api_client.get_open_position(symbol)

        if open_position is None:  # No open position for this symbol
            await _handle_no_open_position(api_client, symbol, config, lg, market_data, analysis_results)
        else:  # There is an existing open position for this symbol
            await _manage_existing_open_position(
                api_client, symbol, config, lg, market_data, analysis_results, open_position
            )
    except Exception as trade_exec_err:
        # This is a catch-all for unexpected errors during the trading execution phase (entry/management)
        lg.critical(
            f"{NEON_RED}!!! CRITICAL ERROR during trading logic execution for {symbol}: {trade_exec_err} !!!{RESET_ALL_STYLE}",
            exc_info=True,
        )
        # Consider tripping circuit breaker or other master error handling here
        # e.g., if hasattr(api_client, 'trip_circuit_breaker'):
        # api_client.trip_circuit_breaker(reason=f"Critical trading error on {symbol}: {trade_exec_err}")

    cycle_duration_seconds = loop.time() - cycle_start_time
    lg.info(f"{NEON_BLUE}---== Cycle End ({symbol}, Duration: {cycle_duration_seconds:.2f}s) ==---{RESET_ALL_STYLE}\n")

```