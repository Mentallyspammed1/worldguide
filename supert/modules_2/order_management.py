# File: order_management.py
import time
import traceback
from decimal import Decimal
from typing import Any

# Third-party Libraries
try:
    import ccxt
    from colorama import Fore, Back, Style
except ImportError:
    class DummyCCXTExchange: pass
    class DummyCCXT:
        Exchange = DummyCCXTExchange
        InsufficientFunds = Exception
        NetworkError = Exception
        ExchangeError = Exception
        OrderNotFound = Exception
    ccxt = DummyCCXT() # type: ignore[assignment]
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""
    Fore, Back, Style = DummyColor(), DummyColor(), DummyColor()

# Custom module imports
from logger_setup import logger
from config import CONFIG
from utils import (
    safe_decimal_conversion,
    format_order_id,
    format_price,
    format_amount,
    send_sms_alert,
)
from risk_calculator import calculate_position_size
from indicator_calculator import analyze_order_book


def get_current_position(exchange: ccxt.Exchange, symbol: str) -> dict[str, Any]:
    """Fetches current position details (Bybit V5 focus), returns Decimals."""
    default_pos: dict[str, Any] = {"side": CONFIG.pos_none, "qty": Decimal("0.0"), "entry_price": Decimal("0.0")}
    market_id = None
    market = None
    try:
        market = exchange.market(symbol)
        market_id = market["id"]
    except Exception as e:
        logger.error(f"{Fore.RED}Position Check: Failed to get market info for '{symbol}': {e}{Style.RESET_ALL}")
        return default_pos

    try:
        if not exchange.has.get("fetchPositions"):
            logger.warning(
                f"{Fore.YELLOW}Position Check: fetchPositions not supported by {exchange.id}.{Style.RESET_ALL}"
            )
            return default_pos

        params = (
            {"category": "linear"}
            if market.get("linear")
            else ({"category": "inverse"} if market.get("inverse") else {})
        )
        logger.debug(f"Position Check: Fetching positions for {symbol} (MarketID: {market_id}) with params: {params}")
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)
        active_pos = None
        for pos in fetched_positions:
            pos_info = pos.get("info", {})
            pos_market_id = pos_info.get("symbol")
            position_idx = pos_info.get("positionIdx", 0)
            pos_side_v5 = pos_info.get("side", "None")
            size_str = pos_info.get("size")

            if pos_market_id == market_id and position_idx == 0 and pos_side_v5 != "None":
                size = safe_decimal_conversion(size_str)
                if abs(size) > CONFIG.position_qty_epsilon:
                    active_pos = pos
                    break
        if active_pos:
            try:
                size = safe_decimal_conversion(active_pos.get("info", {}).get("size"))
                entry_price = safe_decimal_conversion(active_pos.get("info", {}).get("avgPrice"))
                side = CONFIG.pos_long if active_pos.get("info", {}).get("side") == "Buy" else CONFIG.pos_short
                logger.info(
                    f"{Fore.YELLOW}Position Check: Found ACTIVE {side} position: Qty={abs(size):.8f} @ Entry={entry_price:.4f}{Style.RESET_ALL}"
                )
                return {"side": side, "qty": abs(size), "entry_price": entry_price}
            except Exception as parse_err:
                logger.warning(
                    f"{Fore.YELLOW}Position Check: Error parsing active position data: {parse_err}. Data: {active_pos}{Style.RESET_ALL}"
                )
                return default_pos
        else:
            logger.info(f"Position Check: No active One-Way position found for {market_id}.")
            return default_pos
    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.warning(
            f"{Fore.YELLOW}Position Check: Error fetching positions for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        logger.debug(traceback.format_exc())
    return default_pos


def close_position(
    exchange: ccxt.Exchange, symbol: str, position_to_close: dict[str, Any], reason: str = "Signal"
) -> dict[str, Any] | None:
    """Closes the specified active position with re-validation, uses Decimal."""
    initial_side = position_to_close.get("side", CONFIG.pos_none)
    initial_qty = position_to_close.get("qty", Decimal("0.0"))
    market_base = symbol.split("/")[0]
    logger.info(
        f"{Fore.YELLOW}Close Position: Initiated for {symbol}. Reason: {reason}. Initial state: {initial_side} Qty={initial_qty:.8f}{Style.RESET_ALL}"
    )

    live_position = get_current_position(exchange, symbol)
    if live_position["side"] == CONFIG.pos_none:
        logger.warning(
            f"{Fore.YELLOW}Close Position: Re-validation shows NO active position for {symbol}. Aborting.{Style.RESET_ALL}"
        )
        if initial_side != CONFIG.pos_none:
            logger.warning(
                f"{Fore.YELLOW}Close Position: Discrepancy detected (was {initial_side}, now None).{Style.RESET_ALL}"
            )
        return None

    live_amount_to_close = live_position["qty"]
    live_position_side = live_position["side"]
    side_to_execute_close = CONFIG.side_sell if live_position_side == CONFIG.pos_long else CONFIG.side_buy

    try:
        amount_str = format_amount(exchange, symbol, live_amount_to_close)
        amount_float = float(amount_str)
        if amount_float <= float(CONFIG.position_qty_epsilon):
            logger.error(
                f"{Fore.RED}Close Position: Closing amount after precision is negligible ({amount_str}). Aborting.{Style.RESET_ALL}"
            )
            return None

        logger.warning(
            f"{Back.YELLOW}{Fore.BLACK}Close Position: Attempting to CLOSE {live_position_side} ({reason}): "
            f"Exec {side_to_execute_close.upper()} MARKET {amount_str} {symbol} (reduce_only=True)...{Style.RESET_ALL}"
        )
        params = {"reduceOnly": True}
        order = exchange.create_market_order(
            symbol=symbol, side=side_to_execute_close, amount=amount_float, params=params
        )

        fill_price = safe_decimal_conversion(order.get("average"))
        filled_qty = safe_decimal_conversion(order.get("filled"))
        cost = safe_decimal_conversion(order.get("cost"))
        order_id_short = format_order_id(order.get("id"))

        logger.success( # type: ignore[attr-defined]
            f"{Fore.GREEN}{Style.BRIGHT}Close Position: Order ({reason}) placed for {symbol}. "
            f"Filled: {filled_qty:.8f}/{amount_str}, AvgFill: {fill_price:.4f}, Cost: {cost:.2f} USDT. ID:...{order_id_short}{Style.RESET_ALL}"
        )
        send_sms_alert(
            f"[{market_base}] Closed {live_position_side} {amount_str} @ ~{fill_price:.4f} ({reason}). ID:...{order_id_short}"
        )
        return order
    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
        logger.error(
            f"{Fore.RED}Close Position ({reason}): Failed for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        err_str = str(e).lower()
        if isinstance(e, ccxt.ExchangeError) and (
            "order would not reduce position size" in err_str
            or "position is zero" in err_str
            or "position size is zero" in err_str
        ):
            logger.warning(
                f"{Fore.YELLOW}Close Position: Exchange indicates position already closed/closing. Assuming closed.{Style.RESET_ALL}"
            )
            return None
        send_sms_alert(f"[{market_base}] ERROR Closing ({reason}): {type(e).__name__}. Check logs.")
    return None


def wait_for_order_fill(
    exchange: ccxt.Exchange, order_id: str, symbol: str, timeout_seconds: int
) -> dict[str, Any] | None:
    """Waits for a specific order to be filled (status 'closed')."""
    start_time = time.time()
    logger.info(
        f"{Fore.CYAN}Waiting for order ...{format_order_id(order_id)} ({symbol}) fill (Timeout: {timeout_seconds}s)...{Style.RESET_ALL}"
    )
    while time.time() - start_time < timeout_seconds:
        try:
            order = exchange.fetch_order(order_id, symbol)
            status = order.get("status")
            logger.debug(f"Order ...{format_order_id(order_id)} status: {status}")
            if status == "closed":
                logger.success(f"{Fore.GREEN}Order ...{format_order_id(order_id)} confirmed FILLED.{Style.RESET_ALL}") # type: ignore[attr-defined]
                return order
            elif status in ["canceled", "rejected", "expired"]:
                logger.error(
                    f"{Fore.RED}Order ...{format_order_id(order_id)} failed with status '{status}'.{Style.RESET_ALL}"
                )
                return None
            time.sleep(0.5)
        except ccxt.OrderNotFound:
            logger.warning(
                f"{Fore.YELLOW}Order ...{format_order_id(order_id)} not found yet. Retrying...{Style.RESET_ALL}"
            )
            time.sleep(1)
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            logger.warning(
                f"{Fore.YELLOW}Error checking order ...{format_order_id(order_id)}: {type(e).__name__} - {e}. Retrying...{Style.RESET_ALL}"
            )
            time.sleep(1)
    logger.error(
        f"{Fore.RED}Order ...{format_order_id(order_id)} did not fill within {timeout_seconds}s timeout.{Style.RESET_ALL}"
    )
    return None


def place_risked_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    risk_percentage: Decimal,
    current_atr: Decimal | None,
    sl_atr_multiplier: Decimal,
    leverage: int,
    max_order_cap_usdt: Decimal,
    margin_check_buffer: Decimal,
    tsl_percent: Decimal,
    tsl_activation_offset_percent: Decimal,
) -> dict[str, Any] | None:
    """Places market entry, waits for fill, then places exchange-native fixed SL and TSL using Decimal."""
    market_base = symbol.split("/")[0]
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}Place Order: Initiating {side.upper()} for {symbol}...{Style.RESET_ALL}")
    if current_atr is None or current_atr <= Decimal("0"):
        logger.error(
            f"{Fore.RED}Place Order ({side.upper()}): Invalid ATR ({current_atr}). Cannot place order.{Style.RESET_ALL}"
        )
        return None

    entry_price_estimate: Decimal | None = None
    initial_sl_price_estimate: Decimal | None = None
    final_quantity: Decimal | None = None
    market_info: dict | None = None

    try:
        logger.debug("Fetching balance & market details...")
        balance = exchange.fetch_balance()
        market_info = exchange.market(symbol)
        limits = market_info.get("limits", {})
        amount_limits = limits.get("amount", {})
        price_limits = limits.get("price", {})
        min_qty = safe_decimal_conversion(amount_limits.get("min")) if amount_limits.get("min") else None
        max_qty = safe_decimal_conversion(amount_limits.get("max")) if amount_limits.get("max") else None
        min_price = safe_decimal_conversion(price_limits.get("min")) if price_limits.get("min") else None

        usdt_balance = balance.get(CONFIG.usdt_symbol, {})
        usdt_total = safe_decimal_conversion(usdt_balance.get("total"))
        usdt_free = safe_decimal_conversion(usdt_balance.get("free"))
        usdt_equity = usdt_total if usdt_total > CONFIG.position_qty_epsilon else usdt_free

        if usdt_equity <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Place Order ({side.upper()}): Zero/Invalid equity ({usdt_equity:.4f}).{Style.RESET_ALL}")
            return None
        logger.debug(f"Equity={usdt_equity:.4f}, Free={usdt_free:.4f} USDT")

        ob_data = analyze_order_book(exchange, symbol, CONFIG.shallow_ob_fetch_depth, CONFIG.shallow_ob_fetch_depth)
        best_ask, best_bid = ob_data.get("best_ask"), ob_data.get("best_bid")
        if side == CONFIG.side_buy and best_ask: entry_price_estimate = best_ask
        elif side == CONFIG.side_sell and best_bid: entry_price_estimate = best_bid
        else:
            try: entry_price_estimate = safe_decimal_conversion(exchange.fetch_ticker(symbol).get("last"))
            except Exception as e: logger.error(f"{Fore.RED}Failed to fetch ticker price: {e}{Style.RESET_ALL}"); return None
        if not entry_price_estimate or entry_price_estimate <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Invalid entry price estimate ({entry_price_estimate}).{Style.RESET_ALL}"); return None
        logger.debug(f"Estimated Entry Price ~ {entry_price_estimate:.4f}")

        sl_distance = current_atr * sl_atr_multiplier
        initial_sl_price_raw = (entry_price_estimate - sl_distance) if side == CONFIG.side_buy else (entry_price_estimate + sl_distance)
        if min_price is not None and initial_sl_price_raw < min_price: initial_sl_price_raw = min_price
        if initial_sl_price_raw <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Invalid Initial SL price calc: {initial_sl_price_raw:.4f}{Style.RESET_ALL}"); return None
        initial_sl_price_estimate = safe_decimal_conversion(format_price(exchange, symbol, initial_sl_price_raw))
        logger.info(f"Calculated Initial SL Price (Estimate) ~ {initial_sl_price_estimate:.4f} (Dist: {sl_distance:.4f})")

        calc_qty, req_margin = calculate_position_size(
            exchange, symbol, usdt_equity, risk_percentage, entry_price_estimate, initial_sl_price_estimate, leverage
        )
        if calc_qty is None or req_margin is None: logger.error(f"{Fore.RED}Failed risk calculation.{Style.RESET_ALL}"); return None
        final_quantity = calc_qty

        pos_value = final_quantity * entry_price_estimate
        if pos_value > max_order_cap_usdt:
            logger.warning(f"{Fore.YELLOW}Order value {pos_value:.4f} > Cap {max_order_cap_usdt:.4f}. Capping qty.{Style.RESET_ALL}")
            final_quantity = max_order_cap_usdt / entry_price_estimate
            final_quantity = safe_decimal_conversion(format_amount(exchange, symbol, final_quantity))
            # req_margin = max_order_cap_usdt / Decimal(leverage) # Margin will be recalculated based on final_quantity

        if final_quantity <= CONFIG.position_qty_epsilon: logger.error(f"{Fore.RED}Final Qty negligible: {final_quantity:.8f}{Style.RESET_ALL}"); return None
        if min_qty is not None and final_quantity < min_qty: logger.error(f"{Fore.RED}Final Qty {final_quantity:.8f} < Min {min_qty}{Style.RESET_ALL}"); return None
        if max_qty is not None and final_quantity > max_qty:
            logger.warning(f"{Fore.YELLOW}Final Qty {final_quantity:.8f} > Max {max_qty}. Capping.{Style.RESET_ALL}")
            final_quantity = max_qty
            final_quantity = safe_decimal_conversion(format_amount(exchange, symbol, final_quantity))

        final_req_margin = (final_quantity * entry_price_estimate) / Decimal(leverage)
        req_margin_buffered = final_req_margin * margin_check_buffer
        if usdt_free < req_margin_buffered:
            logger.error(f"{Fore.RED}Insufficient FREE margin. Need ~{req_margin_buffered:.4f}, Have {usdt_free:.4f}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Insufficient Free Margin"); return None
        logger.info(f"{Fore.GREEN}Final Order: Qty={final_quantity:.8f}, EstValue={final_quantity * entry_price_estimate:.4f}, EstMargin={final_req_margin:.4f}. Margin check OK.{Style.RESET_ALL}")

        entry_order: dict[str, Any] | None = None
        order_id: str | None = None
        try:
            qty_float = float(final_quantity)
            logger.warning(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** Placing {side.upper()} MARKET ENTRY: {qty_float:.8f} {symbol} ***{Style.RESET_ALL}")
            entry_order = exchange.create_market_order(symbol=symbol, side=side, amount=qty_float, params={"reduce_only": False})
            order_id = entry_order.get("id")
            if not order_id: logger.error(f"{Fore.RED}Entry order placed but no ID returned! Response: {entry_order}{Style.RESET_ALL}"); return None
            logger.success(f"{Fore.GREEN}Market Entry Order submitted. ID: ...{format_order_id(order_id)}. Waiting for fill...{Style.RESET_ALL}") # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED TO PLACE ENTRY ORDER: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry placement failed: {type(e).__name__}"); return None

        filled_entry = wait_for_order_fill(exchange, order_id, symbol, CONFIG.order_fill_timeout_seconds)
        if not filled_entry:
            logger.error(f"{Fore.RED}Entry order ...{format_order_id(order_id)} did not fill/failed.{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Entry ...{format_order_id(order_id)} fill timeout/fail.")
            try: logger.info(f"{Fore.CYAN}Attempting cancellation of potentially stuck order ...{format_order_id(order_id)}.{Style.RESET_ALL}"); exchange.cancel_order(order_id, symbol)
            except Exception as cancel_e: logger.warning(f"{Fore.YELLOW}Failed to cancel potentially stuck order ...{format_order_id(order_id)}: {cancel_e}{Style.RESET_ALL}")
            return None

        avg_fill_price = safe_decimal_conversion(filled_entry.get("average"))
        filled_qty = safe_decimal_conversion(filled_entry.get("filled"))
        cost = safe_decimal_conversion(filled_entry.get("cost"))
        if avg_fill_price <= CONFIG.position_qty_epsilon or filled_qty <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Invalid fill details for ...{format_order_id(order_id)}: Price={avg_fill_price}, Qty={filled_qty}{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Invalid fill details ...{format_order_id(order_id)}."); return filled_entry
        logger.success(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}ENTRY CONFIRMED: ...{format_order_id(order_id)}. Filled: {filled_qty:.8f} @ {avg_fill_price:.4f}, Cost: {cost:.4f} USDT{Style.RESET_ALL}") # type: ignore[attr-defined]

        sl_distance = current_atr * sl_atr_multiplier # current_atr is not None due to earlier check
        actual_sl_price_raw = (avg_fill_price - sl_distance) if side == CONFIG.side_buy else (avg_fill_price + sl_distance)
        if min_price is not None and actual_sl_price_raw < min_price: actual_sl_price_raw = min_price
        if actual_sl_price_raw <= CONFIG.position_qty_epsilon:
            logger.error(f"{Fore.RED}Invalid ACTUAL SL price calc based on fill: {actual_sl_price_raw:.4f}. Cannot place SL!{Style.RESET_ALL}")
            send_sms_alert(f"[{market_base}] CRITICAL ({side.upper()}): Invalid ACTUAL SL price! Attempting emergency close.")
            close_position(exchange, symbol, {"side": side, "qty": filled_qty}, reason="Invalid SL Calc"); return filled_entry
        actual_sl_price_str = format_price(exchange, symbol, actual_sl_price_raw)
        actual_sl_price_float = float(actual_sl_price_str)

        sl_order_id = "N/A"
        try:
            sl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            sl_qty_str = format_amount(exchange, symbol, filled_qty)
            sl_qty_float = float(sl_qty_str)
            logger.info(f"{Fore.CYAN}Placing Initial Fixed SL ({sl_atr_multiplier}*ATR)... Side: {sl_side.upper()}, Qty: {sl_qty_float:.8f}, StopPx: {actual_sl_price_str}{Style.RESET_ALL}")
            sl_params = {"stopPrice": actual_sl_price_float, "reduceOnly": True}
            sl_order = exchange.create_order(symbol, "stopMarket", sl_side, sl_qty_float, params=sl_params)
            sl_order_id = format_order_id(sl_order.get("id"))
            logger.success(f"{Fore.GREEN}Initial Fixed SL order placed. ID: ...{sl_order_id}, Trigger: {actual_sl_price_str}{Style.RESET_ALL}") # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED to place Initial Fixed SL order: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed initial SL placement: {type(e).__name__}")

        tsl_order_id, tsl_act_price_str = "N/A", "N/A"
        try:
            act_offset = avg_fill_price * tsl_activation_offset_percent
            act_price_raw = (avg_fill_price + act_offset) if side == CONFIG.side_buy else (avg_fill_price - act_offset)
            if min_price is not None and act_price_raw < min_price: act_price_raw = min_price
            if act_price_raw <= CONFIG.position_qty_epsilon: raise ValueError(f"Invalid TSL activation price {act_price_raw:.4f}")
            tsl_act_price_str = format_price(exchange, symbol, act_price_raw)
            tsl_act_price_float = float(tsl_act_price_str)
            tsl_side = CONFIG.side_sell if side == CONFIG.side_buy else CONFIG.side_buy
            tsl_trail_value_str = str(tsl_percent * Decimal("100"))
            tsl_qty_str = format_amount(exchange, symbol, filled_qty)
            tsl_qty_float = float(tsl_qty_str)
            logger.info(f"{Fore.CYAN}Placing Trailing SL ({tsl_percent:.2%})... Side: {tsl_side.upper()}, Qty: {tsl_qty_float:.8f}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}")
            tsl_params = {"trailingStop": tsl_trail_value_str, "activePrice": tsl_act_price_float, "reduceOnly": True}
            tsl_order = exchange.create_order(symbol, "stopMarket", tsl_side, tsl_qty_float, params=tsl_params)
            tsl_order_id = format_order_id(tsl_order.get("id"))
            logger.success(f"{Fore.GREEN}Trailing SL order placed. ID: ...{tsl_order_id}, Trail%: {tsl_trail_value_str}, ActPx: {tsl_act_price_str}{Style.RESET_ALL}") # type: ignore[attr-defined]
            sms_msg = (f"[{market_base}] ENTERED {side.upper()} {filled_qty:.8f} @ {avg_fill_price:.4f}. "
                       f"Init SL ~{actual_sl_price_str}. TSL {tsl_percent:.2%} act@{tsl_act_price_str}. "
                       f"IDs E:...{format_order_id(order_id)}, SL:...{sl_order_id}, TSL:...{tsl_order_id}")
            send_sms_alert(sms_msg)
        except Exception as e:
            logger.error(f"{Fore.RED}{Style.BRIGHT}FAILED to place Trailing SL order: {e}{Style.RESET_ALL}"); logger.debug(traceback.format_exc())
            send_sms_alert(f"[{market_base}] ERROR ({side.upper()}): Failed TSL placement: {type(e).__name__}")
        return filled_entry
    except (ccxt.InsufficientFunds, ccxt.NetworkError, ccxt.ExchangeError, ValueError, Exception) as e:
        logger.error(f"{Fore.RED}{Style.BRIGHT}Place Order ({side.upper()}): Overall process failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()}): Overall process failed: {type(e).__name__}")
    return None


def cancel_open_orders(exchange: ccxt.Exchange, symbol: str, reason: str = "Cleanup") -> None:
    """Attempts to cancel all open orders for the specified symbol."""
    logger.info(f"{Fore.CYAN}Order Cancel: Attempting for {symbol} (Reason: {reason})...{Style.RESET_ALL}")
    try:
        if not exchange.has.get("fetchOpenOrders"):
            logger.warning(f"{Fore.YELLOW}Order Cancel: fetchOpenOrders not supported.{Style.RESET_ALL}"); return
        open_orders = exchange.fetch_open_orders(symbol)
        if not open_orders: logger.info(f"{Fore.CYAN}Order Cancel: No open orders found for {symbol}.{Style.RESET_ALL}"); return

        logger.warning(f"{Fore.YELLOW}Order Cancel: Found {len(open_orders)} open orders for {symbol}. Cancelling...{Style.RESET_ALL}")
        cancelled_count, failed_count = 0, 0
        for order in open_orders:
            order_id, order_info = order.get("id"), f"...{format_order_id(order.get('id'))} ({order.get('type')} {order.get('side')})"
            if order_id:
                try:
                    exchange.cancel_order(order_id, symbol); logger.info(f"{Fore.CYAN}Order Cancel: Success for {order_info}{Style.RESET_ALL}"); cancelled_count += 1; time.sleep(0.1)
                except ccxt.OrderNotFound: logger.warning(f"{Fore.YELLOW}Order Cancel: Not found (already closed/cancelled?): {order_info}{Style.RESET_ALL}"); cancelled_count += 1
                except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e: logger.error(f"{Fore.RED}Order Cancel: FAILED for {order_info}: {type(e).__name__} - {e}{Style.RESET_ALL}"); failed_count += 1
        logger.info(f"{Fore.CYAN}Order Cancel: Finished. Cancelled: {cancelled_count}, Failed: {failed_count}.{Style.RESET_ALL}")
        if failed_count > 0: send_sms_alert(f"[{symbol.split('/')[0]}] WARNING: Failed to cancel {failed_count} orders during {reason}.")
    except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        logger.error(f"{Fore.RED}Order Cancel: Failed fetching open orders for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")

# End of order_management.py
```

```python
