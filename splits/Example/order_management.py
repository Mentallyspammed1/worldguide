# File: order_management.py
# -*- coding: utf-8 -*-

"""
Functions for Placing, Cancelling, Fetching, and Updating Orders on Bybit V5
"""

import logging
import sys
import time
from decimal import Decimal
from typing import Optional, Dict, List, Literal, Union

try:
    import ccxt
except ImportError:
    print("Error: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)
try:
    from colorama import Fore, Style, Back
except ImportError:
    print("Warning: colorama library not found. Logs will not be colored. Install: pip install colorama")

    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()

from config import Config
from utils import (
    retry_api_call,
    _get_v5_category,
    safe_decimal_conversion,
    format_price,
    format_amount,
    format_order_id,
    send_sms_alert,
)

# Import necessary function from market_data module for slippage check
from market_data import fetch_l2_order_book_validated

logger = logging.getLogger(__name__)


# Snippet 4 / Function 4: Place Market Order with Slippage Check
@retry_api_call(max_retries=1, initial_delay=0)  # Market orders usually not retried automatically
def place_market_order_slippage_check(
    exchange: ccxt.bybit,
    symbol: str,
    side: Literal["buy", "sell"],
    amount: Decimal,
    config: Config,
    max_slippage_pct: Optional[Decimal] = None,
    is_reduce_only: bool = False,
    client_order_id: Optional[str] = None,
) -> Optional[Dict]:
    """
    Places a market order on Bybit V5 after checking the current spread against a slippage threshold.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        side: 'buy' or 'sell'.
        amount: Order quantity in base currency (Decimal). Must be positive.
        config: Configuration object.
        max_slippage_pct: Maximum allowed spread percentage (e.g., Decimal('0.005') for 0.5%).
                          Uses `config.DEFAULT_SLIPPAGE_PCT` if None.
        is_reduce_only: If True, set the reduceOnly flag.
        client_order_id: Optional client order ID string (max length 36 for Bybit V5 linear).

    Returns:
        The order dictionary returned by ccxt, or None if the order failed or was aborted pre-flight.
    """
    func_name = "place_market_order_slippage_check"
    market_base = symbol.split("/")[0]
    action = "CLOSE" if is_reduce_only else "ENTRY"
    log_prefix = f"[Market Order {action} {side.upper()}]"
    effective_max_slippage = max_slippage_pct if max_slippage_pct is not None else config.DEFAULT_SLIPPAGE_PCT

    logger.info(
        f"{Fore.BLUE}{log_prefix} Init for {format_amount(exchange, symbol, amount)} {symbol}. "
        f"Max Slippage: {effective_max_slippage:.4%}, ReduceOnly: {is_reduce_only}, Coid: {client_order_id or 'None'}{Style.RESET_ALL}"
    )

    if amount <= config.POSITION_QTY_EPSILON:
        logger.error(f"{Fore.RED}{log_prefix} Amount is zero or negative ({amount}). Aborting.{Style.RESET_ALL}")
        return None

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}[{func_name}] Cannot determine market category for {symbol}. Aborting market order.{Style.RESET_ALL}"
            )
            return None

        # 1. Perform Slippage Check using validated L2 OB fetch
        logger.debug(f"[{func_name}] Performing pre-order slippage check (Depth: {config.SHALLOW_OB_FETCH_DEPTH})...")
        # Note: This adds API call latency before placing the order. Consider the trade-off.
        # Use the imported function from market_data module
        ob_data = fetch_l2_order_book_validated(exchange, symbol, config.SHALLOW_OB_FETCH_DEPTH, config)

        if ob_data and ob_data.get("bids") and ob_data.get("asks"):
            # Ensure lists are not empty before accessing index 0
            best_bid = ob_data["bids"][0][0]
            best_ask = ob_data["asks"][0][0]
            spread = (best_ask - best_bid) / best_bid if best_bid > Decimal("0") else Decimal("inf")

            logger.debug(
                f"[{func_name}] Current shallow OB: Best Bid={format_price(exchange, symbol, best_bid)}, "
                f"Best Ask={format_price(exchange, symbol, best_ask)}, Spread={spread:.4%}"
            )

            if spread > effective_max_slippage:
                logger.error(
                    f"{Fore.RED}{log_prefix} ABORTED due to high slippage. "
                    f"Current Spread {spread:.4%} > Max Allowed {effective_max_slippage:.4%}.{Style.RESET_ALL}"
                )
                send_sms_alert(
                    f"[{market_base}] ORDER ABORT ({side.upper()} {action}): High Slippage {spread:.4%}", config
                )
                return None
        else:
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} Could not get valid L2 order book data to check slippage. Proceeding with caution.{Style.RESET_ALL}"
            )
            # Consider if aborting is necessary when slippage check fails
            # return None # Uncomment to abort if check fails

        # 2. Prepare and Place Order
        amount_str = format_amount(exchange, symbol, amount)  # Use precision formatting
        amount_float = float(amount_str)  # CCXT generally expects float amount

        params: Dict[str, Any] = {"category": category}
        if is_reduce_only:
            params["reduceOnly"] = True
        if client_order_id:
            # Bybit V5 linear/inverse clientOrderId max length is 36
            valid_coid = client_order_id[:36]
            params["clientOrderId"] = valid_coid
            if len(valid_coid) < len(client_order_id):
                logger.warning(
                    f"[{func_name}] Client Order ID '{client_order_id}' truncated to 36 chars: '{valid_coid}'"
                )

        # Log prominently before placing
        bg = Back.GREEN if side == config.SIDE_BUY else Back.RED
        fg = Fore.BLACK  # High contrast text
        logger.warning(
            f"{bg}{fg}{Style.BRIGHT}*** PLACING MARKET {side.upper()} {'REDUCE' if is_reduce_only else 'ENTRY'}: "
            f"{amount_str} {symbol} (Params: {params}) ***{Style.RESET_ALL}"
        )

        # Place the market order
        order = exchange.create_market_order(symbol, side, amount_float, params=params)

        # 3. Log Result
        order_id = order.get("id")
        # Check response for client OID first, then fallback to submitted param
        client_oid_resp = order.get("clientOrderId", params.get("clientOrderId", "N/A"))
        status = order.get("status", "?")  # e.g., 'open', 'closed', 'canceled'
        filled_qty = safe_decimal_conversion(order.get("filled", "0.0"))
        avg_price = safe_decimal_conversion(order.get("average"))  # May be None initially

        # Use logger.info for success
        logger.info(
            f"{Fore.GREEN}{log_prefix} Market order submitted successfully. "
            f"ID: ...{format_order_id(order_id)}, ClientOID: {client_oid_resp}, Status: {status}, "
            f"Target Qty: {amount_str}, Filled Qty: {format_amount(exchange, symbol, filled_qty)}, "
            f"Avg Price: {format_price(exchange, symbol, avg_price)}{Style.RESET_ALL}"
        )
        return order

    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError) as e:
        # Log specific CCXT exceptions clearly
        logger.error(
            f"{Fore.RED}{log_prefix} API Error placing market order: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()} {action}): {type(e).__name__}", config)
        # Don't raise here as market orders are typically not retried by the decorator
        return None
    except Exception as e:
        logger.critical(
            f"{Back.RED}[{func_name}] Unexpected critical error placing market order: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()} {action}): Unexpected {type(e).__name__}.", config)
        return None


# Snippet 5 / Function 5: Cancel All Open Orders
@retry_api_call(max_retries=2, initial_delay=1.0)  # Decorator for fetch/network issues
def cancel_all_orders(exchange: ccxt.bybit, symbol: str, config: Config, reason: str = "Cleanup") -> bool:
    """
    Cancels all open orders for a specific symbol on Bybit V5.

    Fetches open orders first (defaults to regular 'Order' filter, consider 'StopOrder')
    and attempts to cancel each one individually. Handles `OrderNotFound` gracefully.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol for which to cancel orders.
        config: Configuration object.
        reason: A short string indicating the reason for cancellation (for logging).

    Returns:
        True if all found open orders were successfully cancelled or confirmed gone,
        False if any cancellation failed (excluding OrderNotFound).

    Raises:
        Reraises CCXT exceptions from fetch_open_orders for the retry decorator.
        Does not re-raise OrderNotFound during cancellation.
    """
    func_name = "cancel_all_orders"
    market_base = symbol.split("/")[0]
    log_prefix = f"[Cancel All ({reason})]"
    logger.info(f"{Fore.CYAN}{log_prefix} Attempting for {symbol}...{Style.RESET_ALL}")

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}[{func_name}] Cannot determine market category for {symbol}. Aborting cancel all.{Style.RESET_ALL}"
            )
            return False

        # V5: Fetch open regular orders.
        # To cancel stops, you might need a separate call with orderFilter='StopOrder' or 'tpslOrder'
        # Or use the unified cancel_all_orders endpoint if available and suitable.
        # For now, focuses on regular limit/market orders pending execution.
        fetch_params = {"category": category, "orderFilter": "Order"}  # Default filter
        logger.debug(f"[{func_name}] Fetching open regular orders for {symbol} with params: {fetch_params}")
        open_orders = exchange.fetch_open_orders(symbol, params=fetch_params)

        if not open_orders:
            logger.info(
                f"{Fore.CYAN}{log_prefix} No open regular orders found for {symbol} to cancel.{Style.RESET_ALL}"
            )
            # Consider also checking/cancelling stop orders here if required
            # open_stop_orders = fetch_open_orders_filtered(exchange, symbol, config, order_filter='StopOrder')
            # if open_stop_orders: ... cancel them ...
            return True

        logger.warning(
            f"{Fore.YELLOW}{log_prefix} Found {len(open_orders)} open regular order(s) for {symbol}. Attempting cancellation...{Style.RESET_ALL}"
        )

        success_count = 0
        fail_count = 0
        # Calculate a small delay based on rate limit to avoid hammering
        cancel_delay = max(
            0.05, 1.0 / (exchange.rateLimit if exchange.rateLimit and exchange.rateLimit > 0 else 20)
        )  # e.g., 50ms

        # V5 requires category for cancellation as well
        cancel_params = {"category": category}

        for order in open_orders:
            order_id = order.get("id")
            order_info_log = (
                f"ID: ...{format_order_id(order_id)} "
                f"({order.get('type', '?').upper()} {order.get('side', '?').upper()} "
                f"Amt: {format_amount(exchange, symbol, order.get('amount'))})"
            )

            if not order_id:
                logger.warning(f"[{func_name}] Skipping order with missing ID in fetched data: {order}")
                continue

            try:
                logger.debug(f"[{func_name}] Cancelling order {order_info_log} with params: {cancel_params}")
                # Use cancel_order for individual cancellation
                exchange.cancel_order(order_id, symbol, params=cancel_params)
                logger.info(f"{Fore.CYAN}{log_prefix} Successfully cancelled order {order_info_log}{Style.RESET_ALL}")
                success_count += 1
            except ccxt.OrderNotFound:
                # This is not an error in the context of 'cancel all' - the order is already gone.
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} Order {order_info_log} already cancelled or filled (Not Found). Considered OK.{Style.RESET_ALL}"
                )
                success_count += 1  # Treat as success
            except (ccxt.NetworkError, ccxt.RateLimitExceeded) as e_cancel:
                # Log transient errors but continue the loop. The outer retry might handle it.
                logger.warning(
                    f"{Fore.YELLOW}{log_prefix} Network/RateLimit error cancelling {order_info_log}: {e_cancel}. Loop continues...{Style.RESET_ALL}"
                )
                fail_count += 1  # Count as failure for this attempt
            except ccxt.ExchangeError as e_cancel:
                # Log persistent exchange errors
                logger.error(
                    f"{Fore.RED}{log_prefix} FAILED to cancel order {order_info_log}: {e_cancel}{Style.RESET_ALL}"
                )
                fail_count += 1
            except Exception as e_cancel:
                logger.error(
                    f"{Fore.RED}{log_prefix} Unexpected error cancelling order {order_info_log}: {e_cancel}{Style.RESET_ALL}",
                    exc_info=True,
                )
                fail_count += 1

            time.sleep(cancel_delay)  # Pause between cancellation attempts

        # Report Summary after attempting all
        total_attempted = len(open_orders)
        if fail_count > 0:
            # If failures occurred, re-fetch to see if they were transient or persistent
            try:
                logger.warning(f"[{func_name}] Re-checking open orders after {fail_count} cancellation failure(s)...")
                remaining_orders = exchange.fetch_open_orders(symbol, params=fetch_params)
                if not remaining_orders:
                    logger.warning(
                        f"{Fore.YELLOW}[{func_name}] {log_prefix}: Initial cancellation reported {fail_count} failures, but re-check shows no open orders remain. Likely transient errors.{Style.RESET_ALL}"
                    )
                    return True  # All gone now
                else:
                    logger.error(
                        f"{Fore.RED}[{func_name}] {log_prefix}: Finished cancellation attempt for {symbol}. "
                        f"Failed: {fail_count}, Success/Gone: {success_count}. {len(remaining_orders)} order(s) might still remain.{Style.RESET_ALL}"
                    )
                    send_sms_alert(
                        f"[{market_base}] ERROR: Failed to cancel {fail_count} orders ({reason}). Check logs.", config
                    )
                    return False
            except Exception as e_recheck:
                logger.error(
                    f"{Fore.RED}[{func_name}] Error re-checking orders after failures: {e_recheck}. Assuming failures persist."
                )
                send_sms_alert(
                    f"[{market_base}] ERROR: Failed to cancel {fail_count} orders ({reason}). Check logs.", config
                )
                return False
        else:
            # Use logger.info for success
            logger.info(
                f"{Fore.GREEN}[{func_name}] {log_prefix}: Successfully cancelled or confirmed gone "
                f"all {total_attempted} open regular orders found for {symbol}.{Style.RESET_ALL}"
            )
            return True

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        # Errors during the initial fetch_open_orders
        logger.error(
            f"{Fore.RED}[{func_name}] {log_prefix}: API error during 'cancel all' setup/fetch for {symbol}: {e}{Style.RESET_ALL}"
        )
        raise  # Re-raise for retry decorator
    except Exception as e:
        logger.error(
            f"{Fore.RED}[{func_name}] {log_prefix}: Unexpected error during 'cancel all' for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return False


# Snippet 7 / Function 7: Place Limit Order with TIF/Flags
@retry_api_call(max_retries=1, initial_delay=0)  # Limit orders usually not retried automatically
def place_limit_order_tif(
    exchange: ccxt.bybit,
    symbol: str,
    side: Literal["buy", "sell"],
    amount: Decimal,
    price: Decimal,
    config: Config,
    time_in_force: str = "GTC",
    is_reduce_only: bool = False,
    is_post_only: bool = False,
    client_order_id: Optional[str] = None,
) -> Optional[Dict]:
    """
    Places a limit order on Bybit V5 with options for Time-In-Force, Post-Only, and Reduce-Only.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        side: 'buy' or 'sell'.
        amount: Order quantity in base currency (Decimal). Must be positive.
        price: Limit price for the order (Decimal). Must be positive.
        config: Configuration object.
        time_in_force: Time-In-Force policy ('GTC', 'IOC', 'FOK'). CCXT standard.
                       Bybit V5 also supports 'PostOnly' as a separate flag.
        is_reduce_only: If True, set the reduceOnly flag.
        is_post_only: If True, ensures the order is only accepted if it does not immediately match.
        client_order_id: Optional client order ID string (max length 36 for Bybit V5 linear).

    Returns:
        The order dictionary returned by ccxt, or None if the order placement failed.
    """
    func_name = "place_limit_order_tif"
    log_prefix = f"[Limit Order {side.upper()}]"
    logger.info(
        f"{Fore.BLUE}{log_prefix} Init: {format_amount(exchange, symbol, amount)} {symbol} @ {format_price(exchange, symbol, price)} "
        f"(TIF:{time_in_force}, Reduce:{is_reduce_only}, Post:{is_post_only}, Coid:{client_order_id or 'None'})...{Style.RESET_ALL}"
    )

    # Validate inputs
    if amount <= config.POSITION_QTY_EPSILON:
        logger.error(f"{Fore.RED}{log_prefix} Invalid amount: {amount}. Must be positive.{Style.RESET_ALL}")
        return None
    if price <= Decimal("0"):
        logger.error(f"{Fore.RED}{log_prefix} Invalid price: {price}. Must be positive.{Style.RESET_ALL}")
        return None

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}[{func_name}] Cannot determine market category for {symbol}. Aborting limit order.{Style.RESET_ALL}"
            )
            return None

        # Format amount and price strings using market precision
        amount_str = format_amount(exchange, symbol, amount)
        price_str = format_price(exchange, symbol, price)
        # Convert to float for CCXT create_limit_order call
        amount_float = float(amount_str)
        price_float = float(price_str)

        # Prepare V5 parameters
        params: Dict[str, Any] = {"category": category}

        # Handle Time In Force
        valid_tif = ["GTC", "IOC", "FOK"]  # Standard CCXT TIFs supported by Bybit V5
        tif_upper = time_in_force.upper()
        if tif_upper in valid_tif:
            params["timeInForce"] = tif_upper
        else:
            logger.warning(f"[{func_name}] Unsupported TimeInForce '{time_in_force}' specified. Defaulting to GTC.")
            params["timeInForce"] = "GTC"

        # Handle Post-Only flag
        if is_post_only:
            params["postOnly"] = True

        # Handle Reduce-Only flag
        if is_reduce_only:
            params["reduceOnly"] = True

        # Handle Client Order ID
        if client_order_id:
            valid_coid = client_order_id[:36]  # Enforce max length
            params["clientOrderId"] = valid_coid
            if len(valid_coid) < len(client_order_id):
                logger.warning(
                    f"[{func_name}] Client Order ID '{client_order_id}' truncated to 36 chars: '{valid_coid}'"
                )

        logger.info(
            f"{Fore.CYAN}{log_prefix} Placing -> Amount:{amount_float}, Price:{price_float}, Params:{params}{Style.RESET_ALL}"
        )

        # Place the limit order
        order = exchange.create_limit_order(symbol, side, amount_float, price_float, params=params)

        # Log the result
        order_id = order.get("id")
        client_oid_resp = order.get("clientOrderId", params.get("clientOrderId", "N/A"))
        status = order.get("status", "?")  # 'open', 'closed', etc.
        effective_tif = order.get("timeInForce", params.get("timeInForce", "?"))
        is_post_only_resp = order.get("postOnly", params.get("postOnly", False))

        # Use logger.info for success
        logger.info(
            f"{Fore.GREEN}{log_prefix} Limit order placed successfully. ID:...{format_order_id(order_id)}, "
            f"ClientOID:{client_oid_resp}, Status:{status}, TIF:{effective_tif}, PostOnly:{is_post_only_resp}{Style.RESET_ALL}"
        )
        return order

    except ccxt.OrderImmediatelyFillable as e:
        # This specific error occurs when postOnly=True and the order would match immediately
        if params.get("postOnly"):
            logger.warning(
                f"{Fore.YELLOW}{log_prefix} PostOnly Limit Order Rejected (would fill immediately): {e}{Style.RESET_ALL}"
            )
            return None  # Return None as the order wasn't placed on the book
        else:
            # Should not happen if postOnly is False, but log as error if it does
            logger.error(
                f"{Fore.RED}{log_prefix} Unexpected OrderImmediatelyFillable without PostOnly flag: {e}{Style.RESET_ALL}"
            )
            # Decide whether to raise or return None
            # raise e # Re-raise if this is unexpected
            return None
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(f"{Fore.RED}{log_prefix} API Error placing limit order: {type(e).__name__} - {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{symbol.split('/')[0]}] ORDER FAIL (Limit {side.upper()}): {type(e).__name__}", config)
        return None  # Don't raise as limit orders usually not retried
    except Exception as e:
        logger.critical(
            f"{Back.RED}[{func_name}] Unexpected critical error placing limit order: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        send_sms_alert(
            f"[{symbol.split('/')[0]}] ORDER FAIL (Limit {side.upper()}): Unexpected {type(e).__name__}.", config
        )
        return None


# Snippet 14 / Function 14: Place Native Stop Loss Order (Stop Market)
@retry_api_call(max_retries=1, initial_delay=0)  # Stop orders usually not retried automatically
def place_native_stop_loss(
    exchange: ccxt.bybit,
    symbol: str,
    side: Literal["buy", "sell"],
    amount: Decimal,
    stop_price: Decimal,
    config: Config,
    trigger_by: Literal["LastPrice", "MarkPrice", "IndexPrice"] = "MarkPrice",
    client_order_id: Optional[str] = None,
    position_idx: Literal[0, 1, 2] = 0,
) -> Optional[Dict]:
    """
    Places a native Stop Market order on Bybit V5, intended as a Stop Loss (reduceOnly=True).
    Uses V5 specific parameters like 'stopLoss', 'slTriggerBy', 'slOrderType'.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        side: 'buy' (for closing short) or 'sell' (for closing long).
        amount: Order quantity in base currency (Decimal). Must match position size.
        stop_price: The price at which the stop loss triggers (Decimal).
        config: Configuration object.
        trigger_by: Price type to trigger the stop ('LastPrice', 'MarkPrice', 'IndexPrice'). Default 'MarkPrice'.
        client_order_id: Optional client order ID string.
        position_idx: Position index (0 for One-Way mode, 1 for Buy Hedge, 2 for Sell Hedge). Default 0.

    Returns:
        The order dictionary returned by ccxt (may represent the trigger order), or None if placement failed.
    """
    func_name = "place_native_stop_loss"
    log_prefix = f"[Native SL {side.upper()}]"
    logger.info(
        f"{Fore.CYAN}{log_prefix} Init: {format_amount(exchange, symbol, amount)} {symbol}, Trigger @ {format_price(exchange, symbol, stop_price)} "
        f"(By: {trigger_by}), PosIdx:{position_idx}, Coid:{client_order_id or 'None'}...{Style.RESET_ALL}"
    )

    # Validate inputs
    if amount <= config.POSITION_QTY_EPSILON:
        logger.error(f"{Fore.RED}{log_prefix} Invalid amount: {amount}. Must be positive.{Style.RESET_ALL}")
        return None
    if stop_price <= Decimal("0"):
        logger.error(f"{Fore.RED}{log_prefix} Invalid stop price: {stop_price}. Must be positive.{Style.RESET_ALL}")
        return None

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category or category not in ["linear", "inverse"]:
            logger.error(
                f"{Fore.RED}[{func_name}] Cannot place stop loss for non-contract symbol: {symbol} (Category: {category}).{Style.RESET_ALL}"
            )
            return None

        # Format amount and stop price strings
        amount_str = format_amount(exchange, symbol, amount)
        amount_float = float(amount_str)
        stop_price_str = format_price(exchange, symbol, stop_price)

        # Prepare V5 parameters for a Stop Market Loss order
        # Using create_order with specific params seems more reliable for V5 stops
        params: Dict[str, Any] = {
            "category": category,
            "stopLoss": stop_price_str,  # The trigger price for the stop loss
            "slTriggerBy": trigger_by,  # Trigger type (LastPrice, MarkPrice, IndexPrice)
            "reduceOnly": True,  # Essential for stop loss to only close position
            "positionIdx": position_idx,  # 0 for one-way, 1/2 for hedge
            "tpslMode": "Full",  # 'Full' for position TP/SL, 'Partial' requires slSize
            "slOrderType": "Market",  # Execute as Market order when triggered
            # 'slSize': amount_str,        # Only needed if tpslMode='Partial'
        }

        if client_order_id:
            valid_coid = client_order_id[:36]
            params["clientOrderId"] = valid_coid
            if len(valid_coid) < len(client_order_id):
                logger.warning(f"[{func_name}] Client Order ID '{client_order_id}' truncated: '{valid_coid}'")

        bg = Back.YELLOW
        fg = Fore.BLACK
        logger.warning(
            f"{bg}{fg}{Style.BRIGHT}{log_prefix} Placing NATIVE Stop Loss (Market exec) -> "
            f"Qty:{amount_float}, Side:{side}, TriggerPx:{stop_price_str}, TriggerBy:{trigger_by}, "
            f"Reduce:True, PosIdx:{position_idx}, Params:{params}{Style.RESET_ALL}"
        )

        # Use create_order with type='market' and stop loss params
        # CCXT might internally map this or pass params directly
        # The 'side' here indicates the direction of the order needed to CLOSE the position
        # (e.g., if LONG, SL side is 'sell'; if SHORT, SL side is 'buy')
        sl_order = exchange.create_order(
            symbol=symbol,
            type="market",  # Base type is market (triggered)
            side=side,  # Direction to close the position
            amount=amount_float,
            params=params,
        )

        # Log the result - response structure might vary based on CCXT version / exchange behavior
        order_id = sl_order.get("id")  # This might be the ID of the trigger order
        client_oid_resp = sl_order.get("clientOrderId", params.get("clientOrderId", "N/A"))
        status = sl_order.get("status", "?")  # Status of the trigger order ('untriggered', 'triggered', 'active'?)

        # Try to confirm trigger price and type from response info
        info = sl_order.get("info", {})
        returned_stop_price_str = info.get("stopLoss", sl_order.get("stopPrice"))  # Check info first
        returned_stop_price = safe_decimal_conversion(returned_stop_price_str, None)
        returned_trigger = info.get("slTriggerBy", trigger_by)  # Fallback to input if not in response

        # Use logger.info for success
        logger.info(
            f"{Fore.GREEN}{log_prefix} Native SL trigger order placed successfully. "
            f"ID:...{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, "
            f"Trigger:{format_price(exchange, symbol, returned_stop_price)} (by {returned_trigger}){Style.RESET_ALL}"
        )
        return sl_order

    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadSymbol) as e:
        logger.error(f"{Fore.RED}{log_prefix} API Error placing native SL: {type(e).__name__} - {e}{Style.RESET_ALL}")
        # Consider sending SMS alert for critical SL failures
        # send_sms_alert(f"[{symbol.split('/')[0]}] SL PLACE FAIL ({side.upper()}): {type(e).__name__}", config)
        return None
    except Exception as e:
        logger.critical(
            f"{Back.RED}[{func_name}] Unexpected critical error placing native SL: {e}{Style.RESET_ALL}", exc_info=True
        )
        send_sms_alert(
            f"[{symbol.split('/')[0]}] SL PLACE FAIL ({side.upper()}): Unexpected {type(e).__name__}", config
        )
        return None


# Snippet 15 / Function 15: Fetch Open Orders (Filtered)
@retry_api_call(max_retries=2, initial_delay=1.0)
def fetch_open_orders_filtered(
    exchange: ccxt.bybit,
    symbol: str,
    config: Config,
    side: Optional[Literal["buy", "sell"]] = None,
    order_type: Optional[str] = None,
    order_filter: Optional[Literal["Order", "StopOrder", "tpslOrder"]] = None,
) -> Optional[List[Dict]]:
    """
    Fetches open orders for a specific symbol on Bybit V5, with optional filtering
    by side, CCXT order type, and/or Bybit V5 `orderFilter`. Defaults to fetching 'Order'.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol.
        config: Configuration object.
        side: Optional filter by 'buy' or 'sell'.
        order_type: Optional filter by CCXT order type (e.g., 'limit', 'market', 'stop', 'stop_market').
                    Filtering logic attempts basic matching.
        order_filter: Optional Bybit V5 specific filter ('Order', 'StopOrder', 'tpslOrder').
                      If None, defaults to 'Order' (regular limit/market).

    Returns:
        A list of open order dictionaries matching the filters, or None if an API error occurs.
        Returns an empty list if no matching orders are found.
    """
    func_name = "fetch_open_orders_filtered"
    v5_filter = order_filter if order_filter else "Order"  # Default to regular orders if not specified
    filter_log = f"(Side:{side or 'Any'}, Type:{order_type or 'Any'}, V5Filter:{v5_filter})"
    logger.debug(f"[{func_name}] Fetching open orders for {symbol} {filter_log}...")

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(
                f"{Fore.RED}[{func_name}] Cannot determine market category for {symbol}. Cannot fetch orders.{Style.RESET_ALL}"
            )
            return None

        # Prepare parameters for fetch_open_orders
        params: Dict[str, Any] = {"category": category}
        params["orderFilter"] = v5_filter  # Use the determined V5 filter

        logger.debug(f"[{func_name}] Calling fetch_open_orders with symbol='{symbol}', params={params}")
        open_orders = exchange.fetch_open_orders(symbol=symbol, params=params)

        if not open_orders:
            logger.debug(f"[{func_name}] No open orders found matching V5 filter '{v5_filter}' for {symbol}.")
            return []

        # Apply client-side filtering based on optional 'side' and 'order_type' args
        filtered_orders = open_orders
        initial_count = len(filtered_orders)

        # Filter by side
        if side:
            side_lower = side.lower()
            filtered_orders = [o for o in filtered_orders if o.get("side", "").lower() == side_lower]
            logger.debug(f"[{func_name}] Filtered by side='{side}'. Count: {initial_count} -> {len(filtered_orders)}.")
            initial_count = len(filtered_orders)  # Update count for next filter log

        # Filter by CCXT order type (more complex due to variations like 'stop_market')
        if order_type:
            # Normalize the filter type for comparison (lowercase, remove separators)
            norm_type_filter = order_type.lower().replace("_", "").replace("-", "")
            count_before_type_filter = len(filtered_orders)

            def check_type(order: Dict) -> bool:
                """Helper to check if an order matches the type filter."""
                o_type = order.get("type", "").lower().replace("_", "").replace("-", "")
                # Direct match
                if o_type == norm_type_filter:
                    return True
                # Handle stop/trigger orders where base type might be 'market' or 'limit'
                # Check if the filter includes 'stop' or 'takeprofit' and if the order has trigger price info
                has_trigger_price = (
                    order.get("stopPrice")
                    or order.get("info", {}).get("stopLoss")
                    or order.get("info", {}).get("takeProfit")
                    or order.get("info", {}).get("triggerPrice")
                )

                if ("stop" in norm_type_filter or "takeprofit" in norm_type_filter) and has_trigger_price:
                    # If filter is 'stopmarket', check if order type is 'market' and has trigger
                    if "market" in norm_type_filter and o_type == "market":
                        return True
                    # If filter is 'stoplimit', check if order type is 'limit' and has trigger
                    if "limit" in norm_type_filter and o_type == "limit":
                        return True
                    # If filter is just 'stop' or 'takeprofit', match any order with a trigger
                    if norm_type_filter in ["stop", "takeprofit"]:
                        return True

                return False

            filtered_orders = [o for o in filtered_orders if check_type(o)]
            logger.debug(
                f"[{func_name}] Filtered by type='{order_type}'. Count: {count_before_type_filter} -> {len(filtered_orders)}."
            )

        logger.info(f"[{func_name}] Fetched and filtered {len(filtered_orders)} open orders for {symbol} {filter_log}.")
        return filtered_orders

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching open orders for {symbol}: {e}{Style.RESET_ALL}")
        raise  # Re-raise for retry decorator
    except Exception as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Unexpected error fetching open orders for {symbol}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# Snippet 18 / Function 18: Place Native Trailing Stop Order
@retry_api_call(max_retries=1, initial_delay=0)  # Stop orders usually not retried automatically
def place_native_trailing_stop(
    exchange: ccxt.bybit,
    symbol: str,
    side: Literal["buy", "sell"],
    amount: Decimal,
    trailing_offset: Union[Decimal, str],
    config: Config,
    activation_price: Optional[Decimal] = None,
    trigger_by: Literal["LastPrice", "MarkPrice", "IndexPrice"] = "MarkPrice",
    client_order_id: Optional[str] = None,
    position_idx: Literal[0, 1, 2] = 0,
) -> Optional[Dict]:
    """
    Places a native Trailing Stop Market order on Bybit V5 (reduceOnly=True).
    Uses V5 specific parameters like 'trailingStop' (percentage) or 'trailingMove' (absolute delta)
    and optional 'activePrice'.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol.
        side: 'buy' (to close short) or 'sell' (to close long).
        amount: Order quantity (must match position size for full close).
        trailing_offset: The trailing distance.
                         - Decimal: Absolute price distance (e.g., Decimal('100') for $100).
                         - str ending with '%': Percentage distance (e.g., "1.5%" for 1.5%).
                           Valid percentage range: 0.01% to 10.0%.
        config: Configuration object.
        activation_price: Optional price at which the trailing stop becomes active. If None, it activates immediately.
        trigger_by: Price type used for trailing ('LastPrice', 'MarkPrice', 'IndexPrice'). Default 'MarkPrice'.
        client_order_id: Optional client order ID.
        position_idx: Position index (0 for One-Way, 1/2 for Hedge). Default 0.

    Returns:
        The order dictionary representing the placed trigger order, or None if failed.

    Raises:
        ValueError: If inputs like amount, offset, or activation price are invalid.
    """
    func_name = "place_native_trailing_stop"
    log_prefix = f"[Native TSL {side.upper()}]"

    # --- Input Validation ---
    if amount <= config.POSITION_QTY_EPSILON:
        raise ValueError(f"{log_prefix} Invalid amount: {amount}. Must be positive.")
    if activation_price is not None and activation_price <= Decimal("0"):
        raise ValueError(f"{log_prefix} Activation price, if provided, must be positive: {activation_price}")

    params: Dict[str, Any] = {}
    trail_log_str = ""  # For logging the type of trail

    # Determine trailing parameter based on offset type
    if isinstance(trailing_offset, str) and trailing_offset.endswith("%"):
        try:
            percent_val_str = trailing_offset.rstrip("%")
            percent_val = Decimal(percent_val_str)
            # Bybit V5 percentage range: 0.1% to 10% (adjust if needed based on docs) - Let's use 0.01% to 10% for broader range check
            if not (Decimal("0.01") <= percent_val <= Decimal("10.0")):
                raise ValueError(f"Percentage must be between 0.01% and 10.0%. Got: {percent_val}%")
            # Bybit expects percentage as string value, e.g., "1.5" for 1.5%
            params["trailingStop"] = str(
                percent_val.quantize(Decimal("0.01"))
            )  # Format to 2 decimal places for percentage string
            trail_log_str = f"{percent_val}%"
        except (ValueError, InvalidOperation) as e:
            raise ValueError(f"{log_prefix} Invalid trailing percentage '{trailing_offset}': {e}") from e
    elif isinstance(trailing_offset, Decimal):
        if trailing_offset <= Decimal("0"):
            raise ValueError(f"{log_prefix} Absolute trailing delta must be positive: {trailing_offset}")
        try:
            # Format the absolute delta using market price precision
            delta_str = format_price(exchange, symbol, trailing_offset)
            params["trailingMove"] = delta_str  # V5 param for absolute price offset
            trail_log_str = f"{delta_str} (abs)"
        except Exception as fmt_e:
            # Catch potential errors during formatting (e.g., market not loaded)
            raise ValueError(
                f"{log_prefix} Cannot format trailing offset {trailing_offset} using market precision: {fmt_e}"
            ) from fmt_e
    else:
        raise ValueError(
            f"{log_prefix} Invalid trailing_offset type: {type(trailing_offset)}. Must be Decimal (absolute) or str ('x.y%')."
        )

    # --- Log Initialization ---
    act_px_log = format_price(exchange, symbol, activation_price) if activation_price else "Immediate"
    logger.info(
        f"{Fore.CYAN}{log_prefix} Init: {format_amount(exchange, symbol, amount)} {symbol}, Trail:{trail_log_str}, "
        f"ActPx:{act_px_log}, TriggerBy:{trigger_by}, PosIdx:{position_idx}, Coid:{client_order_id or 'None'}...{Style.RESET_ALL}"
    )

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category or category not in ["linear", "inverse"]:
            raise ValueError(
                f"{log_prefix} Cannot place trailing stop for non-contract symbol: {symbol} (Category: {category})."
            )

        # Format amount and activation price
        amount_str = format_amount(exchange, symbol, amount)
        amount_float = float(amount_str)
        activation_price_str = format_price(exchange, symbol, activation_price) if activation_price else None

        # --- Assemble V5 Parameters ---
        # Update params dict with common TSL settings
        params.update(
            {
                "category": category,
                "reduceOnly": True,  # Essential for TSL to close position
                "positionIdx": position_idx,  # 0 for one-way, 1/2 for hedge
                "tpslMode": "Full",  # Assumes trailing stop applies to the full position
                "triggerBy": trigger_by,  # Price type for trailing calculation
                "tslOrderType": "Market",  # Execute as Market order when triggered
            }
        )
        if activation_price_str is not None:
            params["activePrice"] = activation_price_str  # V5 param for activation price

        if client_order_id:
            valid_coid = client_order_id[:36]
            params["clientOrderId"] = valid_coid
            if len(valid_coid) < len(client_order_id):
                logger.warning(f"[{func_name}] Client Order ID '{client_order_id}' truncated: '{valid_coid}'")

        # --- Place Order ---
        bg = Back.YELLOW
        fg = Fore.BLACK
        logger.warning(
            f"{bg}{fg}{Style.BRIGHT}{log_prefix} Placing NATIVE TSL (Market exec) -> "
            f"Qty:{amount_float}, Side:{side}, Trail:{trail_log_str}, ActPx:{activation_price_str or 'Immediate'}, "
            f"TriggerBy:{trigger_by}, Reduce:True, PosIdx:{position_idx}, Params:{params}{Style.RESET_ALL}"
        )

        # Use create_order with type='market' and TSL params
        tsl_order = exchange.create_order(
            symbol=symbol,
            type="market",  # Base type is market (triggered)
            side=side,  # Direction to close the position
            amount=amount_float,
            params=params,
        )

        # --- Log Result ---
        order_id = tsl_order.get("id")
        client_oid_resp = tsl_order.get("clientOrderId", params.get("clientOrderId", "N/A"))
        status = tsl_order.get("status", "?")  # Status of the trigger ('untriggered', etc.)

        # Confirm parameters from response info
        info = tsl_order.get("info", {})
        returned_trail_value = info.get("trailingStop") or info.get("trailingMove")  # Check both params
        returned_act_price_str = info.get("activePrice", tsl_order.get("activationPrice"))
        returned_act_price = safe_decimal_conversion(returned_act_price_str, None)
        returned_trigger = info.get("triggerBy", trigger_by)

        # Use logger.info for success
        logger.info(
            f"{Fore.GREEN}{log_prefix} Native TSL trigger order placed successfully. "
            f"ID:...{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, "
            f"Trail:{returned_trail_value}, ActPx:{format_price(exchange, symbol, returned_act_price)}, TriggerBy:{returned_trigger}{Style.RESET_ALL}"
        )
        return tsl_order

    # Catch specific API/Input errors first
    except (
        ccxt.InsufficientFunds,
        ccxt.InvalidOrder,
        ccxt.ExchangeError,
        ccxt.NetworkError,
        ccxt.BadSymbol,
        ValueError,
    ) as e:
        logger.error(
            f"{Fore.RED}{log_prefix} API/Input Error placing native TSL: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        return None  # Do not raise, TSL placement usually not retried
    # Catch unexpected errors
    except Exception as e:
        logger.critical(
            f"{Back.RED}[{func_name}] Unexpected critical error placing native TSL: {e}{Style.RESET_ALL}", exc_info=True
        )
        send_sms_alert(
            f"[{symbol.split('/')[0]}] TSL PLACE FAIL ({side.upper()}): Unexpected {type(e).__name__}", config
        )
        return None


# Snippet 22 / Function 22: Update Limit Order (Edit Order)
@retry_api_call(max_retries=1, initial_delay=0)  # Edits usually not retried automatically
def update_limit_order(
    exchange: ccxt.bybit,
    symbol: str,
    order_id: str,
    config: Config,
    new_amount: Optional[Decimal] = None,
    new_price: Optional[Decimal] = None,
    new_client_order_id: Optional[str] = None,
) -> Optional[Dict]:
    """
    Attempts to modify the amount and/or price of an existing open limit order on Bybit V5.
    Requires `edit_order` support in the CCXT version.
    By default, disallows modifying partially filled orders (can be changed via flag if needed).

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol.
        order_id: The ID of the limit order to modify.
        config: Configuration object.
        new_amount: Optional new order quantity (Decimal). If None, amount is unchanged.
        new_price: Optional new limit price (Decimal). If None, price is unchanged.
        new_client_order_id: Optional new client order ID to assign during modification.

    Returns:
        The updated order dictionary returned by ccxt, or None if modification failed or wasn't possible.

    Raises:
        ValueError: If inputs are invalid (e.g., no change specified, negative amount/price).
    """
    func_name = "update_limit_order"
    log_prefix = f"[Update Order ...{format_order_id(order_id)}]"

    # --- Input Validation ---
    if new_amount is None and new_price is None:
        raise ValueError(f"{log_prefix} No new amount or price provided for update.")
    if new_amount is not None and new_amount <= config.POSITION_QTY_EPSILON:
        raise ValueError(f"{log_prefix} Invalid new amount specified: {new_amount}. Must be positive.")
    if new_price is not None and new_price <= Decimal("0"):
        raise ValueError(f"{log_prefix} Invalid new price specified: {new_price}. Must be positive.")

    # Log intent
    amount_log = format_amount(exchange, symbol, new_amount) if new_amount else "NoChange"
    price_log = format_price(exchange, symbol, new_price) if new_price else "NoChange"
    coid_log = new_client_order_id or "NoChange"
    logger.info(
        f"{Fore.CYAN}{log_prefix} Init update for {symbol}: NewAmount:{amount_log}, NewPrice:{price_log}, NewCoid:{coid_log}{Style.RESET_ALL}"
    )

    try:
        # Check if exchange supports editing orders
        if not exchange.has.get("editOrder"):
            logger.error(
                f"{Fore.RED}{log_prefix} Exchange '{exchange.id}' via CCXT does not support 'editOrder'. Cannot update.{Style.RESET_ALL}"
            )
            return None

        # Fetch the current state of the order to validate and get details
        logger.debug(f"[{func_name}] Fetching current order state for ID {order_id}...")
        market = exchange.market(symbol)  # Need market for category
        category = _get_v5_category(market)
        if not category:
            raise ValueError(f"Cannot determine category for {symbol}")
        fetch_params = {"category": category}
        current_order = exchange.fetch_order(order_id, symbol, params=fetch_params)

        # Validate current order state
        status = current_order.get("status")
        order_type = current_order.get("type")
        filled_qty = safe_decimal_conversion(current_order.get("filled", "0.0"), Decimal("0.0"))

        if status != "open":
            # Cannot modify orders that are not open (e.g., filled, canceled, rejected)
            raise ccxt.InvalidOrder(f"{log_prefix} Cannot update order. Current status is '{status}' (must be 'open').")
        if order_type != "limit":
            # Bybit typically only allows modifying limit orders
            raise ccxt.InvalidOrder(
                f"{log_prefix} Cannot update order. Current type is '{order_type}' (must be 'limit')."
            )

        # Check for partial fills (Bybit V5 might allow modifying partially filled, but can be complex)
        # Default behavior: Disallow modification if partially filled.
        allow_partial_fill_update = False  # Set to True to attempt modification even if partially filled
        if not allow_partial_fill_update and filled_qty > config.POSITION_QTY_EPSILON:
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] Update aborted: Order ...{format_order_id(order_id)} is partially filled "
                f"({format_amount(exchange, symbol, filled_qty)}). Modification disallowed by default.{Style.RESET_ALL}"
            )
            return None  # Or raise InvalidOrder if preferred

        # Determine the final amount and price for the edit call
        final_amount_dec = (
            new_amount if new_amount is not None else safe_decimal_conversion(current_order.get("amount"))
        )
        final_price_dec = new_price if new_price is not None else safe_decimal_conversion(current_order.get("price"))

        # Ensure final values are valid after potentially using current values
        if final_amount_dec is None or final_amount_dec <= config.POSITION_QTY_EPSILON:
            raise ValueError(f"Invalid final amount calculated for update: {final_amount_dec}")
        if final_price_dec is None or final_price_dec <= Decimal("0"):
            raise ValueError(f"Invalid final price calculated for update: {final_price_dec}")

        # Prepare parameters for edit_order
        edit_params: Dict[str, Any] = {"category": category}
        if new_client_order_id:
            valid_coid = new_client_order_id[:36]
            edit_params["clientOrderId"] = valid_coid
            if len(valid_coid) < len(new_client_order_id):
                logger.warning(f"[{func_name}] New Client Order ID '{new_client_order_id}' truncated: '{valid_coid}'")

        # Convert final amount/price to float for CCXT call
        final_amount_float = float(format_amount(exchange, symbol, final_amount_dec))
        final_price_float = float(format_price(exchange, symbol, final_price_dec))

        # Log the edit attempt
        logger.info(
            f"{Fore.CYAN}[{func_name}] Submitting update via edit_order -> "
            f"ID:{order_id}, Side:{current_order['side']}, Amount:{final_amount_float}, Price:{final_price_float}, Params:{edit_params}{Style.RESET_ALL}"
        )

        # Call edit_order
        updated_order = exchange.edit_order(
            id=order_id,
            symbol=symbol,
            type="limit",  # Must specify type again
            side=current_order["side"],  # Must specify side again
            amount=final_amount_float,
            price=final_price_float,
            params=edit_params,
        )

        # Process the response
        if updated_order:
            new_id = updated_order.get("id", order_id)  # ID might change on some exchanges after edit
            status_after = updated_order.get("status", "?")
            new_client_oid_resp = updated_order.get("clientOrderId", edit_params.get("clientOrderId", "N/A"))
            # Use logger.info for success
            logger.info(
                f"{Fore.GREEN}[{func_name}] Update successful. New/Current ID:...{format_order_id(new_id)}, "
                f"Status:{status_after}, ClientOID:{new_client_oid_resp}{Style.RESET_ALL}"
            )
            return updated_order
        else:
            # Should not happen if editOrder succeeds, but handle defensively
            logger.warning(
                f"{Fore.YELLOW}[{func_name}] edit_order call succeeded but returned no data. Check order status manually.{Style.RESET_ALL}"
            )
            # Fetch the order again to confirm?
            # return exchange.fetch_order(order_id, symbol, params=fetch_params)
            return None

    # Catch specific, expected errors during update
    except (
        ccxt.OrderNotFound,
        ccxt.InvalidOrder,
        ccxt.NotSupported,
        ccxt.ExchangeError,
        ccxt.NetworkError,
        ccxt.BadSymbol,
        ValueError,
    ) as e:
        logger.error(
            f"{Fore.RED}[{func_name}] Failed to update order ...{format_order_id(order_id)}: {type(e).__name__} - {e}{Style.RESET_ALL}"
        )
        return None  # Return None on failure, don't raise as edits not retried
    # Catch unexpected errors
    except Exception as e:
        logger.critical(
            f"{Back.RED}[{func_name}] Unexpected critical error updating order ...{format_order_id(order_id)}: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        return None


# --- END OF FILE order_management.py ---

# ---------------------------------------------------------------------------
