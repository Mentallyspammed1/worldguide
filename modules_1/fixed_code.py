```python
import asyncio
import logging
import sys  # Retained as it was in original imports
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation # ROUND_DOWN, ROUND_UP kept as they were in original imports
from typing import Any, Dict, Optional, Union # Added Union

# Third-party libraries (assuming ccxt is used like this)
import ccxt  # Type hint placeholder, actual ccxt library

# Local application/library specific imports
# These are assumed to be async functions, correctly imported from their respective modules.
# Example: from .exchange_api import (
#    fetch_balance, fetch_current_price_ccxt, place_trade, get_open_position,
#    _set_position_protection, set_trailing_stop_loss, get_market_info,
#    fetch_klines_ccxt, fetch_orderbook_ccxt, set_leverage_ccxt
# )
# For brevity, these imports are omitted here but are assumed to be present.
# We'll assume these functions are defined and available:
# get_market_info, fetch_klines_ccxt, fetch_current_price_ccxt, fetch_orderbook_ccxt,
# get_open_position, fetch_balance, set_leverage_ccxt, place_trade,
# set_trailing_stop_loss, _set_position_protection

# --- Constants ---
# ANSI escape codes for colored logging
NEON_BLUE = "\033[1;94m"
RESET = "\033[0m"

# Default delay in seconds to wait for a position to be confirmed after placing an order
POSITION_CONFIRM_DELAY_SECONDS = 5

# Placeholder for exchange_api functions if they are not imported from a module
# These would typically be in a separate module, e.g., 'exchange_api.py'
# Ensure these functions are defined as async def
# Example:
# async def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Dict[str, Any]: ...
# (Actual definitions of these functions are omitted as per the problem focusing on enhancing provided code)


async def _execute_close_position(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict[str, Any],
    open_position: Dict[str, Any],
    logger: logging.Logger,
    reason: str = "exit signal"
) -> bool:
    """
    Closes an existing open position for the given symbol.
    """
    lg = logger  # Shorthand for logger
    # Assume pos_size, close_side_signal are derived from open_position and reason
    # For example:
    pos_size = Decimal(str(open_position.get('contracts', '0'))) # Example
    close_side = "SELL" if open_position.get('side') == "long" else "BUY" # Example
    close_side_signal = {"signal": close_side, "price": await fetch_current_price_ccxt(exchange, symbol, lg)} # Example

    lg.info(f"Attempting to close {symbol} position due to: {reason}.")

    # Ensure pos_size is valid
    if pos_size <= Decimal("0"):
        lg.warning(f"Cannot close position for {symbol}, size is zero or invalid: {pos_size}")
        return False

    # Place a market order to close the position
    close_order = await place_trade(
        exchange=exchange,
        symbol=symbol,
        trade_signal=close_side_signal,  # This needs to be structured as expected by place_trade
        position_size=pos_size,
        market_info=market_info,
        logger=lg,
        order_type='market',
        reduce_only=True
    )

    if close_order and close_order.get('id'):
        lg.info(f"Close order placed for {symbol}, ID: {close_order['id']}. Reason: {reason}.")
        # Potentially wait and confirm closure
        return True
    else:
        lg.error(f"Failed to place close order for {symbol}. Reason: {reason}.")
        return False


async def analyze_and_trade_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    config: Dict[str, Any],
    logger: logging.Logger,
    enable_trading: bool
) -> None:
    """
    Analyzes a trading symbol based on configuration and market data,
    and executes trades if conditions are met and trading is enabled.
    """
    lg = logger  # Shorthand for logger
    cycle_start_time = asyncio.get_event_loop().time()

    lg.info(f"Starting analysis cycle for {symbol}...")

    # Fetch market data
    market_info = await get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"Failed to fetch market info for {symbol}. Skipping cycle.")
        return

    # Assumed variables derived from config or calculated earlier
    # (These would be defined based on `config` or other logic before this point)
    ccxt_interval = config.get("kline_interval", "1h")
    kline_limit = config.get("kline_limit", 100)
    # `active_weights` would be derived from indicator calculations based on klines_df, etc.
    active_weights: Dict[str, str] = {} # Placeholder, should be calculated
    # `signal` would be the final trading signal ("BUY", "SELL", "HOLD") from indicators
    signal: str = "HOLD" # Placeholder
    # `pos_size_dec` is the calculated position size
    pos_size_dec: Decimal = Decimal("0") # Placeholder
    # `entry_type` ('market' or 'limit')
    entry_type: str = config.get("entry_order_type", "market") # Placeholder
    # `limit_px` if entry_type is 'limit'
    limit_px: Optional[Decimal] = None # Placeholder
    # SL/TP related variables
    tsl_enabled_config: bool = config.get("trailing_stop_loss", {}).get("enabled", False) # Placeholder
    sl_f: Optional[Decimal] = None # Placeholder for stop loss factor/price
    tp_f: Optional[Decimal] = None # Placeholder for take profit factor/price


    klines_df = await fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)
    # ... (Data processing, indicator calculation, signal generation logic would be here)
    # This part would set `active_weights`, `signal`, etc.

    current_price_fetch = await fetch_current_price_ccxt(exchange, symbol, lg)
    # ... (Potentially use current_price_fetch in signal generation or checks)

    # Conditional fetching of order book data
    orderbook_data: Optional[Dict[str, Any]] = None
    use_orderbook_indicator = config.get("indicators", {}).get("orderbook", False)
    orderbook_weight_str = active_weights.get("orderbook", "0")
    try:
        orderbook_weight = Decimal(orderbook_weight_str)
    except InvalidOperation:
        lg.warning(f"Invalid Decimal string for orderbook_weight: {orderbook_weight_str}. Defaulting to 0.")
        orderbook_weight = Decimal("0")

    if use_orderbook_indicator and orderbook_weight != Decimal("0"):
        orderbook_limit = config.get("orderbook_limit", 100)
        orderbook_data = await fetch_orderbook_ccxt(exchange, symbol, orderbook_limit, lg)
    # ... (Use orderbook_data in indicators if fetched)

    if not enable_trading:
        lg.info(f"Trading is disabled. Analysis only for {symbol}.")
        end_time_no_trade = asyncio.get_event_loop().time()
        lg.info(
            f"{NEON_BLUE}---== Analysis Cycle End ({symbol}, {end_time_no_trade - cycle_start_time:.2f}s) ==---{RESET}\n"
        )
        return

    open_position = await get_open_position(exchange, symbol, market_info, lg)

    # --- Scenario 1: No Open Position ---
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"Signal '{signal}' received for {symbol}, no open position. Preparing to open new position.")
            # ... (Position sizing logic to calculate pos_size_dec would be here)
            # Example: pos_size_dec = calculate_position_size(...)

            if pos_size_dec <= Decimal("0"):
                lg.warning(f"Calculated position size is {pos_size_dec} for {symbol}. Cannot place trade.")
            else:
                quote_currency = config.get("quote_currency", "USDT")
                balance = await fetch_balance(exchange, quote_currency, lg)
                # ... (Further checks involving balance, risk management)

                # Set leverage if applicable (for futures/margin)
                if not market_info.get('spot', True): # If not a spot market
                    leverage = int(config.get("leverage", 1))
                    if leverage > 0:
                        lg.info(f"Setting leverage to {leverage}x for {symbol}.")
                        if not await set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                            lg.warning(f"Failed to set leverage for {symbol}. Proceeding with default/current leverage.")
                    # ... (Handle leverage setting failure if critical)

                lg.info(f"Placing {signal} order for {pos_size_dec} of {symbol}.")
                trade_order = await place_trade(
                    exchange, symbol, signal, pos_size_dec, market_info, lg,
                    entry_type, limit_px, reduce_only=False
                )

                if trade_order and trade_order.get('id'):
                    lg.info(f"Trade order {trade_order['id']} placed for {symbol}.")
                    order_status = trade_order.get('status')
                    # If market order or limit order that filled immediately
                    if entry_type == 'market' or (entry_type == 'limit' and order_status == 'closed'):
                        confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                        lg.info(f"Waiting {confirm_delay}s for position confirmation for {symbol}...")
                        await asyncio.sleep(confirm_delay)
                        confirmed_pos = await get_open_position(exchange, symbol, market_info, lg)

                        if confirmed_pos:
                            lg.info(f"Position confirmed for {symbol}: Side {confirmed_pos.get('side')}, Size {confirmed_pos.get('contracts')}.")
                            # Set Trailing Stop Loss or Static SL/TP
                            # Assumed variables: tsl_enabled_config, sl_f, tp_f
                            tp_target_for_tsl: Optional[Decimal] = None # Placeholder if TSL needs a TP target
                            if tsl_enabled_config:
                                prot_ok = await set_trailing_stop_loss(
                                    exchange, symbol, market_info, confirmed_pos, config, lg, tp_target_for_tsl
                                )
                                lg.info(f"Trailing stop loss setting attempt for {symbol}: {'Success' if prot_ok else 'Failed'}")
                            elif (sl_f and isinstance(sl_f, Decimal) and sl_f > 0) or \
                                 (tp_f and isinstance(tp_f, Decimal) and tp_f > 0):
                                prot_ok = await _set_position_protection(
                                    exchange, symbol, market_info, confirmed_pos, lg, sl_f, tp_f
                                )
                                lg.info(f"SL/TP protection setting attempt for {symbol}: {'Success' if prot_ok else 'Failed'}")
                        else:
                            lg.warning(f"Position not confirmed for {symbol} after order execution and delay.")
                    elif entry_type == 'limit' and order_status == 'open':
                        lg.info(f"Limit order {trade_order['id']} for {symbol} is open. Will monitor.")
                    else:
                        lg.warning(f"Order {trade_order['id']} for {symbol} has status '{order_status}'. Further checks may be needed.")
                else:
                    lg.error(f"Failed to place trade order for {symbol}.")
        else:
            lg.debug(f"No action signal ('{signal}') and no open position for {symbol}.")

    else:  # --- Scenario 2: Existing Open Position ---
        pos_side = open_position.get('side') # 'long' or 'short'
        lg.info(f"Existing {pos_side} position found for {symbol}.")

        # Check for closing signal
        if (pos_side == 'long' and signal == "SELL") or \
           (pos_side == 'short' and signal == "BUY"):
            lg.info(f"Opposing signal '{signal}' received for existing {pos_side} position on {symbol}. Closing position.")
            closed = await _execute_close_position(exchange, symbol, market_info, open_position, lg, "opposing signal")
            if closed: lg.info(f"Position {symbol} closed due to opposing signal.")
            else: lg.error(f"Failed to close {symbol} position on opposing signal.")
            # Cycle ends after attempting to close
            end_time_after_close_attempt = asyncio.get_event_loop().time()
            lg.info(
                f"{NEON_BLUE}---== Analysis Cycle End ({symbol}, {end_time_after_close_attempt - cycle_start_time:.2f}s) ==---{RESET}\n"
            )
            return

        # Time-based exit
        time_exit_minutes_config = config.get("time_based_exit_minutes")
        if isinstance(time_exit_minutes_config, (int, float)) and time_exit_minutes_config > 0:
            pos_ts_ms = open_position.get('timestamp') # Position entry timestamp in milliseconds
            if pos_ts_ms and isinstance(pos_ts_ms, int):
                try:
                    current_loop_time_ms = int(asyncio.get_event_loop().time() * 1000)
                    elapsed_min = (current_loop_time_ms - pos_ts_ms) / 60000.0
                    if elapsed_min >= time_exit_minutes_config:
                        lg.info(f"Time-based exit triggered for {symbol} ({elapsed_min:.2f} min >= {time_exit_minutes_config} min).")
                        closed = await _execute_close_position(exchange, symbol, market_info, open_position, lg, "time-based exit")
                        if closed: lg.info(f"Position {symbol} closed due to time-based exit.")
                        else: lg.error(f"Failed to close {symbol} position on time-based exit.")
                        # Cycle ends after attempting to close
                        end_time_after_time_exit = asyncio.get_event_loop().time()
                        lg.info(
                            f"{NEON_BLUE}---== Analysis Cycle End ({symbol}, {end_time_after_time_exit - cycle_start_time:.2f}s) ==---{RESET}\n"
                        )
                        return
                except Exception as e:
                    lg.error(f"Error calculating time-based exit for {symbol}: {e}")

        # Dynamic SL/TP adjustments (e.g., move SL to Break-Even)
        # Assumed variables: upd_be_sl, be_px, cur_tp_dec
        upd_be_sl: bool = False # Placeholder, logic to determine if SL should be moved to BE
        be_px: Optional[Decimal] = None # Placeholder, break-even price
        cur_tp_dec: Optional[Decimal] = open_position.get('takeProfitPrice') # Placeholder

        if upd_be_sl: # If logic determines SL should be moved to break-even
            lg.info(f"Attempting to move SL to break-even for {symbol} at {be_px}.")
            if await _set_position_protection(exchange, symbol, market_info, open_position, lg, be_px, cur_tp_dec):
                lg.info(f"Successfully updated SL to break-even for {symbol}.")
            else:
                lg.warning(f"Failed to update SL to break-even for {symbol}.")

        # Manage Trailing Stop Loss if not handled by exchange or needs update
        # Assumed variables: is_tsl_exch_active, tsl_tp_target
        is_tsl_exch_active: bool = False # Placeholder, true if exchange handles TSL
        tsl_tp_target: Optional[Decimal] = None # Placeholder, target price for TSL adjustment

        if tsl_enabled_config and not is_tsl_exch_active:
            lg.debug(f"Checking/Updating custom TSL for {symbol}.")
            if await set_trailing_stop_loss(exchange, symbol, market_info, open_position, config, lg, tsl_tp_target):
                lg.info(f"Custom TSL updated/confirmed for {symbol}.")
            # ... (Handle TSL update failure if necessary)
        # ... (Other logic for managing an open position)
        lg.debug(f"End of checks for existing position on {symbol}.")

    end_time_final = asyncio.get_event_loop().time()
    lg.info(
        f"{NEON_BLUE}---== Analysis Cycle End ({symbol}, {end_time_final - cycle_start_time:.2f}s) ==---{RESET}\n"
    )
```