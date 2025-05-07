# File: xrscalper_main.py
import hashlib
import hmac
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from typing import Any, Dict, Optional, Tuple, List, Union

# Third-party imports
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import local modules
import constants
import utils
import config_loader
import logger_setup
import exchange_api
import analyzer
import risk_manager

# Initialize colorama and set precision for Decimal calculations
getcontext().prec = 38
init(autoreset=True)
load_dotenv() # Load environment variables from .env file

# Global variable to hold config (though passing explicitly is often better)
CONFIG: Dict[str, Any] = {}
QUOTE_CURRENCY: str = "USDT" # Default, will be updated from config

# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Performs one cycle of analysis and trading logic for a single symbol.
    """
    global QUOTE_CURRENCY # Access global quote currency for logging/balance checks
    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    # --- Get Market Info ---
    market_info = exchange_api.get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"{constants.NEON_RED}Failed to get market info for {symbol}. Skipping cycle.{constants.RESET}")
        return

    # --- Fetch Data ---
    ccxt_interval = constants.CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
         lg.error(f"Invalid interval '{config['interval']}' in config. Cannot map to CCXT timeframe for {symbol}. Skipping cycle.")
         return

    kline_limit = 500
    klines_df = exchange_api.fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)
    if klines_df.empty or len(klines_df) < 50:
        lg.error(f"{constants.NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}). Skipping cycle.{constants.RESET}")
        return

    current_price = exchange_api.fetch_current_price_ccxt(exchange, symbol, lg)
    if current_price is None:
         lg.warning(f"{constants.NEON_YELLOW}Failed to fetch current ticker price for {symbol}. Using last close from klines as fallback.{constants.RESET}")
         try:
             if not isinstance(klines_df.index, pd.DatetimeIndex): lg.error(f"{constants.NEON_RED}Kline DataFrame index is not DatetimeIndex.{constants.RESET}"); return
             if not klines_df.index.is_monotonic_increasing: lg.warning("Kline DataFrame index not sorted, sorting..."); klines_df.sort_index(inplace=True)
             last_close_val = klines_df['close'].iloc[-1]
             if pd.notna(last_close_val) and last_close_val > 0:
                  current_price = Decimal(str(last_close_val)); lg.info(f"Using last close price as current price: {current_price}")
             else: lg.error(f"{constants.NEON_RED}Last close price from klines is invalid ({last_close_val}). Cannot proceed.{constants.RESET}"); return
         except IndexError: lg.error(f"{constants.NEON_RED}Kline DataFrame is empty or index error getting last close.{constants.RESET}"); return
         except Exception as e: lg.error(f"{constants.NEON_RED}Error getting last close price from klines: {e}. Cannot proceed.{constants.RESET}"); return

    orderbook_data = None
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    orderbook_enabled = config.get("indicators",{}).get("orderbook", False)
    orderbook_weight = Decimal(str(active_weights.get("orderbook", "0")))
    if orderbook_enabled and orderbook_weight != 0:
         lg.debug(f"Fetching order book for {symbol} (Weight: {orderbook_weight})...")
         orderbook_data = exchange_api.fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], lg)
         if not orderbook_data: lg.warning(f"{constants.NEON_YELLOW}Failed to fetch orderbook data for {symbol}, proceeding without it.{constants.RESET}")
    else: lg.debug(f"Orderbook analysis skipped (Disabled or Zero Weight).")

    # --- Analyze Data ---
    analyzer_instance = analyzer.TradingAnalyzer(klines_df.copy(), lg, config, market_info)
    if not analyzer_instance.indicator_values:
         lg.error(f"{constants.NEON_RED}Indicator calculation failed or produced no values for {symbol}. Skipping signal generation.{constants.RESET}")
         return

    # --- Generate Signal ---
    signal = analyzer_instance.generate_trading_signal(current_price, orderbook_data)

    # --- Calculate Potential TP/SL ---
    _, tp_calc, sl_calc = analyzer_instance.calculate_entry_tp_sl(current_price, signal)
    price_precision = utils.get_price_precision(market_info, lg)
    min_tick_size = utils.get_min_tick_size(market_info, lg)
    current_atr = analyzer_instance.indicator_values.get("ATR")

    # --- Log Analysis Summary ---
    lg.info(f"Current Price: {current_price:.{price_precision}f}")
    lg.info(f"ATR: {current_atr:.{price_precision+1}f}" if isinstance(current_atr, Decimal) else 'ATR: N/A')
    lg.info(f"Calculated Initial SL (for sizing): {sl_calc if sl_calc else 'N/A'}")
    lg.info(f"Calculated Initial TP (potential target): {tp_calc if tp_calc else 'N/A'}")
    tsl_enabled = config.get('enable_trailing_stop'); be_enabled = config.get('enable_break_even')
    time_exit_minutes = config.get('time_based_exit_minutes')
    time_exit_str = f"{time_exit_minutes} min" if time_exit_minutes else "Disabled"
    lg.info(f"Position Management: TSL={'Enabled' if tsl_enabled else 'Disabled'}, BE={'Enabled' if be_enabled else 'Disabled'}, TimeExit={time_exit_str}")

    # --- Trading Execution Logic ---
    if not config.get("enable_trading", False):
        lg.debug(f"Trading is disabled in config. Analysis complete for {symbol}.")
        cycle_end_time = time.monotonic(); lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")
        return

    # --- Check Existing Position ---
    open_position = exchange_api.get_open_position(exchange, symbol, market_info, lg)

    # ================= Scenario 1: No Open Position =================
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"*** {signal} Signal & No Position: Initiating Trade Sequence for {symbol} ***")
            balance = exchange_api.fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= 0: lg.error(f"{constants.NEON_RED}Trade Aborted ({symbol} {signal}): Cannot fetch balance or balance is zero/negative.{constants.RESET}"); return
            if sl_calc is None: lg.error(f"{constants.NEON_RED}Trade Aborted ({symbol} {signal}): Initial SL calculation failed. Cannot calculate position size.{constants.RESET}"); return

            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 1))
                if leverage > 0:
                    if not exchange_api.set_leverage_ccxt(exchange, symbol, leverage, market_info, lg): lg.error(f"{constants.NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set leverage to {leverage}x.{constants.RESET}"); return
                else: lg.info(f"Leverage setting skipped: Leverage config is zero or negative ({leverage}).")
            else: lg.info(f"Leverage setting skipped (Spot market).")

            position_size = risk_manager.calculate_position_size(
                balance=balance, risk_per_trade=config["risk_per_trade"], initial_stop_loss_price=sl_calc,
                entry_price=current_price, market_info=market_info, exchange=exchange, logger=lg, quote_currency=QUOTE_CURRENCY
            )
            if position_size is None or position_size <= 0: lg.error(f"{constants.NEON_RED}Trade Aborted ({symbol} {signal}): Position size calculation failed or resulted in zero/negative ({position_size}).{constants.RESET}"); return

            entry_order_type = config.get("entry_order_type", "market"); limit_entry_price: Optional[Decimal] = None
            if entry_order_type == "limit":
                 offset_buy = Decimal(str(config.get("limit_order_offset_buy", "0.0005"))); offset_sell = Decimal(str(config.get("limit_order_offset_sell", "0.0005")))
                 if signal == "BUY":
                      raw_limit = current_price * (Decimal(1) - offset_buy)
                      if min_tick_size > 0: limit_entry_price = (raw_limit / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size
                      else: rounding_factor = Decimal('1e-' + str(price_precision)); limit_entry_price = raw_limit.quantize(rounding_factor, rounding=ROUND_DOWN)
                 else: # SELL
                      raw_limit = current_price * (Decimal(1) + offset_sell)
                      if min_tick_size > 0: limit_entry_price = (raw_limit / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
                      else: rounding_factor = Decimal('1e-' + str(price_precision)); limit_entry_price = raw_limit.quantize(rounding_factor, rounding=ROUND_UP)
                 if limit_entry_price <= 0: lg.error(f"{constants.NEON_RED}Trade Aborted ({symbol} {signal}): Calculated limit entry price non-positive ({limit_entry_price}). Switching to Market.{constants.RESET}"); entry_order_type = "market"; limit_entry_price = None
                 else: lg.info(f"Calculated Limit Entry Price for {signal}: {limit_entry_price}")

            lg.info(f"==> Placing {signal} {entry_order_type.upper()} order | Size: {position_size} <==")
            trade_order = exchange_api.place_trade(
                exchange=exchange, symbol=symbol, trade_signal=signal, position_size=position_size, market_info=market_info, logger=lg,
                order_type=entry_order_type, limit_price=limit_entry_price, reduce_only=False, params=None, quote_currency=QUOTE_CURRENCY
            )

            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']; order_status = trade_order.get('status')
                if entry_order_type == 'market' or order_status == 'closed': # Handle market or immediate limit fill
                    confirm_delay = config.get("position_confirm_delay_seconds", constants.POSITION_CONFIRM_DELAY_SECONDS)
                    fill_type = "Market" if entry_order_type == 'market' else "Immediate Limit"
                    lg.info(f"{fill_type} order {order_id} placed/filled. Waiting {confirm_delay}s for confirmation...")
                    time.sleep(confirm_delay)
                    lg.info(f"Attempting position confirmation for {symbol} after {fill_type} fill {order_id}...")
                    confirmed_position = exchange_api.get_open_position(exchange, symbol, market_info, lg)

                    if confirmed_position:
                        lg.info(f"{constants.NEON_GREEN}Position Confirmed after {fill_type} Order!{constants.RESET}")
                        try:
                            entry_price_actual = confirmed_position.get('entryPriceDecimal')
                            if entry_price_actual is None or entry_price_actual <= 0:
                                lg.warning(f"Could not get valid actual entry price from confirmed position. Using initial estimate {current_price} for protection.")
                                entry_price_actual = current_price # Fallback
                            lg.info(f"Actual Entry Price: ~{entry_price_actual:.{price_precision}f}")
                            _, tp_final, sl_final = analyzer_instance.calculate_entry_tp_sl(entry_price_actual, signal)

                            protection_set_success = False
                            if config.get("enable_trailing_stop", False):
                                 lg.info(f"Setting Exchange Trailing Stop Loss (TP target: {tp_final})...")
                                 protection_set_success = exchange_api.set_trailing_stop_loss(exchange=exchange, symbol=symbol, market_info=market_info, position_info=confirmed_position, config=config, logger=lg, take_profit_price=tp_final)
                            else:
                                 lg.info(f"Setting Fixed SL ({sl_final}) and TP ({tp_final})...")
                                 if sl_final or tp_final:
                                     protection_set_success = exchange_api._set_position_protection(exchange=exchange, symbol=symbol, market_info=market_info, position_info=confirmed_position, logger=lg, stop_loss_price=sl_final, take_profit_price=tp_final)
                                 else: lg.warning(f"{constants.NEON_YELLOW}Fixed SL/TP calculation failed. No fixed protection set.{constants.RESET}")

                            if protection_set_success: lg.info(f"{constants.NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({symbol} {signal}) ===")
                            else: lg.error(f"{constants.NEON_RED}=== TRADE placed BUT FAILED TO SET PROTECTION ({symbol} {signal}) ==="); lg.warning(f"{constants.NEON_YELLOW}>>> MANUAL MONITORING REQUIRED! <<<")
                        except Exception as post_trade_err: lg.error(f"{constants.NEON_RED}Error during post-trade protection setting ({symbol}): {post_trade_err}{constants.RESET}", exc_info=True); lg.warning(f"{constants.NEON_YELLOW}Position open but protection setup failed. Manual check needed!{constants.RESET}")
                    else: lg.error(f"{constants.NEON_RED}{fill_type} trade order {order_id} placed, but FAILED TO CONFIRM open position! Manual investigation required!{constants.RESET}")

                elif entry_order_type == 'limit' and order_status == 'open':
                     lg.info(f"Limit order {order_id} placed successfully and is OPEN. Will check status next cycle.")
                else: # Limit order failed or other status
                     lg.error(f"Limit order {order_id} placement resulted in status: {order_status}. Trade did not open.")
            else: lg.error(f"{constants.NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). Order placement function returned None. ===")
        else: # signal == HOLD
            lg.info(f"Signal is HOLD and no open position for {symbol}. No entry action taken.")

    # ================= Scenario 2: Existing Open Position =================
    else: # open_position is not None
        pos_side = open_position.get('side', 'unknown'); pos_size = open_position.get('contractsDecimal', Decimal('0'))
        entry_price = open_position.get('entryPriceDecimal'); pos_timestamp_ms = open_position.get('timestamp_ms')

        if pos_side not in ['long', 'short'] or pos_size == 0 or entry_price is None or entry_price <= 0:
             lg.error(f"{constants.NEON_RED}Cannot manage position for {symbol}: Invalid details retrieved (Side: {pos_side}, Size: {pos_size}, Entry: {entry_price}). Skipping.{constants.RESET}"); lg.debug(f"Problematic position data: {open_position}"); return

        lg.info(f"Managing existing {pos_side.upper()} position for {symbol}. Size: {pos_size}, Entry: {entry_price}")
        exit_signal_triggered = (pos_side == 'long' and signal == "SELL") or (pos_side == 'short' and signal == "BUY")

        if exit_signal_triggered:
            lg.warning(f"{constants.NEON_YELLOW}*** EXIT Signal Triggered: New signal ({signal}) opposes existing {pos_side} position. Closing position... ***{constants.RESET}")
            try:
                close_side_signal = "SELL" if pos_side == 'long' else "BUY"; size_to_close = abs(pos_size)
                if size_to_close <= 0: raise ValueError(f"Position size to close is zero or negative ({size_to_close}).")
                lg.info(f"==> Placing {close_side_signal} MARKET order (reduceOnly=True) | Size: {size_to_close} <==")
                close_order = exchange_api.place_trade(exchange=exchange, symbol=symbol, trade_signal=close_side_signal, position_size=size_to_close, market_info=market_info, logger=lg, order_type='market', reduce_only=True, quote_currency=QUOTE_CURRENCY)
                if close_order: lg.info(f"{constants.NEON_GREEN}Position CLOSE order placed successfully for {symbol}. Order ID: {close_order.get('id', 'N/A')}{constants.RESET}"); return # Exit after placing close
                else: lg.error(f"{constants.NEON_RED}Failed to place CLOSE order for {symbol}. Manual check required!{constants.RESET}"); return # Exit after failed close
            except Exception as close_err: lg.error(f"{constants.NEON_RED}Error attempting to close position {symbol}: {close_err}{constants.RESET}", exc_info=True); lg.warning(f"{constants.NEON_YELLOW}Manual intervention may be needed!{constants.RESET}"); return # Exit after error

        else: # No Exit Signal, Manage Position
            lg.info(f"Signal ({signal}) allows holding the existing {pos_side} position. Performing management checks...")

            # --- Time-Based Exit Check ---
            time_exit_minutes_config = config.get("time_based_exit_minutes")
            if time_exit_minutes_config and time_exit_minutes_config > 0:
                if pos_timestamp_ms:
                    try:
                        current_time_ms = time.time() * 1000; time_elapsed_ms = current_time_ms - float(pos_timestamp_ms)
                        time_elapsed_minutes = time_elapsed_ms / (1000 * 60)
                        lg.debug(f"Time-Based Exit Check: Elapsed = {time_elapsed_minutes:.2f} min, Limit = {time_exit_minutes_config} min")
                        if time_elapsed_minutes >= time_exit_minutes_config:
                            lg.warning(f"{constants.NEON_YELLOW}*** TIME-BASED EXIT Triggered ({time_elapsed_minutes:.1f} >= {time_exit_minutes_config} min). Closing position... ***{constants.RESET}")
                            close_side_signal = "SELL" if pos_side == 'long' else "BUY"; size_to_close = abs(pos_size)
                            if size_to_close > 0:
                                close_order = exchange_api.place_trade(exchange, symbol, close_side_signal, size_to_close, market_info, lg, order_type='market', reduce_only=True, quote_currency=QUOTE_CURRENCY)
                                if close_order: lg.info(f"{constants.NEON_GREEN}Time-based CLOSE order placed successfully for {symbol}. ID: {close_order.get('id', 'N/A')}{constants.RESET}")
                                else: lg.error(f"{constants.NEON_RED}Failed to place time-based CLOSE order for {symbol}. Manual check required!{constants.RESET}")
                            else: lg.warning("Time-based exit triggered but position size is zero.")
                            return # Exit after triggering time-based close
                    except (ValueError, TypeError) as time_conv_err: lg.error(f"{constants.NEON_RED}Error converting position timestamp for time check: {pos_timestamp_ms} -> {time_conv_err}{constants.RESET}")
                    except Exception as time_err: lg.error(f"{constants.NEON_RED}Error during time-based exit check: {time_err}{constants.RESET}")
                else: lg.warning("Time-based exit enabled, but position timestamp not found.")

            # --- TSL Active Check ---
            is_tsl_active_exchange = False
            try:
                 tsl_value_str = open_position.get('trailingStopLossValue')
                 if tsl_value_str and str(tsl_value_str).strip() and str(tsl_value_str) != '0':
                      tsl_value = Decimal(str(tsl_value_str))
                      if tsl_value > 0: is_tsl_active_exchange = True; lg.debug("Exchange Trailing Stop Loss appears to be active.")
            except (InvalidOperation, ValueError, TypeError) as tsl_check_err: lg.warning(f"Could not reliably determine if exchange TSL is active (value: {tsl_value_str}): {tsl_check_err}")

            # --- Break-Even Check ---
            if config.get("enable_break_even", False) and not is_tsl_active_exchange:
                lg.debug(f"Checking Break-Even conditions for {symbol}...")
                try:
                    if entry_price is None or entry_price <= 0: raise ValueError("Invalid entry price for BE check")
                    if not isinstance(current_atr, Decimal) or current_atr <= 0: raise ValueError("Invalid ATR for BE check")
                    be_trigger_atr_mult_str = config.get("break_even_trigger_atr_multiple", "1.0"); be_offset_ticks = int(config.get("break_even_offset_ticks", 2))
                    profit_target_atr = Decimal(str(be_trigger_atr_mult_str))
                    price_diff = (current_price - entry_price) if pos_side == 'long' else (entry_price - current_price)
                    profit_in_atr = price_diff / current_atr if current_atr > 0 else Decimal('0')
                    lg.debug(f"BE Check: CurrentPrice={current_price:.{price_precision}f}, Entry={entry_price:.{price_precision}f}")
                    lg.debug(f"BE Check: Price Diff={price_diff:.{price_precision}f}, Profit ATRs={profit_in_atr:.2f}, Target ATRs={profit_target_atr}")

                    if profit_in_atr >= profit_target_atr:
                        if min_tick_size <= 0: raise ValueError("Invalid min_tick_size for BE offset")
                        tick_offset = min_tick_size * be_offset_ticks; be_stop_price: Optional[Decimal] = None
                        if pos_side == 'long': raw_be_stop = entry_price + tick_offset; be_stop_price = (raw_be_stop / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
                        else: raw_be_stop = entry_price - tick_offset; be_stop_price = (raw_be_stop / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size
                        if be_stop_price is None or be_stop_price <= 0: raise ValueError(f"Calculated BE stop price invalid: {be_stop_price}")

                        current_sl_price: Optional[Decimal] = None
                        current_sl_str = open_position.get('stopLossPrice') or open_position.get('info', {}).get('stopLoss')
                        if current_sl_str and str(current_sl_str).strip() and str(current_sl_str) != '0':
                            try: current_sl_price = Decimal(str(current_sl_str))
                            except (InvalidOperation, ValueError, TypeError) as sl_parse_err: lg.warning(f"Could not parse current stop loss '{current_sl_str}': {sl_parse_err}")

                        update_be_sl = False
                        if current_sl_price is None: update_be_sl = True; lg.info("BE triggered: No current SL found. Setting BE SL.")
                        elif pos_side == 'long' and be_stop_price > current_sl_price: update_be_sl = True; lg.info(f"BE triggered: Target BE SL {be_stop_price} is tighter than Current SL {current_sl_price}. Updating.")
                        elif pos_side == 'short' and be_stop_price < current_sl_price: update_be_sl = True; lg.info(f"BE triggered: Target BE SL {be_stop_price} is tighter than Current SL {current_sl_price}. Updating.")
                        else: lg.debug(f"BE Triggered, but current SL ({current_sl_price}) is already better than target ({be_stop_price}). No update needed.")

                        if update_be_sl:
                            lg.warning(f"{constants.NEON_PURPLE}*** Moving Stop Loss to Break-Even for {symbol} at {be_stop_price} ***{constants.RESET}")
                            current_tp_price: Optional[Decimal] = None
                            current_tp_str = open_position.get('takeProfitPrice') or open_position.get('info', {}).get('takeProfit')
                            if current_tp_str and str(current_tp_str).strip() and str(current_tp_str) != '0':
                                 try: current_tp_price = Decimal(str(current_tp_str))
                                 except: pass
                            success = exchange_api._set_position_protection(exchange=exchange, symbol=symbol, market_info=market_info, position_info=open_position, logger=lg, stop_loss_price=be_stop_price, take_profit_price=current_tp_price)
                            if success: lg.info(f"{constants.NEON_GREEN}Break-Even SL set/updated successfully.{constants.RESET}")
                            else: lg.error(f"{constants.NEON_RED}Failed to set/update Break-Even SL.{constants.RESET}")
                    else: lg.debug(f"BE Profit target not reached ({profit_in_atr:.2f} < {profit_target_atr} ATRs).")
                except ValueError as ve: lg.warning(f"BE Check skipped for {symbol}: {ve}")
                except (InvalidOperation, TypeError) as dec_err: lg.error(f"{constants.NEON_RED}Error during break-even check ({symbol}) (Decimal/Type Error): {dec_err}{constants.RESET}", exc_info=False)
                except Exception as be_err: lg.error(f"{constants.NEON_RED}Error during break-even check ({symbol}): {be_err}{constants.RESET}", exc_info=True)
            elif is_tsl_active_exchange: lg.debug(f"Break-even check skipped: Exchange Trailing Stop Loss is active.")
            else: lg.debug(f"Break-even check skipped: Disabled in config.")

    # --- Cycle End Logging ---
    cycle_end_time = time.monotonic()
    lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """Main function to initialize the bot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY

    # Load config first to get timezone and quote currency
    # load_config returns a tuple: (config_dict, timezone_obj)
    CONFIG, timezone_obj = config_loader.load_config(constants.CONFIG_FILE)
    QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")

    # Setup initial logger using the loaded timezone
    init_logger = logger_setup.setup_logger(
        "init",
        log_directory=constants.LOG_DIRECTORY,
        timezone=timezone_obj
    )

    init_logger.info(f"--- Starting XR Scalper Bot ({datetime.now(timezone_obj).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")
    init_logger.info(f"Config loaded from {constants.CONFIG_FILE}. Quote Currency: {QUOTE_CURRENCY}")
    init_logger.info(f"Using Timezone: {timezone_obj}")
    try:
        pandas_ta_version = ta.version() if callable(getattr(ta, 'version', None)) else getattr(ta, 'version', 'N/A')
    except Exception: pandas_ta_version = 'Error getting version'
    init_logger.info(f"Versions: CCXT={ccxt.__version__}, Pandas={pd.__version__}, PandasTA={pandas_ta_version}, Python={sys.version.split()[0]}")

    # --- Trading Enabled Warning ---
    if CONFIG.get("enable_trading"):
         init_logger.warning(f"{constants.NEON_YELLOW}!!! LIVE TRADING IS ENABLED !!!{constants.RESET}")
         if CONFIG.get("use_sandbox"): init_logger.warning(f"{constants.NEON_YELLOW}Using SANDBOX (Testnet) Environment.{constants.RESET}")
         else: init_logger.warning(f"{constants.NEON_RED}!!! CAUTION: USING REAL MONEY ENVIRONMENT !!!{constants.RESET}")
         risk_pct = CONFIG.get('risk_per_trade', 0) * 100; leverage = CONFIG.get('leverage', 0); symbols_to_trade_list = CONFIG.get('symbols_to_trade', [])
         init_logger.info(f"Critical Settings: Risk/Trade={risk_pct:.2f}%, Leverage={leverage}x, Symbols={symbols_to_trade_list}")
         init_logger.info(f"                     TSL Enabled={CONFIG.get('enable_trailing_stop')}, BE Enabled={CONFIG.get('enable_break_even')}")
         init_logger.info("Starting in 5 seconds...")
         time.sleep(5)
    else:
        init_logger.info("Live trading is DISABLED in config. Running in analysis-only mode.")

    # --- Initialize Exchange ---
    exchange = exchange_api.initialize_exchange(CONFIG, init_logger)
    if not exchange: init_logger.error(f"{constants.NEON_RED}Failed to initialize exchange. Exiting bot.{constants.RESET}"); return

    # --- Get Symbols to Trade ---
    symbols_to_trade = CONFIG.get("symbols_to_trade", [])
    if not symbols_to_trade: init_logger.error(f"{constants.NEON_RED}'symbols_to_trade' list is empty in config. Exiting bot.{constants.RESET}"); return

    # --- Main Loop ---
    init_logger.info(f"Starting main trading loop for symbols: {symbols_to_trade}")
    while True:
        loop_start_time = time.monotonic()
        try:
            for symbol in symbols_to_trade:
                safe_symbol_suffix = symbol.replace('/', '_').replace(':', '-')
                symbol_logger = logger_setup.setup_logger(safe_symbol_suffix, constants.LOG_DIRECTORY, timezone_obj)
                try:
                    analyze_and_trade_symbol(exchange, symbol, CONFIG, symbol_logger)
                except Exception as symbol_err:
                    symbol_logger.error(f"{constants.NEON_RED}!!! Unhandled Exception during analysis for {symbol}: {symbol_err} !!!{constants.RESET}", exc_info=True)
                    symbol_logger.error(f"{constants.NEON_YELLOW}Continuing to next symbol or cycle after error.{constants.RESET}")
                finally:
                    # time.sleep(1) # Optional delay between symbols
                    pass

        except KeyboardInterrupt:
            init_logger.info("KeyboardInterrupt received. Shutting down bot...")
            break
        except Exception as loop_err:
            init_logger.error(f"{constants.NEON_RED}!!! Unhandled Exception in main loop: {loop_err} !!!{constants.RESET}", exc_info=True)
            init_logger.warning(f"{constants.NEON_YELLOW}Attempting to continue loop after error...{constants.RESET}")
            time.sleep(constants.LOOP_DELAY_SECONDS * 2)

        # --- Loop Delay ---
        loop_end_time = time.monotonic(); elapsed_time = loop_end_time - loop_start_time
        delay = max(0, constants.LOOP_DELAY_SECONDS - elapsed_time)
        init_logger.debug(f"Loop finished in {elapsed_time:.2f}s. Waiting {delay:.2f}s for next cycle.")
        if delay > 0: time.sleep(delay)

    init_logger.info(f"--- XR Scalper Bot Shutdown Complete ({datetime.now(timezone_obj).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")


if __name__ == "__main__":
    main()

```
