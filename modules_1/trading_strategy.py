# File: trading_strategy.py
import logging
# import time # Retained for time.time() if loop.time() is not strictly epoch, or if preferred for epoch.
            # asyncio.get_event_loop().time() is generally preferred in async code.
import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation
from typing import Any, Dict, Optional, Tuple
import sys

import utils
import ccxt
import ccxt.async_support as ccxt_async # Use async_support for type hints
import pandas as pd
import traceback
try:
    from analysis import TradingAnalyzer
    from exchange_api import (fetch_balance, fetch_current_price_ccxt,
                              fetch_klines_ccxt, fetch_orderbook_ccxt,
                              get_market_info, get_open_position, place_trade,
                              set_leverage_ccxt, set_trailing_stop_loss,
                              _set_position_protection) # _set_position_protection is for fixed SL/TP
    from risk_manager import calculate_position_size
    from utils import (CCXT_INTERVAL_MAP, POSITION_CONFIRM_DELAY_SECONDS,
                       get_min_tick_size, get_price_precision,
                       DEFAULT_INDICATOR_PERIODS,
                       NEON_GREEN, NEON_PURPLE, NEON_RED, NEON_YELLOW, RESET_ALL_STYLE, NEON_BLUE, NEON_CYAN) # Use RESET_ALL_STYLE
except ImportError as e:
    _NEON_RED = "\033[1;91m" # Define fallback colors
    _RESET_ALL_STYLE = "\033[0m"
    print(f"{_NEON_RED}CRITICAL ERROR: Failed to import required modules in trading_strategy.py: {e}{_RESET_ALL_STYLE}", file=sys.stderr)
    if 'traceback' not in sys.modules: import traceback # Import if not already imported
    traceback.print_exc(file=sys.stderr)
    raise
except Exception as e:
     _NEON_RED = "\033[1;91m" # Define fallback colors
     _RESET_ALL_STYLE = "\033[0m"
     print(f"{_NEON_RED}CRITICAL ERROR: An unexpected error occurred during module import in trading_strategy.py: {e}{_RESET_ALL_STYLE}", file=sys.stderr)
     if 'traceback' not in sys.modules: import traceback
     traceback.print_exc(file=sys.stderr)
     raise

# Define color constants within this module using imported values
NEON_GREEN = utils.NEON_GREEN
NEON_BLUE = utils.NEON_BLUE
NEON_PURPLE = utils.NEON_PURPLE
NEON_YELLOW = utils.NEON_YELLOW
NEON_RED = utils.NEON_RED
NEON_CYAN = utils.NEON_CYAN
RESET = utils.RESET_ALL_STYLE # Use the correct reset style


# --- Formatting Helpers ---
def _format_signal(signal_text: Any) -> str: # Use Any for signal_text type hint
    """Formats a trading signal (BUY, SELL, HOLD) with color."""
    signal_str = str(signal_text).upper() # Ensure uppercase for comparison
    if signal_str == "BUY": return f"{NEON_GREEN}{signal_str}{RESET}"
    if signal_str == "SELL": return f"{NEON_RED}{signal_str}{RESET}"
    if signal_str == "HOLD": return f"{NEON_YELLOW}{signal_str}{RESET}"
    return f"{signal_str}{RESET}" # Default formatting for unknown signals

def _format_side(side_text: Optional[str]) -> str:
    """Formats position side (LONG, SHORT, UNKNOWN) with color."""
    if side_text is None: return f"{NEON_YELLOW}UNKNOWN{RESET}"
    side_upper = side_text.upper()
    if side_upper == "LONG": return f"{NEON_GREEN}{side_upper}{RESET}"
    if side_upper == "SHORT": return f"{NEON_RED}{side_upper}{RESET}"
    return side_upper # Default formatting

def _format_price_or_na(price_val: Optional[Decimal], precision_places: int, label: str = "") -> str:
    """Formats a price (Decimal) with precision and color, or returns N/A if None/invalid."""
    color = NEON_CYAN
    if price_val is not None and isinstance(price_val, Decimal):
        # Handle 0.0 explicitly if it's a valid Decimal(0)
        if price_val == Decimal(0):
            # If a label is provided, 0.0 might be significant (e.g., SL/TP removed)
             return f"{NEON_YELLOW}0.0{RESET}" if label else f"{color}{price_val:.{precision_places}f}{RESET}"
        if price_val > Decimal(0): # Format any positive value
            try:
                # Ensure formatting handles the Decimal correctly
                return f"{color}{price_val:.{precision_places}f}{RESET}"
            except Exception as e:
                # Fallback formatting if the primary fails
                return f"{NEON_YELLOW}{price_val} (fmt err for {label}: {e}){RESET}"
        # Potentially handle negative values if they are expected and need formatting (e.g., negative PnL)
        # For prices, negative is usually an error state or not applicable.
        return f"{NEON_YELLOW}{price_val} (unexpected value for {label}){RESET}"
    return f"{NEON_YELLOW}N/A{RESET}" # Indicate None or invalid input


# --- Core Trading Logic ---
async def _execute_close_position(
    exchange: ccxt_async.Exchange, # Use async_support Exchange type hint
    symbol: str, market_info: Dict[str, Any],
    open_position: Dict[str, Any], logger: logging.Logger, reason: str = "exit signal"
) -> bool:
    """Executes a market order to close an existing position."""
    lg = logger
    pos_side = open_position.get('side') # Expected 'long' or 'short'
    pos_size = open_position.get('contractsDecimal') # Expected Decimal

    if not pos_side or pos_side not in ['long', 'short']:
        lg.error(f"{NEON_RED}Cannot close position for {symbol}: Invalid position side '{pos_side}'.{RESET}")
        return False
    if not isinstance(pos_size, Decimal) or pos_size <= Decimal('0'): # Check against Decimal('0')
        lg.warning(f"{NEON_YELLOW}Attempted to close {symbol} ({reason}), but size invalid/zero ({pos_size}). No close order.{RESET}")
        return False # Or True if this state means "effectively closed" or no action needed. False implies failure.

    try:
        # Determine the side of the closing order (opposite of the position side)
        close_side_signal = "SELL" if pos_side == 'long' else "BUY"
        
        # Determine amount precision for logging size
        amount_precision = market_info.get('amountPrecision', 8)
        if not (isinstance(amount_precision, int) and amount_precision >= 0): amount_precision = 8

        lg.info(f"{NEON_YELLOW}==> Closing {_format_side(pos_side)} position for {symbol} due to {reason} <==")
        # Log formatted position size using determined precision
        lg.info(f"{NEON_YELLOW}==> Placing {_format_signal(close_side_signal)} MARKET order (reduceOnly=True) | Size: {pos_size:.{amount_precision}f} <==")

        # Place the market order to close the position (reduceOnly=True)
        close_order = await place_trade(
            exchange=exchange, symbol=symbol, trade_signal=close_side_signal,
            position_size=pos_size, market_info=market_info, logger=lg,
            order_type='market', reduce_only=True
        )

        if close_order and close_order.get('id'):
            lg.info(f"{NEON_GREEN}Position CLOSE order placed for {symbol}. Order ID: {close_order['id']}. Status: {close_order.get('status', 'N/A')}{RESET}")
            # Consider adding a small delay and confirming closure if critical, but for market reduce only, it's often near instant.
            # A more robust approach would fetch positions again after a short delay to confirm closure.
            return True
        else:
            lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Placement returned None/no ID.{RESET}")
            lg.warning(f"{NEON_RED}Manual check/intervention required for {symbol}!{RESET}")
            return False
    except Exception as close_err:
         lg.error(f"{NEON_RED}Error attempting to close position for {symbol} ({reason}): {close_err}{RESET}", exc_info=True)
         lg.warning(f"{NEON_RED}Manual intervention may be needed for {symbol}!{RESET}")
         return False


async def _fetch_and_prepare_market_data(
    exchange: ccxt_async.Exchange, # Use async_support Exchange type hint
    symbol: str, config: Dict[str, Any], lg: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Fetches market info, klines, current price, and orderbook."""
    market_info = await get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"{NEON_RED}Failed to get market info for {symbol}. Skipping cycle.{RESET}"); return None

    # Ensure market info contains necessary keys for price/amount precision, fallback if needed
    market_info.setdefault('pricePrecision', get_price_precision(market_info, lg)) # Ensure pricePrecision is set
    market_info.setdefault('amountPrecision', market_info.get('amountPrecision', 8)) # Ensure amountPrecision is set
    market_info.setdefault('minTickSize', get_min_tick_size(market_info, lg)) # Ensure minTickSize is set
    market_info.setdefault('minQty', Decimal(str(market_info.get('limits',{}).get('amount',{}).get('min', '0')))) # Ensure minQty is set

    interval_config_val = config.get("interval")
    if interval_config_val is None:
         lg.error(f"{NEON_RED}Interval not specified for {symbol}. Skipping.{RESET}"); return None
    interval_str = str(interval_config_val)
    ccxt_interval = CCXT_INTERVAL_MAP.get(interval_str)
    if not ccxt_interval:
         lg.error(f"{NEON_RED}Invalid interval '{interval_str}' for {symbol}. Not found in CCXT_INTERVAL_MAP. Skipping.{RESET}"); return None

    kline_limit = config.get("kline_limit", 500)
    # Ensure kline_limit is a positive integer
    if not isinstance(kline_limit, int) or kline_limit <= 0:
        lg.warning(f"{NEON_YELLOW}Invalid kline_limit ({kline_limit}) for {symbol}. Using default 500.{RESET}"); kline_limit = 500

    klines_df = await fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)

    # Use configured min_kline_length, fallback to a safe default
    min_kline_length = config.get("min_kline_length", 100)
    if not isinstance(min_kline_length, int) or min_kline_length <= 0:
         lg.warning(f"{NEON_YELLOW}Invalid min_kline_length ({min_kline_length}) for {symbol}. Using default 100.{RESET}"); min_kline_length = 100

    if klines_df is None or klines_df.empty or len(klines_df) < min_kline_length:
        lg.error(f"{NEON_RED}Insufficient kline data for {symbol} (got {len(klines_df) if klines_df is not None else 0}, need {min_kline_length}). Skipping.{RESET}"); return None

    current_price_decimal: Optional[Decimal] = None
    try:
        current_price_fetch = await fetch_current_price_ccxt(exchange, symbol, lg)
        if current_price_fetch and isinstance(current_price_fetch, Decimal) and current_price_fetch > Decimal('0'): # Check against Decimal('0')
            current_price_decimal = current_price_fetch
        else:
             lg.warning(f"{NEON_YELLOW}Failed ticker price fetch or invalid price ({current_price_fetch}) for {symbol}. Using last kline close.{RESET}")
             if not klines_df.empty and 'close' in klines_df.columns:
                last_close_val = klines_df['close'].iloc[-1]
                # Ensure last_close_val is a valid Decimal before using
                if isinstance(last_close_val, Decimal) and pd.notna(last_close_val) and last_close_val > Decimal('0') :
                    current_price_decimal = last_close_val
                elif pd.notna(last_close_val):
                    try:
                         # Attempt to convert to Decimal if not already
                         temp_dec_price = Decimal(str(last_close_val))
                         if temp_dec_price > Decimal('0'): current_price_decimal = temp_dec_price
                         else: lg.error(f"{NEON_RED}Last kline close value '{last_close_val}' invalid (non-positive) for {symbol}.{RESET}")
                    except: lg.error(f"{NEON_RED}Last kline close value '{last_close_val}' invalid for {symbol}.{RESET}") # Catch conversion errors
                else: lg.error(f"{NEON_RED}Last kline close value is NaN for {symbol}.{RESET}")
             else: lg.error(f"{NEON_RED}Cannot use last kline close: DataFrame empty or 'close' missing for {symbol}.{RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Error fetching/processing current price for {symbol}: {e}{RESET}", exc_info=True)

    # Use price_precision from market_info for logging
    price_precision = market_info.get('pricePrecision', 4) # Fallback if not set in market_info
    if not (current_price_decimal and isinstance(current_price_decimal, Decimal) and current_price_decimal > Decimal('0')): # Final check on current price
         lg.error(f"{NEON_RED}Cannot get valid current price for {symbol} ({current_price_decimal}). Skipping.{RESET}"); return None
    lg.debug(f"Current price for {symbol}: {current_price_decimal:.{price_precision}f}")


    orderbook_data = None
    # Check if orderbook indicator is enabled AND has a non-zero weight
    orderbook_enabled = config.get("indicators",{}).get("orderbook", False)
    active_weights = config.get("weight_sets", {}).get(config.get("active_weight_set", "default"), {})
    orderbook_weight = active_weights.get("orderbook", 0) # Default weight to 0 if not found
    
    # Check if weight is valid Decimal and non-zero
    is_orderbook_weighted = False
    try:
        if Decimal(str(orderbook_weight)) != Decimal('0'):
             is_orderbook_weighted = True
    except InvalidOperation:
         lg.warning(f"{NEON_YELLOW}Invalid weight for 'orderbook' ({orderbook_weight}) for {symbol}. Skipping orderbook fetch.{RESET}")


    if orderbook_enabled and is_orderbook_weighted:
         orderbook_limit = config.get("orderbook_limit", 100)
         # Ensure orderbook_limit is a positive integer
         if not isinstance(orderbook_limit, int) or orderbook_limit <= 0:
             lg.warning(f"{NEON_YELLOW}Invalid orderbook_limit ({orderbook_limit}) for {symbol}. Using default 100.{RESET}"); orderbook_limit = 100

         orderbook_data = await fetch_orderbook_ccxt(exchange, symbol, orderbook_limit, lg)
         if not orderbook_data: lg.warning(f"{NEON_YELLOW}Failed to fetch orderbook for {symbol}, proceeding without.{RESET}")

    # Use determined precision values from market_info
    price_precision = market_info.get('pricePrecision', 4)
    min_tick_size = market_info.get('minTickSize', Decimal('1e-4'))
    amount_precision = market_info.get('amountPrecision', 8)
    min_qty = market_info.get('minQty', Decimal('0'))

    return {
        "market_info": market_info, "klines_df": klines_df, "current_price_decimal": current_price_decimal,
        "orderbook_data": orderbook_data, "price_precision": price_precision,
        "min_tick_size": min_tick_size, "amount_precision": amount_precision, "min_qty": min_qty
    }


def _perform_trade_analysis(
    klines_df: pd.DataFrame, current_price_decimal: Decimal, orderbook_data: Optional[Dict[str, Any]],
    config: Dict[str, Any], market_info: Dict[str, Any], lg: logging.Logger, price_precision: int
) -> Optional[Dict[str, Any]]:
    """Performs trading analysis and generates signals."""
    # Ensure market_info is passed to TradingAnalyzer
    analyzer = TradingAnalyzer(klines_df.copy(), lg, config, market_info)
    
    # Ensure indicator_values were populated
    if not analyzer.indicator_values:
         lg.error(f"{NEON_RED}Indicator calculation failed for {market_info.get('symbol', 'UNKNOWN_SYMBOL')}. Skipping signal generation.{RESET}"); return None # Use market_info['symbol']

    signal = analyzer.generate_trading_signal(current_price_decimal, orderbook_data)
    # Calculate initial TP/SL targets based on current price for analysis summary logging
    _, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price_decimal, signal)
    current_atr: Optional[Decimal] = analyzer.indicator_values.get("ATR")

    lg.info(f"--- {NEON_PURPLE}Analysis Summary ({market_info.get('symbol', 'UNKNOWN_SYMBOL')}){RESET} ---") # Use market_info['symbol']
    # Use price_precision passed from fetch_and_prepare_market_data
    lg.info(f"  Current Price: {NEON_CYAN}{current_price_decimal:.{price_precision}f}{RESET}")
    
    # Log ATR value and period clearly
    analyzer_atr_period = analyzer.config.get('atr_period', DEFAULT_INDICATOR_PERIODS.get('atr_period', 14))
    atr_log_str = f"  ATR ({analyzer_atr_period}): {NEON_YELLOW}N/A{RESET}"
    # Check if current_atr is a valid Decimal and positive before formatting
    if isinstance(current_atr, Decimal) and pd.notna(current_atr) and current_atr > Decimal('0'):
        try:
            # Format ATR with slightly more precision than price, but ensure positive
            atr_log_str = f"  ATR ({analyzer_atr_period}): {NEON_CYAN}{current_atr:.{max(0, price_precision + 2)}f}{RESET}"
        except Exception as fmt_e:
            atr_log_str = f"  ATR ({analyzer_atr_period}): {NEON_YELLOW}{current_atr} (fmt err: {fmt_e}){RESET}"
    # Log if ATR is present but invalid/zero
    elif current_atr is not None : atr_log_str = f"  ATR ({analyzer_atr_period}): {NEON_YELLOW}{current_atr} (invalid/zero){RESET}"
    lg.info(atr_log_str)

    # Log initial TP/SL targets using formatted price helper
    lg.info(f"  Initial SL (sizing): {_format_price_or_na(sl_calc, price_precision, 'SL Calc')}")
    lg.info(f"  Initial TP (target): {_format_price_or_na(tp_calc, price_precision, 'TP Calc')}")

    # Log config settings for position management
    tsl_enabled_config = config.get('enable_trailing_stop', False)
    be_enabled_config = config.get('enable_break_even', False)
    time_exit_minutes_config = config.get('time_based_exit_minutes')
    
    tsl_conf_str = f"{NEON_GREEN}Enabled{RESET}" if tsl_enabled_config else f"{NEON_RED}Disabled{RESET}"
    be_conf_str = f"{NEON_GREEN}Enabled{RESET}" if be_enabled_config else f"{NEON_RED}Disabled{RESET}"
    
    # Log time exit setting clearly
    time_exit_log_str = "Disabled"
    if time_exit_minutes_config is not None:
        try:
            time_exit_val = float(time_exit_minutes_config)
            if time_exit_val > 0: time_exit_log_str = f"{time_exit_val:.1f} min"
        except (ValueError, TypeError):
            lg.warning(f"{NEON_YELLOW}Invalid time_based_exit_minutes config: {time_exit_minutes_config}. Treating as disabled.{RESET}")
    lg.info(f"  Config: TSL={tsl_conf_str}, BE={be_conf_str}, TimeExit={time_exit_log_str}")

    lg.info(f"  Generated Signal: {_format_signal(signal)}")
    lg.info(f"-----------------------------")

    return {
        "signal": signal, "tp_calc": tp_calc, "sl_calc": sl_calc, "analyzer": analyzer,
        "tsl_enabled_config": tsl_enabled_config, "be_enabled_config": be_enabled_config,
        "time_exit_minutes_config": time_exit_minutes_config, "current_atr": current_atr
    }


async def _handle_no_open_position(
    exchange: ccxt_async.Exchange, # Use async_support Exchange type hint
    symbol: str, config: Dict[str, Any], lg: logging.Logger,
    market_data: Dict[str, Any], analysis_results: Dict[str, Any]
):
    """Handles logic when there is no open position and an entry signal is generated."""
    signal = analysis_results["signal"]
    # Only proceed if the signal is BUY or SELL
    if signal not in ["BUY", "SELL"]:
        lg.info(f"Signal is {_format_signal(signal)} and no open position for {symbol}. No entry action.")
        return

    lg.info(f"{NEON_PURPLE}*** {_format_signal(signal)} Signal & No Position: Initiating Trade for {symbol} ***{RESET}")

    # Fetch the latest balance for sizing
    # Use quote_currency from config, fallback to USDT
    quote_currency = config.get("quote_currency", "USDT")
    # Pass params for balance fetch, potentially including accountType for Bybit
    balance_params = {}
    if exchange.id == 'bybit' and exchange.options.get('defaultType', '').lower() == 'unified':
        balance_params['accountType'] = 'UNIFIED'
    # Add any other balance params from config if available
    balance_params.update(config.get('balance_fetch_params', {}))

    balance = await fetch_balance(exchange, quote_currency, lg, params=balance_params)

    # Check if balance is a valid positive Decimal
    if not (balance is not None and isinstance(balance, Decimal) and balance > Decimal('0')):
        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid or non-positive balance ({balance}). Cannot calculate size.{RESET}"); return

    # Get risk_per_trade from config, validate it's a Decimal between 0 and 1
    risk_per_trade_config = config.get("risk_per_trade", 0.0)
    try:
        risk_pct = Decimal(str(risk_per_trade_config))
        if not (Decimal('0') <= risk_pct <= Decimal('1')): # Check against Decimal('0') and Decimal('1')
             raise ValueError("risk_per_trade must be between 0 and 1.")
    except (InvalidOperation, ValueError, TypeError) as e:
         lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid risk_per_trade config ('{risk_per_trade_config}'). Error: {e}{RESET}"); return

    # Calculate the risk amount in quote currency
    risk_amount = balance * risk_pct
    # If risk_pct is > 0, risk_amount should also be > 0. Log warning if not.
    if risk_pct > Decimal('0') and risk_amount <= Decimal('0'):
         lg.warning(f"{NEON_YELLOW}Trade Aborted ({symbol} {signal}): Calculated risk amount non-positive ({risk_amount}). Balance: {balance}, Risk %: {risk_pct}.{RESET}"); return


    # Get initial stop loss price calculated during analysis (used for sizing)
    sl_calc = analysis_results["sl_calc"]
    # Check if sl_calc is a valid positive Decimal
    if not (sl_calc is not None and isinstance(sl_calc, Decimal) and sl_calc > Decimal('0')):
         lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Initial SL invalid ({sl_calc}) for sizing.{RESET}"); return

    # Get market info and current price from market_data
    market_info = market_data["market_info"]
    current_price_decimal = market_data["current_price_decimal"]

    # Check if the market is a contract type (futures/swaps) before setting leverage
    if market_info.get('is_contract', False):
        lev_config = config.get("leverage", 1)
        # Ensure leverage is a positive integer
        if not isinstance(lev_config, int) or lev_config <= 0:
             lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid leverage config ({lev_config}). Must be a positive integer.{RESET}"); return

        # Attempt to set leverage
        if not await set_leverage_ccxt(exchange, symbol, lev_config, market_info, lg):
             lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set leverage {lev_config}x.{RESET}"); return
    else:
         lg.debug(f"Leverage setting skipped for {symbol}: not a contract market.")


    # Calculate the position size based on risk, SL, entry price, and market info
    pos_size_dec = calculate_position_size(balance, risk_pct, sl_calc, current_price_decimal, market_info, exchange, lg)

    # Check if the calculated position size is a valid positive Decimal
    if not (pos_size_dec is not None and isinstance(pos_size_dec, Decimal) and pos_size_dec > Decimal('0')):
        lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Position size calc invalid ({pos_size_dec}).{RESET}"); return

    # Get amount precision and minimum quantity from market_data
    amount_precision = market_data.get("amount_precision", 8) # Fallback precision
    min_qty = market_data.get("min_qty", Decimal('0')) # Fallback min quantity

    # Quantize the position size to the exchange's amount precision
    try:
         # Ensure amount_precision is a non-negative integer for quantization
         if not isinstance(amount_precision, int) or amount_precision < 0: amount_precision = 8
         quant_factor = Decimal('1e-' + str(amount_precision))
         pos_size_dec = pos_size_dec.quantize(quant_factor, ROUND_DOWN) # Quantize down to be conservative
         lg.debug(f"Quantized position size for {symbol}: {pos_size_dec:.{amount_precision}f}")
    except Exception as e:
        lg.error(f"{NEON_RED}Error quantizing size {pos_size_dec} for {symbol}: {e}{RESET}", exc_info=True); return

    # Final check against the minimum quantity allowed by the exchange
    if min_qty > Decimal('0') and pos_size_dec < min_qty:
         lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Final size {pos_size_dec} < MinQty {min_qty}.{RESET}"); return

    # Determine entry order type and price
    entry_type = config.get("entry_order_type", "market").lower()
    limit_px: Optional[Decimal] = None
    min_tick_size = market_data.get("min_tick_size") # Get min tick size from market_data
    price_precision = market_data.get("price_precision", 4) # Fallback price precision

    if entry_type == "limit":
         # Ensure min_tick_size is a valid positive Decimal for limit orders
         if not (min_tick_size is not None and isinstance(min_tick_size, Decimal) and min_tick_size > Decimal('0')):
              lg.warning(f"{NEON_YELLOW}Min tick invalid for limit order {symbol}. Switching to Market.{RESET}"); entry_type = "market"; limit_px = None # Reset limit_px
         else:
             try:
                 # Calculate limit price based on current price and offset percentage from config
                 offset_key = f"limit_order_offset_{signal.lower()}" # e.g., "limit_order_offset_buy"
                 offset_pct_config = config.get(offset_key, "0.0005") # Default offset percentage (as string)
                 offset_pct = Decimal(str(offset_pct_config))
                 if offset_pct < Decimal('0'): raise ValueError("Limit order offset percentage cannot be negative.")

                 # Calculate the raw limit price
                 if signal == "BUY":
                     raw_px = current_price_decimal * (Decimal('1') - offset_pct) # Buy below current price
                 elif signal == "SELL":
                     raw_px = current_price_decimal * (Decimal('1') + offset_pct) # Sell above current price
                 else: # Should not happen due to initial signal check, but defensive
                     raise ValueError(f"Invalid signal '{signal}' for limit price calculation.")

                 # Quantize the raw limit price to the nearest tick size
                 # Round down for BUY limit, round up for SELL limit
                 rnd_mode = ROUND_DOWN if signal == "BUY" else ROUND_UP
                 limit_px = (raw_px / min_tick_size).quantize(Decimal('1'), rounding=rnd_mode) * min_tick_size

                 # Final check on the calculated limit price
                 if not (limit_px is not None and isinstance(limit_px, Decimal) and limit_px > Decimal('0')):
                     raise ValueError(f"Calculated Limit price invalid or non-positive: {limit_px}")
                 
                 # Log the calculated limit price using formatted price helper
                 lg.info(f"Calculated Limit Entry for {signal} on {symbol}: {_format_price_or_na(limit_px, price_precision, 'Limit Px')}")

             except Exception as e_lim:
                 lg.error(f"{NEON_RED}Error calc limit price for {symbol} ({signal}): {e_lim}. Switching to Market.{RESET}",exc_info=False); entry_type="market";limit_px=None # Ensure limit_px is None on error

    # Log the trade placement details
    # Use formatted price helper for limit price in log message
    limit_px_log_label = f"Limit Px {signal}"
    lg.info(f"{NEON_YELLOW}==> Placing {_format_signal(signal)} {entry_type.upper()} | Size: {pos_size_dec:.{amount_precision}f}"
            f"{f' @ {_format_price_or_na(limit_px, price_precision, limit_px_log_label)}' if limit_px is not None else ''} <==") # Check if limit_px is not None

    # Place the trade order
    # Pass limit_price as float or None to place_trade as expected by CCXT
    limit_price_for_api = float(limit_px) if limit_px is not None else None

    trade_order = await place_trade(exchange,symbol,signal,pos_size_dec,market_info,lg,entry_type,limit_price_for_api,False)

    # Check if the order placement was successful (returned a dictionary with an 'id')
    if not (trade_order and isinstance(trade_order, dict) and trade_order.get('id')):
        lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {_format_signal(signal)}). No order ID returned. ==={RESET}")
        return

    order_id = trade_order['id']
    order_status = trade_order.get('status', 'unknown')
    lg.info(f"{NEON_GREEN}Order placed for {symbol}: ID={order_id}, Status={order_status}{RESET}")

    # If it was a market order or a limit order that filled immediately ('closed'),
    # wait for confirmation and set position protection (SL/TP/TSL)
    if entry_type == 'market' or (entry_type == 'limit' and order_status == 'closed'):
        confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
        # Ensure confirm_delay is a non-negative number
        if not isinstance(confirm_delay, (int, float)) or confirm_delay < 0:
             lg.warning(f"{NEON_YELLOW}Invalid position_confirm_delay_seconds ({confirm_delay}). Using default {POSITION_CONFIRM_DELAY_SECONDS}s.{RESET}"); confirm_delay = POSITION_CONFIRM_DELAY_SECONDS

        lg.info(f"Waiting {confirm_delay}s for position confirmation ({symbol})...")
        await asyncio.sleep(confirm_delay)

        # Fetch the open position to get actual entry price and position details
        confirmed_pos = await get_open_position(exchange, symbol, market_info, lg)

        # Check if an open position was successfully confirmed
        if not confirmed_pos:
            lg.error(f"{NEON_RED}Order {order_id} ({entry_type}) for {symbol} reported filled/placed, but FAILED TO CONFIRM open position! Manual check!{RESET}")
            return # Do not proceed with protection setup if position not confirmed

        lg.info(f"{NEON_GREEN}Position Confirmed for {symbol} after {entry_type.capitalize()} Order!{RESET}")

        # Set position protection (SL, TP, TSL) based on confirmed position details
        try:
            # Get the actual entry price from the confirmed position
            entry_px_actual = confirmed_pos.get('entryPriceDecimal')

            # If the actual entry price is invalid/missing, use the limit price (if limit order) or current price as fallback
            if not (entry_px_actual is not None and isinstance(entry_px_actual, Decimal) and entry_px_actual > Decimal('0')):
                fallback_price = None
                # If it was a limit order and we have a valid limit price
                if entry_type == 'limit' and limit_px is not None and isinstance(limit_px, Decimal) and limit_px > Decimal('0'):
                    fallback_price = limit_px
                    fallback_price_info = f"limit price ({_format_price_or_na(limit_px, price_precision)})"
                # Otherwise, use the current market price fetched earlier as a last resort fallback
                elif current_price_decimal is not None and isinstance(current_price_decimal, Decimal) and current_price_decimal > Decimal('0'):
                    fallback_price = current_price_decimal
                    fallback_price_info = f"initial estimated price ({_format_price_or_na(current_price_decimal, price_precision)})"
                else:
                    # If all price sources are invalid, we cannot set protection reliably
                    lg.error(f"{NEON_RED}Could not get valid actual or fallback entry price for {symbol}. Cannot set position protection.{RESET}"); return

                entry_px_actual = fallback_price # Use the determined fallback price
                lg.warning(f"{NEON_YELLOW}Could not get valid actual entry price for {symbol}. Using {fallback_price_info} for protection calculations.{RESET}")

            # Ensure entry_px_actual is a valid positive Decimal before calculation
            if not (entry_px_actual is not None and isinstance(entry_px_actual, Decimal) and entry_px_actual > Decimal('0')):
                 lg.error(f"{NEON_RED}Cannot determine valid entry price for protection setup for {symbol} ({entry_px_actual}). Aborting protection setup.{RESET_ALL_STYLE}"); return


            # Recalculate TP/SL targets using the actual confirmed entry price
            analyzer: TradingAnalyzer = analysis_results["analyzer"] # Get the analyzer instance from analysis results
            _, tp_f, sl_f = analyzer.calculate_entry_tp_sl(entry_px_actual, signal)

            # Format calculated TP/SL for logging
            tp_f_s = _format_price_or_na(tp_f, price_precision, "Final TP")
            sl_f_s = _format_price_or_na(sl_f, price_precision, "Final SL")

            # Determine which protection method to use based on config (TSL or Fixed SL/TP)
            tsl_enabled_config = analysis_results.get("tsl_enabled_config", False) # Get TSL enabled config

            prot_ok = False # Flag to track if protection was set successfully
            if tsl_enabled_config:
                 # If TSL is enabled, attempt to set/update Trailing Stop Loss
                 lg.info(f"Setting TSL for {symbol} (TP target: {tp_f_s})...")
                 # Pass the calculated final TP target to set_trailing_stop_loss
                 prot_ok = await set_trailing_stop_loss(exchange,symbol,market_info,confirmed_pos,config,lg,tp_f)
            # If TSL is NOT enabled, attempt to set Fixed SL and/or TP
            elif (sl_f is not None and isinstance(sl_f, Decimal) and sl_f >= Decimal('0')) or \
                 (tp_f is not None and isinstance(tp_f, Decimal) and tp_f >= Decimal('0')): # Allow 0 to remove existing
                 # Only set fixed SL/TP if at least one of them is a valid Decimal >= 0
                 lg.info(f"Setting Fixed SL ({sl_f_s}) and TP ({tp_f_s}) for {symbol}...")
                 # Pass calculated final SL and TP targets to _set_position_protection
                 prot_ok = await _set_position_protection(exchange,symbol,market_info,confirmed_pos,lg,sl_f,tp_f)
            else:
                 # If TSL is disabled and no valid fixed SL/TP targets were calculated
                 lg.debug(f"No valid SL/TP for fixed protection for {symbol}, and TSL disabled. No protection action taken."); prot_ok = True # Consider successful as nothing was intended to be set

            # Log the outcome of the protection setup
            if prot_ok: lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({symbol} {_format_signal(signal)}) ==={RESET}")
            else: lg.error(f"{NEON_RED}=== TRADE ({symbol} {_format_signal(signal)}) placed BUT FAILED TO SET PROTECTION ==={RESET}"); lg.warning(f"{NEON_RED}>>> MANUAL MONITORING REQUIRED! <<<")

        except Exception as post_err:
            # Catch any errors during post-trade protection setup
            lg.error(f"{NEON_RED}Error during post-trade protection setup for {symbol}: {post_err}{RESET}", exc_info=True);
            lg.warning(f"{NEON_RED}Position open but protection failed. Manual check required for {symbol}!{RESET}")

    # If it was a limit order that is still 'open', log that we are waiting for fill
    elif entry_type == 'limit' and order_status == 'open':
         lg.info(f"{NEON_YELLOW}Limit order {order_id} for {symbol} OPEN @ {_format_price_or_na(limit_px, price_precision, f'Limit Px {signal}')}. Waiting for fill.{RESET}")
         # No protection is set until the limit order is filled and a position is confirmed.
    else:
        # Log if the order status is not 'closed' or 'open' as expected
        lg.error(f"{NEON_RED}Limit order {order_id} for {symbol} has unexpected status: {order_status}. Trade not open as expected.{RESET}")


async def _manage_existing_open_position(
    exchange: ccxt_async.Exchange, # Use async_support Exchange type hint
    symbol: str, config: Dict[str, Any], lg: logging.Logger,
    market_data: Dict[str, Any], analysis_results: Dict[str, Any],
    open_position: Dict[str, Any], loop: asyncio.AbstractEventLoop
):
    """Manages an existing open position based on signals, time exit, and break-even/TSL logic."""
    # Get position details from the fetched open_position dictionary
    pos_side = open_position.get('side')
    pos_size_dec = open_position.get('contractsDecimal')
    entry_px_dec = open_position.get('entryPriceDecimal')
    pos_ts_ms = open_position.get('timestamp_ms') # Position entry timestamp in milliseconds (epoch)

    # Get market info, current price, precision, and ATR from market_data and analysis_results
    market_info = market_data["market_info"]
    current_price_decimal = market_data["current_price_decimal"]
    price_precision = market_data.get("price_precision", 4) # Fallback precision
    amount_precision = market_data.get("amount_precision", 8) # Fallback precision
    min_tick_size = market_data.get("min_tick_size") # Min tick size
    current_atr = analysis_results.get("current_atr") # Current ATR value

    # Validate essential position details before proceeding
    if not (pos_side is not None and isinstance(pos_side, str) and pos_side in ['long','short'] and
            pos_size_dec is not None and isinstance(pos_size_dec,Decimal) and pos_size_dec > Decimal('0') and # Check against Decimal('0')
            entry_px_dec is not None and isinstance(entry_px_dec,Decimal) and entry_px_dec > Decimal('0')): # Check against Decimal('0')
         lg.error(f"{NEON_RED}Cannot manage {symbol}: Invalid pos details. Side='{pos_side}',Size='{pos_size_dec}',Entry='{entry_px_dec}'. Aborting management for this cycle.{RESET}"); return

    # Get quote currency precision for logging PnL, fallback if needed
    quote_prec = config.get('quote_currency_precision', market_info.get('quotePrecision', 2))
    # Get quote currency symbol for logging PnL
    quote_currency_symbol = market_info.get('quote', config.get('quote_currency', 'USD')) # Fallback quote currency

    lg.info(f"{NEON_BLUE}--- Managing Position ({NEON_PURPLE}{symbol}{NEON_BLUE}) ---{RESET}")
    lg.info(f"  Side: {_format_side(pos_side)}, Size: {NEON_CYAN}{pos_size_dec:.{amount_precision}f}{RESET}, Entry: {_format_price_or_na(entry_px_dec, price_precision, 'Entry Px')}")
    # Log other position details fetched from the exchange
    lg.info(f"  MarkPx: {_format_price_or_na(open_position.get('markPriceDecimal'), price_precision, 'Mark Px')}, LiqPx: {_format_price_or_na(open_position.get('liquidationPriceDecimal'), price_precision, 'Liq Px')}")
    # Log unrealized PnL
    lg.info(f"  uPnL: {_format_price_or_na(open_position.get('unrealizedPnlDecimal'), quote_prec, 'uPnL')} {quote_currency_symbol}")
    # Log exchange-set SL and TP
    lg.info(f"  Exchange SL: {_format_price_or_na(open_position.get('stopLossPriceDecimal'), price_precision, 'Exch SL')}, TP: {_format_price_or_na(open_position.get('takeProfitPriceDecimal'), price_precision, 'Exch TP')}")
    # Log exchange-set TSL details
    # Check for trailingStopLossValue which can be distance or price depending on exchange/API version
    tsl_active_val_raw = open_position.get('trailingStopLossValue')
    tsl_act_price_raw = open_position.get('trailingStopActivationPrice')
    
    tsl_val_formatted = _format_price_or_na(tsl_active_val_raw, price_precision, 'TSL Val') # Format the value, assume price-like if numeric
    tsl_act_price_formatted = _format_price_or_na(tsl_act_price_raw, price_precision, 'TSL Act Px') # Format activation price

    # Determine if exchange TSL appears active based on available fields/values
    is_tsl_exch_active = False
    if tsl_active_val_raw is not None and str(tsl_active_val_raw).strip() and str(tsl_active_val_raw) != '0':
        try:
             # Try to convert to Decimal and check if positive, or check if it's a non-empty string (like "0" for inactive)
            if (isinstance(tsl_active_val_raw, Decimal) and tsl_active_val_raw > Decimal('0')) or \
               (isinstance(tsl_active_val_raw, str) and tsl_active_val_raw.replace('.', '', 1).isdigit() and Decimal(tsl_active_val_raw) > Decimal('0')):
                 is_tsl_exch_active = True
        except: pass # Ignore conversion errors, assume not active if cannot parse as positive number

    lg.info(f"  Exchange TSL: Active={is_tsl_exch_active}, Value={tsl_val_formatted}, Activation Price={tsl_act_price_formatted}")


    signal = analysis_results["signal"]
    # Check if the generated signal indicates exiting the current position
    if (pos_side == 'long' and signal == "SELL") or (pos_side == 'short' and signal == "BUY"):
        lg.warning(f"{NEON_YELLOW}*** EXIT Signal ({_format_signal(signal)}) opposes {_format_side(pos_side)} position for {symbol}. Closing... ***{RESET}")
        # Execute the close position logic
        await _execute_close_position(exchange, symbol, market_info, open_position, lg, "opposing signal"); return

    # If the signal does not require exiting, proceed with position management (Time Exit, BE, TSL)
    lg.info(f"Signal ({_format_signal(signal)}) allows holding. Position management for {symbol}...")

    # --- Time-Based Exit Check ---
    time_exit_minutes_config = analysis_results["time_based_exit_minutes_config"]
    # Check if time exit is enabled in config and position timestamp is available
    if time_exit_minutes_config is not None and isinstance(time_exit_minutes_config, (int, float)) and time_exit_minutes_config > 0:
        # Ensure pos_ts_ms is a valid integer timestamp
        if pos_ts_ms is not None and isinstance(pos_ts_ms, int):
            try:
                # Get current time in milliseconds using the event loop's time (preferred in asyncio)
                current_time_ms = int(loop.time() * 1000)
                # Calculate elapsed time in minutes
                elapsed_min = (current_time_ms - pos_ts_ms) / 60000.0
                lg.debug(f"Time Exit Check ({symbol}): Elapsed={elapsed_min:.2f}m, Limit={time_exit_minutes_config}m")
                # If elapsed time meets or exceeds the configured limit, trigger exit
                if elapsed_min >= time_exit_minutes_config:
                    lg.warning(f"{NEON_YELLOW}*** TIME-BASED EXIT for {symbol} ({elapsed_min:.1f} >= {time_exit_minutes_config}m). Closing... ***{RESET}")
                    # Execute the close position logic
                    await _execute_close_position(exchange, symbol, market_info, open_position, lg, "time-based exit"); return
            except Exception as terr:
                 # Catch any errors during time calculation/check
                 lg.error(f"{NEON_RED}Time exit check error for {symbol}: {terr}{RESET}", exc_info=True)
        else:
            # Log warning if time exit is enabled but position timestamp is invalid/missing
            lg.warning(f"{NEON_YELLOW}Time exit enabled for {symbol} but position timestamp invalid/missing ({pos_ts_ms}). Cannot perform check.{RESET}")


    # --- Break-Even Stop Loss (BE) Check ---
    be_enabled_config = analysis_results.get("be_enabled_config", False) # Get BE enabled config
    # Only perform BE check if enabled and TSL is NOT currently active on the exchange
    if be_enabled_config and not is_tsl_exch_active:
        lg.info(f"{NEON_BLUE}--- Break-Even Check ({NEON_PURPLE}{symbol}{NEON_BLUE}) ---{RESET}")
        try:
            # Ensure current ATR is a valid positive Decimal for BE calculation
            if not (current_atr is not None and isinstance(current_atr, Decimal) and current_atr > Decimal('0')):
                 lg.warning(f"{NEON_YELLOW}BE check skipped for {symbol}: Current ATR invalid ({current_atr}).{RESET}")
            # Ensure min_tick_size is valid for calculating BE offset
            elif not (min_tick_size is not None and isinstance(min_tick_size, Decimal) and min_tick_size > Decimal('0')):
                 lg.warning(f"{NEON_YELLOW}BE check skipped for {symbol}: Min tick size invalid ({min_tick_size}).{RESET}")
            else:
                # Get BE trigger ATR multiple and offset ticks from config
                be_trig_atr_config = config.get("break_even_trigger_atr_multiple", "1.0") # Default as string
                be_off_ticks_config = config.get("break_even_offset_ticks", 2) # Default as integer

                # Validate BE config values
                try:
                    be_trig_atr = Decimal(str(be_trig_atr_config))
                    if be_trig_atr < Decimal('0'): raise ValueError("must be non-negative.")
                except (InvalidOperation, ValueError, TypeError):
                    lg.error(f"{NEON_RED}Invalid break_even_trigger_atr_multiple config: '{be_trig_atr_config}'. Using default 1.0.{RESET}"); be_trig_atr = Decimal('1.0')

                try:
                    be_off_ticks = int(be_off_ticks_config)
                    if be_off_ticks < 0: raise ValueError("must be non-negative.")
                except (ValueError, TypeError):
                    lg.error(f"{NEON_RED}Invalid break_even_offset_ticks config: '{be_off_ticks_config}'. Using default 2.{RESET}"); be_off_ticks = 2


                # Calculate current profit in ATRs
                px_diff = (current_price_decimal - entry_px_dec) if pos_side == 'long' else (entry_px_dec - current_price_decimal)
                # Avoid division by zero if ATR is somehow zero (already warned above, but double check)
                profit_atr = px_diff / current_atr if current_atr is not None and current_atr > Decimal('0') else Decimal('inf')

                # Log BE status
                lg.info(f"  BE Status ({symbol}): PxDiff={_format_price_or_na(px_diff, price_precision+2, 'PxDiff')}, ProfitATRs={profit_atr:.2f}, TargetATRs={be_trig_atr:.2f}")

                # Check if the profit target for BE trigger is met
                if profit_atr >= be_trig_atr:
                    lg.info(f"  {NEON_GREEN}BE Trigger Met for {symbol}! Calculating BE stop.{RESET}")

                    # Calculate the break-even stop price
                    tick_off = min_tick_size * Decimal(be_off_ticks)
                    raw_be_px = entry_px_dec + tick_off if pos_side == 'long' else entry_px_dec - tick_off

                    # Quantize the raw BE price to the nearest tick size
                    # Round up for long positions, down for short, to ensure it's at or slightly better than entry
                    rnd_mode = ROUND_UP if pos_side == 'long' else ROUND_DOWN
                    be_px = (raw_be_px / min_tick_size).quantize(Decimal('1'), rounding=rnd_mode) * min_tick_size

                    # Final check on calculated BE price
                    if not (be_px is not None and isinstance(be_px, Decimal) and be_px > Decimal('0')): # Check against Decimal('0')
                        lg.error(f"  {NEON_RED}Calculated BE stop invalid for {symbol}: {be_px}. Cannot set.{RESET}")
                    # Additional check to ensure BE price is actually at or beyond entry
                    elif (pos_side == 'long' and be_px < entry_px_dec) or (pos_side == 'short' and be_px > entry_px_dec):
                         lg.warning(f"  {NEON_YELLOW}Calculated BE stop {be_px} is worse than entry {entry_px_dec} for {pos_side} {symbol}. Adjusting to entry.{RESET}")
                         # As a fallback, set BE to exactly entry price quantized
                         be_px = (entry_px_dec / min_tick_size).quantize(Decimal('1'), rounding=rnd_mode) * min_tick_size
                         if be_px <= Decimal('0'): # Safety check again
                             lg.error(f"{NEON_RED}Adjusted BE stop non-positive ({be_px}) for {symbol}. Cannot set.{RESET}"); be_px = None # Invalidate BE price

                    if be_px is not None:
                         lg.info(f"  Target BE Stop Price for {symbol}: {NEON_CYAN}{be_px:.{price_precision}f}{RESET}")

                         # Get the current stop loss price from the open position details
                         cur_sl_dec = open_position.get('stopLossPriceDecimal')

                         # Determine if the BE stop needs to be updated on the exchange
                         upd_be_sl = False
                         # Update if there's no current SL set, or if the calculated BE price is better than the current SL
                         if not (cur_sl_dec is not None and isinstance(cur_sl_dec, Decimal)): # Check if cur_sl_dec is None or not a Decimal
                            upd_be_sl = True; lg.info(f"  BE ({symbol}): No valid current SL ({cur_sl_dec}). Setting BE SL.")
                         # Check if calculated BE price is better than current SL based on position side
                         elif (pos_side=='long' and be_px > cur_sl_dec) or (pos_side=='short' and be_px < cur_sl_dec):
                            upd_be_sl=True; lg.info(f"  BE ({symbol}): Target {_format_price_or_na(be_px, price_precision, 'BE Px')} better than Current {_format_price_or_na(cur_sl_dec, price_precision, 'Cur SL')}. Updating.")
                         else: lg.debug(f"  BE ({symbol}): Current SL {_format_price_or_na(cur_sl_dec, price_precision, 'Cur SL')} already better/equal or calculated BE is worse.")

                         # If an update is needed, call the function to set/update position protection
                         if upd_be_sl:
                            lg.warning(f"{NEON_YELLOW}*** Moving SL to Break-Even for {symbol} at {be_px:.{price_precision}f} ***{RESET}")
                            cur_tp_dec = open_position.get('takeProfitPriceDecimal') # Preserve existing TP if any
                            # Call _set_position_protection with the new BE price for SL and existing TP
                            if await _set_position_protection(exchange,symbol,market_info,open_position,lg,be_px,cur_tp_dec):
                                lg.info(f"{NEON_GREEN}BE SL set/updated successfully for {symbol}.{RESET}")
                            else: lg.error(f"{NEON_RED}Failed to set/update BE SL for {symbol}. Manual check!{RESET}")
                    # If be_px was None due to calculation/validation failure, an error was already logged.
                else: lg.info(f"  BE Profit target not reached for {symbol} ({profit_atr:.2f} < {be_trig_atr:.2f} ATRs). No BE action.")
        except Exception as be_e:
            # Catch any errors during the BE calculation/update process
            lg.error(f"{NEON_RED}Error in BE check ({symbol}): {be_e}{RESET}", exc_info=True)
    # Log if BE check was skipped because TSL is active on exchange
    elif is_tsl_exch_active: lg.debug(f"BE check skipped for {symbol}: TSL active on exchange.")
    # Log if BE check was skipped because BE is disabled in config
    else: lg.debug(f"BE check skipped for {symbol}: BE disabled in config.")


    # --- Trailing Stop Loss (TSL) Check ---
    tsl_enabled_config = analysis_results.get("tsl_enabled_config", False) # Get TSL enabled config
    # Only attempt to set/update TSL if enabled in config and NOT currently active on the exchange
    if tsl_enabled_config and not is_tsl_exch_active:
        lg.info(f"{NEON_BLUE}--- Trailing Stop Loss Check ({NEON_PURPLE}{symbol}{NEON_BLUE}) ---{RESET}")
        lg.info(f"  Attempting to set/update TSL (enabled & not active on exch).")

        # Calculate the TP target that TSL might trail towards (optional, depends on strategy)
        # Here, using entry_px_dec and current pos_side to determine a relevant TP target for TSL.
        # The calculated TP from analysis_results["tp_calc"] was based on initial price,
        # recalculate using the actual entry price from confirmed position.
        analyzer: TradingAnalyzer = analysis_results["analyzer"] # Get the analyzer instance
        _, tsl_tp_target, _ = analyzer.calculate_entry_tp_sl(entry_px_dec, pos_side) # pos_side is 'long' or 'short'

        # Call set_trailing_stop_loss to handle the TSL logic
        # This function itself handles the logic of calculating TSL parameters and calling _set_position_protection
        if await set_trailing_stop_loss(exchange, symbol, market_info, open_position, config, lg, tsl_tp_target):
            lg.info(f"  {NEON_GREEN}TSL setup/update initiated successfully for {symbol}.{RESET}")
        else:
            lg.warning(f"  {NEON_YELLOW}Failed to initiate TSL setup/update for {symbol}.{RESET}")
    # Log if TSL check was skipped because it's already active on exchange
    elif tsl_enabled_config and is_tsl_exch_active:
        lg.debug(f"TSL enabled but already appears active on exchange for {symbol}. No TSL action needed from bot this cycle.")
    # Log if TSL check was skipped because TSL is disabled in config
    else: lg.debug(f"TSL check skipped for {symbol}: TSL disabled in config.")

    lg.info(f"{NEON_CYAN}------------------------------------{RESET}")


async def analyze_and_trade_symbol(
    exchange: ccxt_async.Exchange, # Use async_support Exchange type hint
    symbol: str, config: Dict[str, Any],
    logger: logging.Logger, enable_trading: bool
) -> None:
    """
    Analyzes a single trading symbol and executes trades if enabled.

    Fetches market data, performs analysis, generates a signal, and
    manages positions or places trades based on the signal and config.
    """
    lg = logger # Use the symbol-specific logger passed in
    loop = asyncio.get_event_loop() # Get the running event loop for time/async operations
    cycle_start_time = loop.time() # Get the start time of the cycle for this symbol

    # Fetch and prepare all necessary market data for the symbol
    market_data = await _fetch_and_prepare_market_data(exchange, symbol, config, lg)
    if not market_data:
        # If data fetch failed, log and end the cycle for this symbol
        lg.info(f"{NEON_BLUE}---== Analysis Cycle End ({NEON_PURPLE}{symbol}{NEON_BLUE}, {loop.time() - cycle_start_time:.2f}s, Data Fetch Failed) ==---{RESET}\n")
        return

    # Perform trading analysis and generate a signal
    # Pass relevant data and configuration to the analysis function
    analysis_results = _perform_trade_analysis(
        market_data["klines_df"], market_data["current_price_decimal"], market_data["orderbook_data"],
        config, market_data["market_info"], lg, market_data["price_precision"] # Pass price_precision for formatting
    )
    if not analysis_results:
        # If analysis failed, log and end the cycle for this symbol
        lg.info(f"{NEON_BLUE}---== Analysis Cycle End ({NEON_PURPLE}{symbol}{NEON_BLUE}, {loop.time() - cycle_start_time:.2f}s, Analysis Failed) ==---{RESET}\n")
        return

    # If trading is disabled, only perform analysis and log
    if not enable_trading:
        lg.debug(f"Trading disabled. Analysis complete for {symbol}.")
        lg.info(f"{NEON_BLUE}---== Analysis Cycle End ({NEON_PURPLE}{symbol}{NEON_BLUE}, {loop.time() - cycle_start_time:.2f}s, Trading Disabled) ==---{RESET}\n")
        return

    # If trading is enabled, fetch the current open position for this symbol
    open_position = await get_open_position(exchange, symbol, market_data["market_info"], lg)

    # Based on whether there's an open position, either handle entry or manage the existing position
    if open_position is None:
        # If no open position, handle entry if the signal is BUY or SELL
        await _handle_no_open_position(exchange, symbol, config, lg, market_data, analysis_results)
    else:
        # If there is an open position, manage it (check time exit, BE, TSL, opposing signal)
        # Pass the event loop object for time calculations if needed (e.g., for time exit)
        await _manage_existing_open_position(exchange, symbol, config, lg, market_data, analysis_results, open_position, loop)

    # Log the end of the analysis cycle for this symbol, including duration
    lg.info(f"{NEON_BLUE}---== Analysis Cycle End ({NEON_PURPLE}{symbol}{NEON_BLUE}, {loop.time() - cycle_start_time:.2f}s) ==---{RESET}\n")