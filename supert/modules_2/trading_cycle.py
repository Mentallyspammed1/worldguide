# File: trading_cycle.py
import time
import traceback
from decimal import Decimal
from typing import Any

# Third-party Libraries
try:
    import ccxt # For type hinting Exchange object
    import pandas as pd
    from colorama import Fore, Back, Style
except ImportError:
    class DummyCCXTExchange: pass
    ccxt = type('ccxt', (), {'Exchange': DummyCCXTExchange})() # type: ignore[call-arg]
    pd = None # type: ignore[assignment]
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""
    Fore, Back, Style = DummyColor(), DummyColor(), DummyColor()

# Custom module imports
from logger_setup import logger
from config import CONFIG
from utils import safe_decimal_conversion, send_sms_alert
from indicator_calculator import (
    calculate_supertrend,
    calculate_stochrsi_momentum,
    calculate_ehlers_fisher,
    calculate_ehlers_ma,
    analyze_volume_atr,
    analyze_order_book,
)
from order_management import (
    get_current_position,
    close_position,
    place_risked_market_order,
    cancel_open_orders,
)
from strategy import generate_signals


def trade_logic(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> None:
    """Executes the main trading logic for one cycle based on selected strategy."""
    if pd is None:
        logger.critical("Pandas library is not available. Cannot execute trade logic.")
        return

    cycle_time_str = df.index[-1].strftime("%Y-%m-%d %H:%M:%S %Z") if not df.empty else "N/A"
    logger.info(
        f"{Fore.BLUE}{Style.BRIGHT}========== New Check Cycle ({CONFIG.strategy_name}): {symbol} | Candle: {cycle_time_str} =========={Style.RESET_ALL}"
    )

    required_rows = max(
        CONFIG.st_atr_length, CONFIG.confirm_st_atr_length,
        CONFIG.stochrsi_rsi_length + CONFIG.stochrsi_stoch_length, CONFIG.momentum_length,
        CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length,
        CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period,
        CONFIG.atr_calculation_period, CONFIG.volume_ma_period,
    ) + 10

    if df is None or len(df) < required_rows:
        logger.warning(f"{Fore.YELLOW}Trade Logic: Insufficient data ({len(df) if df is not None else 0}, need ~{required_rows}). Skipping.{Style.RESET_ALL}")
        return

    action_taken_this_cycle: bool = False
    try:
        logger.debug("Calculating indicators...")
        df = calculate_supertrend(df, CONFIG.st_atr_length, CONFIG.st_multiplier)
        df = calculate_supertrend(df, CONFIG.confirm_st_atr_length, CONFIG.confirm_st_multiplier, prefix="confirm_")
        df = calculate_stochrsi_momentum(df, CONFIG.stochrsi_rsi_length, CONFIG.stochrsi_stoch_length, CONFIG.stochrsi_k_period, CONFIG.stochrsi_d_period, CONFIG.momentum_length)
        df = calculate_ehlers_fisher(df, CONFIG.ehlers_fisher_length, CONFIG.ehlers_fisher_signal_length)
        df = calculate_ehlers_ma(df, CONFIG.ehlers_fast_period, CONFIG.ehlers_slow_period)
        vol_atr_data = analyze_volume_atr(df, CONFIG.atr_calculation_period, CONFIG.volume_ma_period)
        current_atr = vol_atr_data.get("atr")

        last = df.iloc[-1]
        current_price = safe_decimal_conversion(last.get("close"))
        if pd.isna(current_price) or current_price <= CONFIG.position_qty_epsilon:
            logger.warning(f"{Fore.YELLOW}Last candle close price is invalid ({current_price}). Skipping.{Style.RESET_ALL}"); return
        can_place_order = current_atr is not None and current_atr > CONFIG.position_qty_epsilon
        if not can_place_order: logger.warning(f"{Fore.YELLOW}Invalid ATR ({current_atr}). Cannot calculate SL or place new orders.{Style.RESET_ALL}")

        position = get_current_position(exchange, symbol)
        position_side = position["side"]
        ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit) if CONFIG.fetch_order_book_per_cycle else None

        vol_ratio, vol_spike = vol_atr_data.get("volume_ratio"), False
        if vol_ratio is not None: vol_spike = vol_ratio > CONFIG.volume_spike_threshold
        bid_ask_ratio, spread = (ob_data.get("bid_ask_ratio"), ob_data.get("spread")) if ob_data else (None, None)
        logger.info(f"State | Price: {current_price:.4f}, ATR({CONFIG.atr_calculation_period}): {current_atr:.5f}" if current_atr else f"State | Price: {current_price:.4f}, ATR(N/A)")
        logger.info(f"State | Volume: Ratio={vol_ratio:.2f if vol_ratio else 'N/A'}, Spike={vol_spike} (Req={CONFIG.require_volume_spike_for_entry})")
        logger.info(f"State | OrderBook: Ratio={bid_ask_ratio:.3f if bid_ask_ratio else 'N/A'}, Spread={spread:.4f if spread else 'N/A'}")
        logger.info(f"State | Position: Side={position_side}, Qty={position['qty']:.8f}, Entry={position['entry_price']:.4f}")

        strategy_signals = generate_signals(df, CONFIG.strategy_name)
        logger.debug(f"Strategy Signals ({CONFIG.strategy_name}): {strategy_signals}")

        should_exit_long = position_side == CONFIG.pos_long and strategy_signals["exit_long"]
        should_exit_short = position_side == CONFIG.pos_short and strategy_signals["exit_short"]
        if should_exit_long or should_exit_short:
            exit_reason = strategy_signals["exit_reason"]
            logger.warning(f"{Back.YELLOW}{Fore.BLACK}*** TRADE EXIT SIGNAL: Closing {position_side} due to {exit_reason} ***{Style.RESET_ALL}")
            cancel_open_orders(exchange, symbol, f"SL/TSL before {exit_reason} Exit")
            if close_position(exchange, symbol, position, reason=exit_reason): action_taken_this_cycle = True
            if action_taken_this_cycle: logger.info(f"Pausing for {CONFIG.post_close_delay_seconds}s after closing."); time.sleep(CONFIG.post_close_delay_seconds)
            return

        if position_side != CONFIG.pos_none: logger.info(f"Holding {position_side} position. Waiting for SL/TSL or Strategy Exit."); return
        if not can_place_order: logger.warning(f"{Fore.YELLOW}Holding Cash. Cannot enter due to invalid ATR for SL calculation.{Style.RESET_ALL}"); return

        logger.debug("Checking entry signals...")
        potential_entry = strategy_signals["enter_long"] or strategy_signals["enter_short"]
        if not CONFIG.fetch_order_book_per_cycle and potential_entry and ob_data is None:
            logger.debug("Potential entry signal, fetching OB for confirmation...")
            ob_data = analyze_order_book(exchange, symbol, CONFIG.order_book_depth, CONFIG.order_book_fetch_limit)
            bid_ask_ratio = ob_data.get("bid_ask_ratio") if ob_data else None

        ob_check_required = potential_entry
        ob_available = ob_data is not None and bid_ask_ratio is not None
        passes_long_ob = not ob_check_required or (ob_available and bid_ask_ratio >= CONFIG.order_book_ratio_threshold_long)
        passes_short_ob = not ob_check_required or (ob_available and bid_ask_ratio <= CONFIG.order_book_ratio_threshold_short)
        ob_log = f"OB OK (L:{passes_long_ob},S:{passes_short_ob}, Ratio={bid_ask_ratio:.3f if bid_ask_ratio else 'N/A'}, Req={ob_check_required})"
        vol_check_required, passes_volume = CONFIG.require_volume_spike_for_entry, not CONFIG.require_volume_spike_for_entry or vol_spike
        vol_log = f"Vol OK (Pass:{passes_volume}, Spike={vol_spike}, Req={vol_check_required})"

        enter_long = strategy_signals["enter_long"] and passes_long_ob and passes_volume
        enter_short = strategy_signals["enter_short"] and passes_short_ob and passes_volume
        logger.debug(f"Final Entry Check (Long): Strategy={strategy_signals['enter_long']}, {ob_log}, {vol_log} => Enter={enter_long}")
        logger.debug(f"Final Entry Check (Short): Strategy={strategy_signals['enter_short']}, {ob_log}, {vol_log} => Enter={enter_short}")

        common_params = (exchange, symbol, CONFIG.risk_per_trade_percentage, current_atr, CONFIG.atr_stop_loss_multiplier,
                         CONFIG.leverage, CONFIG.max_order_usdt_amount, CONFIG.required_margin_buffer,
                         CONFIG.trailing_stop_percentage, CONFIG.trailing_stop_activation_offset_percent)
        if enter_long:
            logger.success(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED LONG ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}") # type: ignore[attr-defined]
            cancel_open_orders(exchange, symbol, "Before Long Entry")
            if place_risked_market_order(CONFIG.side_buy, *common_params): action_taken_this_cycle = True # type: ignore[misc]
        elif enter_short:
            logger.success(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}*** TRADE SIGNAL: CONFIRMED SHORT ENTRY ({CONFIG.strategy_name}) for {symbol} ***{Style.RESET_ALL}") # type: ignore[attr-defined]
            cancel_open_orders(exchange, symbol, "Before Short Entry")
            if place_risked_market_order(CONFIG.side_sell, *common_params): action_taken_this_cycle = True # type: ignore[misc]
        elif not action_taken_this_cycle: logger.info("No confirmed entry signal. Holding cash.")
    except Exception as e:
        logger.error(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL UNEXPECTED ERROR in trade_logic: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        send_sms_alert(f"[{symbol.split('/')[0]}] CRITICAL ERROR in trade_logic: {type(e).__name__}")
    finally: logger.info(f"{Fore.BLUE}{Style.BRIGHT}========== Cycle Check End: {symbol} =========={Style.RESET_ALL}\n")

# End of trading_cycle.py
```

```python
