#!/usr/bin/env python

"""Ehlers Volumetric Trend Strategy for Bybit V5

This script implements a trading strategy based on the Ehlers Volumetric Trend
indicator using the Bybit V5 API via CCXT. It leverages custom helper modules
for exchange interaction, indicator calculation, logging, and utilities.

Strategy Logic:
- Fetches OHLCV data for the specified symbol and timeframe.
- Calculates the Ehlers Volumetric Trend (EVT) indicator using VWMA and SuperSmoother.
- Enters LONG when a bullish EVT signal (`evt_buy`) occurs.
- Enters SHORT when a bearish EVT signal (`evt_sell`) occurs.
- Exits positions when the EVT trend reverses.
- Uses ATR-based stop-loss.
- Manages position size based on risk percentage.
- Includes error handling, retries, and rate limit awareness via helper modules.
"""

import logging
import os
import sys
import time
from decimal import ROUND_DOWN, Decimal

# Third-party libraries
import ccxt
import pandas as pd
from dotenv import load_dotenv

# --- Import Custom Modules ---
# Assuming helper files are in the same directory or Python path
try:
    # Logging Setup
    # Exchange Interaction Helpers
    import bybit_helpers as bybit

    # Indicator Calculations
    import indicators as ind

    # Utility Functions (Decimal, Formatting, Retry Decorator, SMS)
    from bybit_utils import (
        format_amount,
        format_order_id,
        format_price,
        retry_api_call,
        safe_decimal_conversion,
        send_sms_alert,
    )
    from neon_logger import setup_logger
except ImportError as e:
    print(f"Error importing helper modules: {e}")
    print(
        "Ensure bybit_helpers.py, indicators.py, neon_logger.py, and bybit_utils.py are accessible."
    )
    sys.exit(1)

# --- Load Environment Variables ---
load_dotenv()


# --- Configuration Class ---
class Config:
    def __init__(self):
        # Exchange & API
        self.EXCHANGE_ID: str = "bybit"
        self.API_KEY: str | None = os.getenv("BYBIT_API_KEY")
        self.API_SECRET: str | None = os.getenv("BYBIT_API_SECRET")
        self.TESTNET_MODE: bool = (
            os.getenv("BYBIT_TESTNET_MODE", "true").lower() == "true"
        )
        self.DEFAULT_RECV_WINDOW: int = int(os.getenv("DEFAULT_RECV_WINDOW", 10000))

        # Symbol & Market
        self.SYMBOL: str = os.getenv(
            "SYMBOL", "BTC/USDT:USDT"
        )  # Example: BTC/USDT Perpetual
        self.USDT_SYMBOL: str = "USDT"
        self.EXPECTED_MARKET_TYPE: str = "swap"
        self.EXPECTED_MARKET_LOGIC: str = "linear"
        self.TIMEFRAME: str = os.getenv("TIMEFRAME", "5m")
        self.OHLCV_LIMIT: int = int(
            os.getenv("OHLCV_LIMIT", 200)
        )  # Candles for indicators

        # Account & Position Settings
        self.DEFAULT_LEVERAGE: int = int(os.getenv("LEVERAGE", 10))
        self.DEFAULT_MARGIN_MODE: str = "cross"  # Or 'isolated'
        self.DEFAULT_POSITION_MODE: str = "one-way"  # Or 'hedge'
        self.RISK_PER_TRADE: Decimal = Decimal(
            os.getenv("RISK_PER_TRADE", "0.01")
        )  # 1% risk

        # Order Settings
        self.DEFAULT_SLIPPAGE_PCT: Decimal = Decimal(
            os.getenv("DEFAULT_SLIPPAGE_PCT", "0.005")
        )  # 0.5%
        self.ORDER_BOOK_FETCH_LIMIT: int = 25
        self.SHALLOW_OB_FETCH_DEPTH: int = 5

        # Fees
        self.TAKER_FEE_RATE: Decimal = Decimal(os.getenv("BYBIT_TAKER_FEE", "0.00055"))
        self.MAKER_FEE_RATE: Decimal = Decimal(os.getenv("BYBIT_MAKER_FEE", "0.0002"))

        # Strategy Parameters (Ehlers Volumetric Trend)
        self.EVT_ENABLED: bool = True  # Master switch for the indicator calc
        self.EVT_LENGTH: int = int(os.getenv("EVT_LENGTH", 7))
        self.EVT_MULTIPLIER: float = float(os.getenv("EVT_MULTIPLIER", 2.5))
        self.STOP_LOSS_ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", 14))
        self.STOP_LOSS_ATR_MULTIPLIER: Decimal = Decimal(
            os.getenv("ATR_MULTIPLIER", "2.5")
        )

        # Retry & Timing
        self.RETRY_COUNT: int = int(os.getenv("RETRY_COUNT", 3))
        self.RETRY_DELAY_SECONDS: float = float(os.getenv("RETRY_DELAY", 2.0))
        self.LOOP_DELAY_SECONDS: int = int(
            os.getenv("LOOP_DELAY", 60)
        )  # Wait time between cycles

        # Logging & Alerts
        self.LOG_CONSOLE_LEVEL: str = os.getenv("LOG_CONSOLE_LEVEL", "INFO")
        self.LOG_FILE_LEVEL: str = os.getenv("LOG_FILE_LEVEL", "DEBUG")
        self.LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "ehlers_strategy.log")
        self.ENABLE_SMS_ALERTS: bool = (
            os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true"
        )
        self.SMS_RECIPIENT_NUMBER: str | None = os.getenv("SMS_RECIPIENT_NUMBER")
        self.SMS_TIMEOUT_SECONDS: int = 30

        # Constants
        self.SIDE_BUY: str = "buy"
        self.SIDE_SELL: str = "sell"
        self.POS_LONG: str = "LONG"
        self.POS_SHORT: str = "SHORT"
        self.POS_NONE: str = "NONE"
        self.POSITION_QTY_EPSILON: Decimal = Decimal("1e-9")

        # --- Derived/Helper Attributes ---
        self.indicator_settings = {
            "atr_period": self.STOP_LOSS_ATR_PERIOD,
            # Add EVT params if indicators.py needs them explicitly in settings dict
            "evt_length": self.EVT_LENGTH,
            "evt_multiplier": self.EVT_MULTIPLIER,
        }
        self.analysis_flags = {
            "use_atr": True,  # Needed for stop-loss
            "use_evt": self.EVT_ENABLED,
            # Add other flags if indicators.py expects them
        }
        self.strategy_params = {  # For indicators.py if it uses strategy name mapping
            "ehlers_volumetric": {
                "evt_length": self.EVT_LENGTH,
                "evt_multiplier": self.EVT_MULTIPLIER,
            }
        }
        self.strategy = {"name": "ehlers_volumetric"}  # If indicators.py needs it


# --- Global Variables ---
logger: logging.Logger = None
# This script assumes the retry_api_call decorator is provided by bybit_utils
# If not, it needs to be defined here or imported differently.

# --- Core Functions ---


def calculate_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame | None:
    """Calculates indicators needed for the strategy."""
    if df is None or df.empty:
        logger.error("Cannot calculate indicators: Input DataFrame is empty.")
        return None
    try:
        # Pass the relevant config parts to the indicator calculator
        indicator_config = {
            "indicator_settings": config.indicator_settings,
            "analysis_flags": config.analysis_flags,
            "strategy_params": config.strategy_params,
            "strategy": config.strategy,
        }
        df_with_indicators = ind.calculate_all_indicators(df, indicator_config)
        # Check if required EVT columns were added
        evt_trend_col = f"evt_trend_{config.EVT_LENGTH}"  # Adjust if indicators.py uses different naming
        if evt_trend_col not in df_with_indicators.columns:
            logger.error(
                f"Required EVT trend column '{evt_trend_col}' not found after calculation."
            )
            return None
        logger.debug(
            f"Indicators calculated. DataFrame shape: {df_with_indicators.shape}"
        )
        return df_with_indicators
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True)
        return None


def generate_signals(df_ind: pd.DataFrame, config: Config) -> str | None:
    """Generates trading signals based on the last row of the indicator DataFrame.

    Returns:
        'buy', 'sell', or None.
    """
    if df_ind is None or df_ind.empty:
        return None

    try:
        latest = df_ind.iloc[-1]
        # Use the column names generated by indicators.py (check its implementation)
        trend_col = f"evt_trend_{config.EVT_LENGTH}"
        buy_col = f"evt_buy_{config.EVT_LENGTH}"
        sell_col = f"evt_sell_{config.EVT_LENGTH}"

        # Check if columns exist
        if not all(col in latest.index for col in [trend_col, buy_col, sell_col]):
            logger.warning(
                f"EVT signal columns ({trend_col}, {buy_col}, {sell_col}) missing in latest data."
            )
            return None

        trend = latest[trend_col]
        buy_signal = latest[buy_col]
        sell_signal = latest[sell_col]

        # Log current state
        logger.debug(
            f"Latest Data Point: Index={latest.name}, Close={latest['close']:.4f}, "
            f"{trend_col}={trend}, {buy_col}={buy_signal}, {sell_col}={sell_signal}"
        )

        # Entry Signals
        if buy_signal:  # Trend initiated bullishly
            logger.info(
                f"{Fore.GREEN}BUY signal generated based on EVT Buy flag.{Style.RESET_ALL}"
            )
            return config.SIDE_BUY
        elif sell_signal:  # Trend initiated bearishly
            logger.info(
                f"{Fore.RED}SELL signal generated based on EVT Sell flag.{Style.RESET_ALL}"
            )
            return config.SIDE_SELL

        # Exit signal detection (Optional - could be handled separately based on position)
        # Example: If trend flips directly (requires checking previous trend)
        # prev_trend = df_ind.iloc[-2][trend_col] if len(df_ind) > 1 else 0
        # if trend == -1 and prev_trend == 1: logger.info("Trend flipped DOWN (Exit Long?)")
        # if trend == 1 and prev_trend == -1: logger.info("Trend flipped UP (Exit Short?)")

        return None  # No entry signal this cycle

    except Exception as e:
        logger.error(f"Error generating signals: {e}", exc_info=True)
        return None


def calculate_stop_loss(
    df_ind: pd.DataFrame, side: str, entry_price: Decimal, config: Config
) -> Decimal | None:
    """Calculates the initial stop-loss price based on ATR."""
    if df_ind is None or df_ind.empty:
        return None
    try:
        atr_col = f"ATRr_{config.STOP_LOSS_ATR_PERIOD}"  # Default name from pandas_ta
        if atr_col not in df_ind.columns:
            logger.error(f"ATR column '{atr_col}' not found for stop-loss calculation.")
            return None

        latest_atr = safe_decimal_conversion(df_ind.iloc[-1][atr_col])
        if latest_atr is None or latest_atr <= Decimal(0):
            logger.warning("Invalid ATR value, cannot calculate stop-loss.")
            # Fallback: Use recent low/high?
            latest_low = safe_decimal_conversion(df_ind.iloc[-1]["low"])
            latest_high = safe_decimal_conversion(df_ind.iloc[-1]["high"])
            if side == config.SIDE_BUY and latest_low:
                return latest_low * Decimal("0.995")  # 0.5% below low
            if side == config.SIDE_SELL and latest_high:
                return latest_high * Decimal("1.005")  # 0.5% above high
            return None

        stop_offset = latest_atr * config.STOP_LOSS_ATR_MULTIPLIER
        if side == config.SIDE_BUY:
            stop_loss_price = entry_price - stop_offset
        elif side == config.SIDE_SELL:
            stop_loss_price = entry_price + stop_offset
        else:
            return None

        # Ensure SL is not illogical (e.g., buy SL above entry)
        if side == config.SIDE_BUY and stop_loss_price >= entry_price:
            logger.warning(
                f"Calculated Buy SL ({stop_loss_price}) >= Entry ({entry_price}). Adjusting slightly below."
            )
            stop_loss_price = entry_price * (
                Decimal(1) - Decimal("0.001")
            )  # Adjust 0.1% below
        if side == config.SIDE_SELL and stop_loss_price <= entry_price:
            logger.warning(
                f"Calculated Sell SL ({stop_loss_price}) <= Entry ({entry_price}). Adjusting slightly above."
            )
            stop_loss_price = entry_price * (
                Decimal(1) + Decimal("0.001")
            )  # Adjust 0.1% above

        logger.info(
            f"Calculated SL for {side.upper()} at {format_price(exchange, config.SYMBOL, stop_loss_price)} (Entry: {format_price(exchange, config.SYMBOL, entry_price)}, ATR: {latest_atr:.4f}, Mult: {config.STOP_LOSS_ATR_MULTIPLIER})"
        )
        return stop_loss_price

    except Exception as e:
        logger.error(f"Error calculating stop-loss: {e}", exc_info=True)
        return None


def calculate_position_size(
    exchange: ccxt.Exchange,
    symbol: str,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    config: Config,
) -> Decimal | None:
    """Calculates position size based on risk percentage and stop-loss distance."""
    try:
        _, available_balance = bybit.fetch_usdt_balance(exchange, config)
        if available_balance is None or available_balance <= Decimal("0"):
            logger.error(
                "Cannot calculate position size: Zero or invalid available balance."
            )
            return None

        risk_amount_usd = available_balance * config.RISK_PER_TRADE
        price_diff = abs(entry_price - stop_loss_price)

        if price_diff <= Decimal("0"):
            logger.error(
                f"Cannot calculate position size: Entry price ({entry_price}) and SL price ({stop_loss_price}) are too close or invalid."
            )
            return None

        position_size_base = risk_amount_usd / price_diff

        # --- Apply exchange contract size and precision constraints ---
        market = exchange.market(symbol)
        min_qty = safe_decimal_conversion(
            market.get("limits", {}).get("amount", {}).get("min"), Decimal("0")
        )
        qty_precision = market.get("precision", {}).get(
            "amount"
        )  # Number of decimal places for quantity

        if qty_precision is None:
            logger.warning(
                f"Could not determine quantity precision for {symbol}. Using raw calculation."
            )
        else:
            # Use ROUND_DOWN to not exceed risk
            position_size_base = position_size_base.quantize(
                Decimal("1." + "0" * qty_precision), rounding=ROUND_DOWN
            )

        if position_size_base < min_qty:
            logger.warning(
                f"Calculated position size ({position_size_base}) is below minimum order size ({min_qty}). No trade possible."
            )
            return None

        logger.info(
            f"Calculated position size: {format_amount(exchange, symbol, position_size_base)} {symbol.split('/')[0]} "
            f"(Risk: {risk_amount_usd:.2f} {config.USDT_SYMBOL}, Balance: {available_balance:.2f} {config.USDT_SYMBOL})"
        )
        return position_size_base

    except Exception as e:
        logger.error(f"Error calculating position size: {e}", exc_info=True)
        return None


# --- Main Execution Block ---


def run_strategy(config: Config, exchange: ccxt.bybit):
    """Main trading loop."""
    logger.info(
        f"{Fore.MAGENTA}--- Starting Ehlers Volumetric Strategy for {config.SYMBOL} on {config.TIMEFRAME} ---{Style.RESET_ALL}"
    )
    logger.info(
        f"Risk per trade: {config.RISK_PER_TRADE:.2%}, Leverage: {config.DEFAULT_LEVERAGE}x"
    )
    logger.info(
        f"EVT Params: Length={config.EVT_LENGTH}, Multiplier={config.EVT_MULTIPLIER}"
    )

    stop_loss_orders = {}  # Dictionary to track SL order IDs for open positions {symbol: order_id}

    while True:
        try:
            # --- 1. Fetch Current State ---
            logger.info(
                "-" * 30
                + f" Cycle Start: {pd.Timestamp.now(tz='UTC').isoformat()} "
                + "-" * 30
            )
            current_position = bybit.get_current_position_bybit_v5(
                exchange, config.SYMBOL, config
            )
            if (
                current_position is None
            ):  # Indicates potential API error, wait and retry
                logger.warning(
                    "Failed to get current position state. Retrying next cycle."
                )
                time.sleep(config.LOOP_DELAY_SECONDS)
                continue

            current_side = current_position["side"]
            current_qty = current_position["qty"]
            logger.info(
                f"Current Position: Side={current_side}, Qty={format_amount(exchange, config.SYMBOL, current_qty)}"
            )

            # --- 2. Fetch Data & Calculate Indicators ---
            ohlcv_df = bybit.fetch_ohlcv_paginated(
                exchange,
                config.SYMBOL,
                config.TIMEFRAME,
                limit_per_req=1000,
                max_total_candles=config.OHLCV_LIMIT,
                config=config,
            )
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning("Could not fetch sufficient OHLCV data. Skipping cycle.")
                time.sleep(config.LOOP_DELAY_SECONDS)
                continue

            ticker = bybit.fetch_ticker_validated(exchange, config.SYMBOL, config)
            if ticker is None or ticker.get("last") is None:
                logger.warning("Could not fetch valid ticker data. Skipping cycle.")
                time.sleep(config.LOOP_DELAY_SECONDS)
                continue
            current_price = ticker["last"]

            df_with_indicators = calculate_indicators(ohlcv_df, config)
            if df_with_indicators is None:
                logger.warning("Failed to calculate indicators. Skipping cycle.")
                time.sleep(config.LOOP_DELAY_SECONDS)
                continue

            # --- 3. Generate Trading Signal ---
            signal = generate_signals(df_with_indicators, config)
            logger.debug(f"Generated Signal: {signal}")

            # --- 4. Handle Exits ---
            if current_side != config.POS_NONE:
                latest_trend = df_with_indicators.iloc[-1].get(
                    f"evt_trend_{config.EVT_LENGTH}"
                )
                should_exit = False
                exit_reason = ""

                if current_side == config.POS_LONG and latest_trend == -1:
                    should_exit = True
                    exit_reason = "EVT Trend flipped Short"
                elif current_side == config.POS_SHORT and latest_trend == 1:
                    should_exit = True
                    exit_reason = "EVT Trend flipped Long"

                # Add other exit conditions here (e.g., time-based, profit target)

                if should_exit:
                    logger.warning(
                        f"{Fore.YELLOW}Exit condition met for {current_side} position: {exit_reason}. Attempting to close.{Style.RESET_ALL}"
                    )
                    # Cancel existing SL order first
                    sl_order_id = stop_loss_orders.pop(config.SYMBOL, None)
                    if sl_order_id:
                        try:
                            cancelled = bybit.cancel_order(
                                exchange, config.SYMBOL, sl_order_id, config
                            )  # Need cancel_order helper
                            if cancelled:
                                logger.info(
                                    f"Successfully cancelled SL order {sl_order_id} before closing."
                                )
                        except Exception as e:
                            logger.error(
                                f"Failed to cancel SL order {sl_order_id}: {e}"
                            )
                    else:
                        logger.warning(
                            "No tracked SL order ID found to cancel for existing position."
                        )

                    # Close the position
                    close_order = bybit.close_position_reduce_only(
                        exchange,
                        config.SYMBOL,
                        config,
                        position_to_close=current_position,
                        reason=exit_reason,
                    )
                    if close_order:
                        logger.success(
                            f"{Fore.GREEN}Position successfully closed based on exit signal.{Style.RESET_ALL}"
                        )
                        if config.ENABLE_SMS_ALERTS:
                            send_sms_alert(
                                f"[{config.SYMBOL}] {current_side} Position Closed: {exit_reason}",
                                config,
                            )
                    else:
                        logger.error(
                            f"{Fore.RED}Failed to close position for exit signal! Manual intervention may be required.{Style.RESET_ALL}"
                        )
                        if config.ENABLE_SMS_ALERTS:
                            send_sms_alert(
                                f"[{config.SYMBOL}] URGENT: Failed to close {current_side} position on exit signal!",
                                config,
                            )
                    # Wait after closing before potentially entering new trade
                    time.sleep(10)
                    continue  # Skip entry logic for this cycle

            # --- 5. Handle Entries ---
            if current_side == config.POS_NONE and signal:
                logger.info(
                    f"{Fore.CYAN}Attempting to enter {signal.upper()} position...{Style.RESET_ALL}"
                )

                # Cancel any potentially lingering orders from previous attempts
                if cancel_all_orders(
                    exchange, config.SYMBOL, config, reason="Pre-Entry Cleanup"
                ):
                    logger.info("Pre-entry order cleanup successful.")
                else:
                    logger.warning(
                        "Pre-entry order cleanup potentially failed. Proceeding with caution."
                    )

                stop_loss_price = calculate_stop_loss(
                    df_with_indicators, signal, current_price, config
                )
                if not stop_loss_price:
                    logger.error("Could not calculate stop-loss. Cannot enter trade.")
                    time.sleep(config.LOOP_DELAY_SECONDS)
                    continue

                position_size = calculate_position_size(
                    exchange, config.SYMBOL, current_price, stop_loss_price, config
                )
                if not position_size:
                    logger.error(
                        "Could not calculate position size. Cannot enter trade."
                    )
                    time.sleep(config.LOOP_DELAY_SECONDS)
                    continue

                # Place Market Order
                entry_order = bybit.place_market_order_slippage_check(
                    exchange, config.SYMBOL, signal, position_size, config
                )

                if entry_order and entry_order.get("id"):
                    logger.success(
                        f"Entry market order submitted successfully. ID: ...{format_order_id(entry_order['id'])}"
                    )
                    # Wait briefly for order fill confirmation / position update
                    time.sleep(5)

                    # Verify position opened (optional but good practice)
                    pos_after_entry = bybit.get_current_position_bybit_v5(
                        exchange, config.SYMBOL, config
                    )
                    filled_qty = safe_decimal_conversion(
                        entry_order.get("filled", 0)
                    )  # Get filled amount from order receipt
                    if filled_qty <= config.POSITION_QTY_EPSILON:
                        filled_qty = position_size  # Assume full fill if receipt is delayed/incomplete? Risky.

                    if (
                        pos_after_entry["side"] == signal.upper()
                        and filled_qty > config.POSITION_QTY_EPSILON
                    ):
                        logger.info(
                            f"Position confirmed open: {pos_after_entry['side']} {format_amount(exchange, config.SYMBOL, pos_after_entry['qty'])}"
                        )

                        # Place Stop Loss
                        sl_side = (
                            config.SIDE_SELL
                            if signal == config.SIDE_BUY
                            else config.SIDE_BUY
                        )
                        sl_order = bybit.place_native_stop_loss(
                            exchange,
                            config.SYMBOL,
                            sl_side,
                            filled_qty,
                            stop_loss_price,
                            config,
                        )

                        if sl_order and sl_order.get("id"):
                            logger.success(
                                f"Native stop-loss order placed successfully. ID: ...{format_order_id(sl_order['id'])}"
                            )
                            stop_loss_orders[config.SYMBOL] = sl_order[
                                "id"
                            ]  # Track the SL order ID
                            if config.ENABLE_SMS_ALERTS:
                                send_sms_alert(
                                    f"[{config.SYMBOL}] Entered {signal.upper()} {format_amount(exchange, config.SYMBOL, filled_qty)} @ {format_price(exchange, config.SYMBOL, current_price)}. SL @ {format_price(exchange, config.SYMBOL, stop_loss_price)}",
                                    config,
                                )
                        else:
                            logger.error(
                                f"{Fore.RED}Failed to place stop-loss order after entry! Attempting to close position immediately.{Style.RESET_ALL}"
                            )
                            if config.ENABLE_SMS_ALERTS:
                                send_sms_alert(
                                    f"[{config.SYMBOL}] URGENT: Failed to place SL after {signal.upper()} entry! Closing position.",
                                    config,
                                )
                            close_order = bybit.close_position_reduce_only(
                                exchange,
                                config.SYMBOL,
                                config,
                                reason="Failed SL Placement",
                            )
                            if close_order:
                                logger.warning(
                                    "Position closed due to failed SL placement."
                                )
                            else:
                                logger.critical(
                                    "CRITICAL: FAILED TO CLOSE POSITION AFTER FAILED SL PLACEMENT!"
                                )
                    else:
                        logger.error(
                            f"Entry order submitted but position confirmation failed or quantity is zero. Order ID: {entry_order.get('id')}, Pos Side: {pos_after_entry['side']}, Filled Qty: {filled_qty}"
                        )
                        # Attempt to cancel the potentially unfilled/partially filled order? Risky. Best to alert.
                        if config.ENABLE_SMS_ALERTS:
                            send_sms_alert(
                                f"[{config.SYMBOL}] URGENT: Entry order {entry_order.get('id')} confirmation failed!",
                                config,
                            )

                else:
                    logger.error(
                        f"{Fore.RED}Entry market order placement failed.{Style.RESET_ALL}"
                    )
                    # SMS alert handled within place_market_order

            # --- 6. Wait for next cycle ---
            logger.info(
                f"Cycle complete. Waiting {config.LOOP_DELAY_SECONDS} seconds..."
            )
            time.sleep(config.LOOP_DELAY_SECONDS)

        except ccxt.NetworkError as e:
            logger.warning(
                f"{Fore.YELLOW}Network Error occurred in main loop: {e}. Retrying after delay...{Style.RESET_ALL}"
            )
            time.sleep(config.LOOP_DELAY_SECONDS * 2)  # Longer delay on network issues
        except ccxt.ExchangeError as e:
            logger.error(
                f"{Fore.RED}Exchange Error occurred in main loop: {e}. Retrying after delay...{Style.RESET_ALL}"
            )
            if config.ENABLE_SMS_ALERTS:
                send_sms_alert(f"[{config.SYMBOL}] Exchange Error: {e}", config)
            time.sleep(config.LOOP_DELAY_SECONDS)
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt received. Shutting down...")
            break
        except Exception as e:
            logger.critical(
                f"{Back.RED}{Fore.WHITE}!!! UNEXPECTED CRITICAL ERROR IN MAIN LOOP !!!{Style.RESET_ALL}",
                exc_info=True,
            )
            if config.ENABLE_SMS_ALERTS:
                send_sms_alert(
                    f"[{config.SYMBOL}] CRITICAL ERROR: {type(e).__name__}. Check logs!",
                    config,
                )
            # Decide whether to break or continue after a delay
            logger.info("Attempting to continue after critical error...")
            time.sleep(
                config.LOOP_DELAY_SECONDS * 3
            )  # Longer delay after critical error

    logger.info(
        f"{Fore.MAGENTA}--- Ehlers Volumetric Strategy Stopped ---{Style.RESET_ALL}"
    )
    # Optional: Close any open positions on shutdown
    # logger.info("Attempting to close any open positions on exit...")
    # current_pos_on_exit = bybit.get_current_position_bybit_v5(exchange, config.SYMBOL, config)
    # if current_pos_on_exit and current_pos_on_exit['side'] != config.POS_NONE:
    #     close_position_reduce_only(exchange, config.SYMBOL, config, position_to_close=current_pos_on_exit, reason="Shutdown")


if __name__ == "__main__":
    # --- Initialize Logger ---
    # Use environment variables for levels if set, otherwise use Config defaults
    logger = setup_logger(
        logger_name="EhlersStrategy",
        log_file=os.getenv("LOG_FILE_PATH", "ehlers_strategy.log"),
        console_level=logging.getLevelName(os.getenv("LOG_CONSOLE_LEVEL", "INFO")),
        file_level=logging.getLevelName(os.getenv("LOG_FILE_LEVEL", "DEBUG")),
        third_party_log_level=logging.WARNING,  # Keep third-party libs quiet by default
    )

    # --- Load Configuration ---
    CONFIG = Config()

    # --- Validate Config ---
    if not CONFIG.API_KEY or not CONFIG.API_SECRET:
        logger.critical(
            f"{Back.RED}API Key or Secret not found in environment variables. Exiting.{Style.RESET_ALL}"
        )
        sys.exit(1)
    logger.info(
        f"Configuration loaded. Testnet: {CONFIG.TESTNET_MODE}, Symbol: {CONFIG.SYMBOL}"
    )

    # --- Initialize Exchange ---
    exchange = bybit.initialize_bybit(CONFIG)
    if not exchange:
        logger.critical(
            f"{Back.RED}Failed to initialize Bybit exchange. Exiting.{Style.RESET_ALL}"
        )
        if CONFIG.ENABLE_SMS_ALERTS:
            send_sms_alert(
                "Strategy exit: Bybit exchange initialization failed.", CONFIG
            )
        sys.exit(1)

    # --- Set Leverage ---
    if not bybit.set_leverage(exchange, CONFIG.SYMBOL, CONFIG.DEFAULT_LEVERAGE, CONFIG):
        logger.critical(
            f"{Back.RED}Failed to set leverage to {CONFIG.DEFAULT_LEVERAGE}x. Exiting.{Style.RESET_ALL}"
        )
        if CONFIG.ENABLE_SMS_ALERTS:
            send_sms_alert(
                f"Strategy exit: Failed to set leverage {CONFIG.DEFAULT_LEVERAGE}x for {CONFIG.SYMBOL}.",
                CONFIG,
            )
        sys.exit(1)

    # --- Validate Market ---
    market_details = bybit.validate_market(exchange, CONFIG.SYMBOL, CONFIG)
    if not market_details:
        logger.critical(
            f"{Back.RED}Market validation failed for {CONFIG.SYMBOL}. Exiting.{Style.RESET_ALL}"
        )
        if CONFIG.ENABLE_SMS_ALERTS:
            send_sms_alert(
                f"Strategy exit: Market validation failed for {CONFIG.SYMBOL}.", CONFIG
            )
        sys.exit(1)

    # --- Start Strategy ---
    try:
        run_strategy(CONFIG, exchange)
    except Exception as e:
        logger.critical(
            f"{Back.RED}Strategy execution failed with unhandled exception: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
        if CONFIG.ENABLE_SMS_ALERTS:
            send_sms_alert(
                f"[{CONFIG.SYMBOL}] CRITICAL FAILURE: Strategy terminated unexpectedly. Check logs!",
                CONFIG,
            )
        sys.exit(1)
    finally:
        # Clean up exchange connection? CCXT might handle this.
        if exchange:
            try:
                exchange.close()
                logger.info("Exchange connection closed.")
            except Exception as e:
                logger.warning(f"Error closing exchange connection: {e}")
