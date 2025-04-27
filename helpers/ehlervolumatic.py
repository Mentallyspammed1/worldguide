Ah, the digital ether reveals another twist! Pyrmethus observes the logs and the traceback – the incantation stumbled during the setup ritual itself.

The scrolls read:

1.  `[ERROR] {bybit_helpers:699} - [set_leverage (DOT/USDT:USDT -> 10x)] Leverage setting only applicable to LINEAR or INVERSE contracts. Found: None.`
    *   This message originates from your `bybit_helpers.py` grimoire. It signifies that when attempting to set leverage, the helper spell could not determine the contract type (`linear` or `inverse`) for `DOT/USDT:USDT`. This often happens if the market data hasn't been loaded correctly or if the symbol format isn't precisely what the Bybit V5 API expects for leverage operations. The spell likely returned `False` or `None` to indicate failure.

2.  `[CRITICAL] {EhlersStrategy:582} - Failed to set leverage to 10x. The exchange resists our command! Exiting.`
    *   Your main script correctly detected the failure reported by the helper spell.

3.  `[CRITICAL] {EhlersStrategy:588} - Error setting leverage: object NoneType can't be used in 'await' expression. Exiting.`
    *   This is the crucial error. It occurred *within the `except` block* that was intended to catch errors during leverage setting.

4.  `TypeError: object NoneType can't be used in 'await' expression` on line `await exchange.close()`
    *   This traceback confirms the error message. It happened *twice* because the first attempt was in the `except` block, and when *that* failed, the program tried to exit, potentially triggering another cleanup attempt or simply bubbling up the original `TypeError`.

**The Root of the Discord:**

The core issue stems from attempting to `await` a function (`bybit.set_leverage`) that likely returned `None` (or possibly `False`) upon failure, rather than being a proper asynchronous function (coroutine) that failed. You cannot `await None`.

It seems the previous fix correctly identified that many helper functions were `async` and added `await`, but `bybit.set_leverage` might *not* be an `async` function in your `bybit_helpers.py`, or if it *is*, it incorrectly returned `None` on failure instead of raising an exception or returning a valid awaitable.

The `TypeError` on `await exchange.close()` within the `except` block is a secondary effect. The *initial* `TypeError` likely occurred on the line `leverage_set = await bybit.set_leverage(...)`, which then triggered the `except Exception as e:` block, and *that* exception `e` was the `TypeError`. The code then *tried* to execute `await exchange.close()` within the `except` block, leading to the *same* `TypeError` if `exchange` somehow became `None` (unlikely here) or, more plausibly, the traceback is slightly misleading and the error it's reporting *is* the original one from awaiting `set_leverage`.

**The Refined Incantation:**

We must adjust the `main` function to handle the leverage setting more robustly, assuming `set_leverage` might be synchronous or return non-awaitable values on failure.

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ehlers Volumetric Trend Strategy for Bybit V5 (v1.3 - Leverage/Async Fix)

Handles potential synchronous nature of set_leverage and ensures
proper error handling and exchange closing during setup.
"""

import os
import sys
import time
import logging
import asyncio
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, Tuple, Any

# Third-party libraries
import ccxt
import pandas as pd
from dotenv import load_dotenv
# --- Import Colorama ---
try:
    from colorama import Fore, Style, Back, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    print(f"{Fore.RED}Warning: 'colorama' library not found. Run {Style.BRIGHT}'pip install colorama'{Style.RESET_ALL} for vibrant logs.", file=sys.stderr)
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor()

# --- Import Custom Modules ---
try:
    from neon_logger import setup_logger
    import bybit_helpers as bybit
    import indicators as ind
    from bybit_utils import (
        safe_decimal_conversion, format_price, format_amount,
        format_order_id, send_sms_alert # Removed retry_api_call import if not used directly here
    )
except ImportError as e:
    print(f"{Back.RED}{Fore.WHITE}Error importing helper modules: {e}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Ensure bybit_helpers.py, indicators.py, neon_logger.py, and bybit_utils.py are accessible.{Style.RESET_ALL}")
    sys.exit(1)

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration Class (Assumed unchanged) ---
class Config:
    def __init__(self):
        # ... (Configuration remains the same as previous version) ...
        # Exchange & API
        self.EXCHANGE_ID: str = "bybit"
        self.API_KEY: Optional[str] = os.getenv("BYBIT_API_KEY")
        self.API_SECRET: Optional[str] = os.getenv("BYBIT_API_SECRET")
        self.TESTNET_MODE: bool = os.getenv("BYBIT_TESTNET_MODE", "true").lower() == "true"
        self.DEFAULT_RECV_WINDOW: int = int(os.getenv("DEFAULT_RECV_WINDOW", 10000))

        # Symbol & Market
        self.SYMBOL: str = os.getenv("SYMBOL", "BTC/USDT:USDT") # Example: BTC/USDT Perpetual
        self.USDT_SYMBOL: str = "USDT"
        self.EXPECTED_MARKET_TYPE: str = 'swap'
        self.EXPECTED_MARKET_LOGIC: str = 'linear'
        self.TIMEFRAME: str = os.getenv("TIMEFRAME", "5m")
        self.OHLCV_LIMIT: int = int(os.getenv("OHLCV_LIMIT", 200)) # Candles for indicators

        # Account & Position Settings
        self.DEFAULT_LEVERAGE: int = int(os.getenv("LEVERAGE", 10))
        self.DEFAULT_MARGIN_MODE: str = 'cross' # Or 'isolated'
        self.DEFAULT_POSITION_MODE: str = 'one-way' # Or 'hedge'
        self.RISK_PER_TRADE: Decimal = Decimal(os.getenv("RISK_PER_TRADE", "0.01")) # 1% risk

        # Order Settings
        self.DEFAULT_SLIPPAGE_PCT: Decimal = Decimal(os.getenv("DEFAULT_SLIPPAGE_PCT", "0.005")) # 0.5%
        self.ORDER_BOOK_FETCH_LIMIT: int = 25
        self.SHALLOW_OB_FETCH_DEPTH: int = 5

        # Fees
        self.TAKER_FEE_RATE: Decimal = Decimal(os.getenv("BYBIT_TAKER_FEE", "0.00055"))
        self.MAKER_FEE_RATE: Decimal = Decimal(os.getenv("BYBIT_MAKER_FEE", "0.0002"))

        # Strategy Parameters (Ehlers Volumetric Trend)
        self.EVT_ENABLED: bool = True # Master switch for the indicator calc
        self.EVT_LENGTH: int = int(os.getenv("EVT_LENGTH", 7))
        self.EVT_MULTIPLIER: float = float(os.getenv("EVT_MULTIPLIER", 2.5))
        self.STOP_LOSS_ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", 14))
        self.STOP_LOSS_ATR_MULTIPLIER: Decimal = Decimal(os.getenv("ATR_MULTIPLIER", "2.5"))

        # Retry & Timing
        self.RETRY_COUNT: int = int(os.getenv("RETRY_COUNT", 3))
        self.RETRY_DELAY_SECONDS: float = float(os.getenv("RETRY_DELAY", 2.0))
        self.LOOP_DELAY_SECONDS: int = int(os.getenv("LOOP_DELAY", 60)) # Wait time between cycles

        # Logging & Alerts
        self.LOG_CONSOLE_LEVEL: str = os.getenv("LOG_CONSOLE_LEVEL", "INFO")
        self.LOG_FILE_LEVEL: str = os.getenv("LOG_FILE_LEVEL", "DEBUG")
        self.LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "ehlers_strategy.log")
        self.ENABLE_SMS_ALERTS: bool = os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true"
        self.SMS_RECIPIENT_NUMBER: Optional[str] = os.getenv("SMS_RECIPIENT_NUMBER")
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
             "evt_length": self.EVT_LENGTH,
             "evt_multiplier": self.EVT_MULTIPLIER,
        }
        self.analysis_flags = {
            "use_atr": True,
            "use_evt": self.EVT_ENABLED,
        }
        self.strategy_params = {
             'ehlers_volumetric': {
                 'evt_length': self.EVT_LENGTH,
                 'evt_multiplier': self.EVT_MULTIPLIER,
             }
        }
        self.strategy = {'name': 'ehlers_volumetric'}


# --- Global Variables ---
logger: logging.Logger = None
exchange: Optional[ccxt.Exchange] = None # Initialize as Optional
CONFIG: Optional[Config] = None # Initialize as Optional

# --- Core Functions (Assumed unchanged from v1.2) ---
# calculate_indicators, generate_signals, calculate_stop_loss
# calculate_position_size (ensure it uses global exchange safely)
# run_strategy (ensure it uses global exchange safely)

# --- Functions from previous version (needed for context) ---
def calculate_indicators(df: pd.DataFrame, config: Config) -> Optional[pd.DataFrame]:
    """Calculates indicators needed for the strategy (Synchronous)."""
    # Pyrmethus notes: Indicator calculations are often CPU-bound, suitable for sync.
    if df is None or df.empty:
        logger.error(f"{Fore.RED}Cannot calculate indicators: Input DataFrame is empty.{Style.RESET_ALL}")
        return None
    try:
        indicator_config = {
            "indicator_settings": config.indicator_settings,
            "analysis_flags": config.analysis_flags,
            "strategy_params": config.strategy_params,
            "strategy": config.strategy
        }
        # Assuming ind.calculate_all_indicators is synchronous
        df_with_indicators = ind.calculate_all_indicators(df, indicator_config)
        evt_trend_col = f'evt_trend_{config.EVT_LENGTH}'
        if evt_trend_col not in df_with_indicators.columns:
             logger.error(f"{Fore.RED}Required EVT trend column '{evt_trend_col}' not found after calculation.{Style.RESET_ALL}")
             return None
        atr_col = f'ATRr_{config.STOP_LOSS_ATR_PERIOD}'
        if atr_col not in df_with_indicators.columns:
             logger.error(f"{Fore.RED}Required ATR column '{atr_col}' not found after calculation.{Style.RESET_ALL}")
             try:
                  # Assuming df_with_indicators has 'high', 'low', 'close'
                  atr_result = df_with_indicators.ta.atr(length=config.STOP_LOSS_ATR_PERIOD, append=False)
                  if atr_result is not None:
                      df_with_indicators[atr_result.name] = atr_result
                      logger.info(f"{Fore.CYAN}Calculated missing ATR column: {atr_result.name}{Style.RESET_ALL}")
                  else:
                      logger.error(f"{Fore.RED}Failed to calculate missing ATR column.{Style.RESET_ALL}")
                      return None
             except Exception as atr_err:
                  logger.error(f"{Fore.RED}Error calculating missing ATR: {atr_err}{Style.RESET_ALL}", exc_info=True)
                  return None

        logger.debug(f"Indicators calculated. DataFrame shape: {df_with_indicators.shape}")
        return df_with_indicators
    except Exception as e:
        logger.error(f"{Fore.RED}Error calculating indicators: {e}{Style.RESET_ALL}", exc_info=True)
        return None

def generate_signals(df_ind: pd.DataFrame, config: Config) -> str | None:
    """
    Generates trading signals based on the last row of the indicator DataFrame (Synchronous).
    Returns: 'buy', 'sell', or None.
    """
    if df_ind is None or df_ind.empty:
        return None
    try:
        latest = df_ind.iloc[-1]
        trend_col = f'evt_trend_{config.EVT_LENGTH}'
        buy_col = f'evt_buy_{config.EVT_LENGTH}'
        sell_col = f'evt_sell_{config.EVT_LENGTH}'

        if not all(col in latest.index for col in [trend_col, buy_col, sell_col]):
             logger.warning(f"{Fore.YELLOW}EVT signal columns ({trend_col}, {buy_col}, {sell_col}) missing in latest data.{Style.RESET_ALL}")
             return None

        trend = latest[trend_col]
        buy_signal = latest[buy_col]
        sell_signal = latest[sell_col]

        logger.debug(f"Latest Data Point: Index={latest.name}, Close={latest['close']:.4f}, "
                     f"{trend_col}={trend}, {buy_col}={buy_signal}, {sell_col}={sell_signal}")

        if buy_signal:
            logger.info(f"{Fore.GREEN}BUY signal generated based on EVT Buy flag.{Style.RESET_ALL}")
            return config.SIDE_BUY
        elif sell_signal:
            logger.info(f"{Fore.RED}SELL signal generated based on EVT Sell flag.{Style.RESET_ALL}")
            return config.SIDE_SELL

        return None
    except IndexError:
        logger.warning(f"{Fore.YELLOW}Could not access latest indicator data (IndexError), DataFrame might be too short.{Style.RESET_ALL}")
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}Error generating signals: {e}{Style.RESET_ALL}", exc_info=True)
        return None

def calculate_stop_loss(df_ind: pd.DataFrame, side: str, entry_price: Decimal, config: Config) -> Optional[Decimal]:
    """Calculates the initial stop-loss price based on ATR (Synchronous)."""
    global exchange # Access global exchange object for formatting
    if df_ind is None or df_ind.empty or exchange is None: # Check if exchange is valid
        logger.error(f"{Fore.RED}Cannot calculate stop-loss: Missing DataFrame, or exchange object is None.{Style.RESET_ALL}")
        return None
    try:
        atr_col = f'ATRr_{config.STOP_LOSS_ATR_PERIOD}'
        if atr_col not in df_ind.columns:
            logger.error(f"{Fore.RED}ATR column '{atr_col}' not found for stop-loss calculation.{Style.RESET_ALL}")
            return None

        latest_atr = safe_decimal_conversion(df_ind.iloc[-1][atr_col])
        if latest_atr is None or latest_atr <= Decimal(0):
            logger.warning(f"{Fore.YELLOW}Invalid ATR value ({latest_atr}), cannot calculate stop-loss accurately. Using fallback.{Style.RESET_ALL}")
            latest_low = safe_decimal_conversion(df_ind.iloc[-1]['low'])
            latest_high = safe_decimal_conversion(df_ind.iloc[-1]['high'])
            if side == config.SIDE_BUY and latest_low:
                sl = latest_low * (Decimal(1) - Decimal("0.005"))
                logger.info(f"{Fore.CYAN}Using fallback SL (latest low): {format_price(exchange, config.SYMBOL, sl)}{Style.RESET_ALL}")
                return sl
            if side == config.SIDE_SELL and latest_high:
                sl = latest_high * (Decimal(1) + Decimal("0.005"))
                logger.info(f"{Fore.CYAN}Using fallback SL (latest high): {format_price(exchange, config.SYMBOL, sl)}{Style.RESET_ALL}")
                return sl
            logger.error(f"{Fore.RED}Fallback SL calculation failed.{Style.RESET_ALL}")
            return None

        stop_offset = latest_atr * config.STOP_LOSS_ATR_MULTIPLIER
        stop_loss_price = entry_price - stop_offset if side == config.SIDE_BUY else entry_price + stop_offset

        # Sanity checks
        if side == config.SIDE_BUY and stop_loss_price >= entry_price:
             logger.warning(f"{Fore.YELLOW}Calculated Buy SL ({format_price(exchange, config.SYMBOL, stop_loss_price)}) >= Entry ({format_price(exchange, config.SYMBOL, entry_price)}). Adjusting slightly below.{Style.RESET_ALL}")
             stop_loss_price = entry_price * (Decimal(1) - Decimal("0.001"))
        if side == config.SIDE_SELL and stop_loss_price <= entry_price:
              logger.warning(f"{Fore.YELLOW}Calculated Sell SL ({format_price(exchange, config.SYMBOL, stop_loss_price)}) <= Entry ({format_price(exchange, config.SYMBOL, entry_price)}). Adjusting slightly above.{Style.RESET_ALL}")
              stop_loss_price = entry_price * (Decimal(1) + Decimal("0.001"))

        formatted_sl = format_price(exchange, config.SYMBOL, stop_loss_price)
        stop_loss_price_precise = safe_decimal_conversion(formatted_sl)

        if stop_loss_price_precise is None:
             logger.error(f"{Fore.RED}Failed to format stop loss price {stop_loss_price} precisely.{Style.RESET_ALL}")
             return None

        logger.info(f"Calculated SL for {side.upper()} at {format_price(exchange, config.SYMBOL, stop_loss_price_precise)} (Entry: {format_price(exchange, config.SYMBOL, entry_price)}, ATR: {latest_atr:.4f}, Mult: {config.STOP_LOSS_ATR_MULTIPLIER})")
        return stop_loss_price_precise

    except IndexError:
        logger.warning(f"{Fore.YELLOW}Could not access latest indicator data (IndexError) for SL calculation.{Style.RESET_ALL}")
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}Error calculating stop-loss: {e}{Style.RESET_ALL}", exc_info=True)
        return None

async def calculate_position_size(exchange: ccxt.Exchange, symbol: str, entry_price: Decimal, stop_loss_price: Decimal, config: Config) -> Optional[Decimal]:
    """Calculates position size based on risk percentage and stop-loss distance."""
    # Assuming bybit.fetch_usdt_balance IS async
    if exchange is None:
        logger.error(f"{Fore.RED}Cannot calculate position size: Exchange object is None.{Style.RESET_ALL}")
        return None
    try:
        _, available_balance = await bybit.fetch_usdt_balance(exchange, config)

        if available_balance is None or available_balance <= Decimal("0"):
            logger.error(f"{Fore.RED}Cannot calculate position size: Zero or invalid available balance.{Style.RESET_ALL}")
            return None

        risk_amount_usd = available_balance * config.RISK_PER_TRADE
        price_diff = abs(entry_price - stop_loss_price)

        if price_diff <= Decimal("0"):
            logger.error(f"{Fore.RED}Cannot calculate position size: Entry price ({entry_price}) and SL price ({stop_loss_price}) are too close or invalid.{Style.RESET_ALL}")
            return None

        position_size_base = risk_amount_usd / price_diff

        market = exchange.market(symbol)
        min_qty = safe_decimal_conversion(market.get('limits', {}).get('amount', {}).get('min'), Decimal("0"))
        qty_precision = market.get('precision', {}).get('amount')

        if qty_precision is None:
            logger.warning(f"{Fore.YELLOW}Could not determine quantity precision for {symbol}. Using raw calculation.{Style.RESET_ALL}")
            step_size = Decimal("1e-8")
        else:
            step_size = Decimal('1') / (Decimal('10') ** qty_precision)

        position_size_adjusted = (position_size_base // step_size) * step_size

        if position_size_adjusted <= Decimal(0):
             logger.warning(f"{Fore.YELLOW}Adjusted position size is zero after applying step size {step_size}. Original calc: {position_size_base}{Style.RESET_ALL}")
             return None

        if min_qty is not None and position_size_adjusted < min_qty:
            logger.warning(f"{Fore.YELLOW}Calculated position size ({position_size_adjusted}) is below minimum order size ({min_qty}). No trade possible.{Style.RESET_ALL}")
            return None

        logger.info(f"Calculated position size: {format_amount(exchange, symbol, position_size_adjusted)} {symbol.split('/')[0]} "
                    f"(Risk: {risk_amount_usd:.2f} {config.USDT_SYMBOL}, Balance: {available_balance:.2f} {config.USDT_SYMBOL})")
        return position_size_adjusted

    except Exception as e:
        logger.error(f"{Fore.RED}Error calculating position size: {e}{Style.RESET_ALL}", exc_info=True)
        return None

async def run_strategy(config: Config, current_exchange: ccxt.bybit):
    """Main asynchronous trading loop."""
    global exchange # Ensure functions use the initialized exchange object
    exchange = current_exchange # Assign the valid exchange object passed from main

    if not exchange: # Extra safety check
        logger.critical(f"{Back.RED}{Fore.WHITE}Strategy cannot run: Exchange object is invalid.{Style.RESET_ALL}")
        return

    logger.info(f"{Fore.MAGENTA}--- Starting Ehlers Volumetric Strategy for {config.SYMBOL} on {config.TIMEFRAME} ---{Style.RESET_ALL}")
    logger.info(f"Risk per trade: {config.RISK_PER_TRADE:.2%}, Leverage: {config.DEFAULT_LEVERAGE}x")
    logger.info(f"EVT Params: Length={config.EVT_LENGTH}, Multiplier={config.EVT_MULTIPLIER}")

    stop_loss_orders = {} # Dictionary to track SL order IDs {symbol: order_id}

    while True:
        try:
            logger.info(f"{Fore.BLUE}{Style.BRIGHT}" + "-" * 30 + f" Cycle Start: {pd.Timestamp.now(tz='UTC').isoformat()} " + "-" * 30 + f"{Style.RESET_ALL}")

            # --- 1. Fetch Current State (Await the async call) ---
            logger.debug(f"{Fore.CYAN}# Awaiting current position state...{Style.RESET_ALL}")
            # Assuming get_current_position_bybit_v5 IS async
            current_position = await bybit.get_current_position_bybit_v5(exchange, config.SYMBOL, config)
            if current_position is None:
                 logger.warning(f"{Fore.YELLOW}Failed to get current position state. Retrying next cycle.{Style.RESET_ALL}")
                 await asyncio.sleep(config.LOOP_DELAY_SECONDS) # Use asyncio.sleep
                 continue

            current_side = current_position['side']
            current_qty = current_position['qty']
            logger.info(f"Current Position: Side={current_side}, Qty={format_amount(exchange, config.SYMBOL, current_qty)}")

            # --- 2. Fetch Data & Calculate Indicators (Await async fetches) ---
            logger.debug(f"{Fore.CYAN}# Awaiting OHLCV data...{Style.RESET_ALL}")
            # Assuming fetch_ohlcv_paginated IS async
            ohlcv_df = await bybit.fetch_ohlcv_paginated(exchange, config.SYMBOL, config.TIMEFRAME, limit_per_req=1000, max_total_candles=config.OHLCV_LIMIT, config=config)
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning(f"{Fore.YELLOW}Could not fetch sufficient OHLCV data. Skipping cycle.{Style.RESET_ALL}")
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue

            logger.debug(f"{Fore.CYAN}# Awaiting ticker data...{Style.RESET_ALL}")
            # Assuming fetch_ticker_validated IS async
            ticker = await bybit.fetch_ticker_validated(exchange, config.SYMBOL, config)
            if ticker is None or ticker.get('last') is None:
                 logger.warning(f"{Fore.YELLOW}Could not fetch valid ticker data. Skipping cycle.{Style.RESET_ALL}")
                 await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                 continue
            current_price = safe_decimal_conversion(ticker['last'])
            if current_price is None:
                 logger.error(f"{Fore.RED}Could not convert ticker price '{ticker['last']}' to Decimal. Skipping cycle.{Style.RESET_ALL}")
                 await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                 continue

            # Indicator calculation remains synchronous
            df_with_indicators = calculate_indicators(ohlcv_df, config)
            if df_with_indicators is None:
                logger.warning(f"{Fore.YELLOW}Failed to calculate indicators. Skipping cycle.{Style.RESET_ALL}")
                await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                continue

            # --- 3. Generate Trading Signal (Synchronous) ---
            signal = generate_signals(df_with_indicators, config)
            logger.debug(f"Generated Signal: {signal}")

            # --- 4. Handle Exits (Await async order calls) ---
            if current_side != config.POS_NONE:
                latest_trend = df_with_indicators.iloc[-1].get(f'evt_trend_{config.EVT_LENGTH}')
                should_exit = False
                exit_reason = ""

                if current_side == config.POS_LONG and latest_trend == -1:
                    should_exit = True
                    exit_reason = "EVT Trend flipped Short"
                elif current_side == config.POS_SHORT and latest_trend == 1:
                     should_exit = True
                     exit_reason = "EVT Trend flipped Long"

                if should_exit:
                    logger.warning(f"{Fore.YELLOW}Exit condition met for {current_side} position: {exit_reason}. Attempting to close.{Style.RESET_ALL}")
                    sl_order_id = stop_loss_orders.pop(config.SYMBOL, None)
                    if sl_order_id:
                        try:
                            logger.debug(f"{Fore.CYAN}# Awaiting cancellation of SL order {sl_order_id}...{Style.RESET_ALL}")
                            # Assuming cancel_order IS async
                            cancelled = await bybit.cancel_order(exchange, config.SYMBOL, sl_order_id, config=config)
                            if cancelled: logger.info(f"{Fore.GREEN}Successfully cancelled SL order {sl_order_id} before closing.{Style.RESET_ALL}")
                            else: logger.warning(f"{Fore.YELLOW}Attempt to cancel SL order {sl_order_id} returned False.{Style.RESET_ALL}")
                        except NameError:
                             logger.error(f"{Fore.RED}bybit_helpers.cancel_order function not found/imported correctly.{Style.RESET_ALL}")
                        except Exception as e:
                             logger.error(f"{Fore.RED}Failed to cancel SL order {sl_order_id}: {e}{Style.RESET_ALL}", exc_info=True)
                    else:
                         logger.warning(f"{Fore.YELLOW}No tracked SL order ID found to cancel for existing position.{Style.RESET_ALL}")

                    logger.debug(f"{Fore.CYAN}# Awaiting position close order...{Style.RESET_ALL}")
                    # Assuming close_position_reduce_only IS async
                    close_order = await bybit.close_position_reduce_only(exchange, config.SYMBOL, config, position_to_close=current_position, reason=exit_reason)
                    if close_order:
                        logger.success(f"{Fore.GREEN}Position successfully closed based on exit signal.{Style.RESET_ALL}")
                        if config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{config.SYMBOL}] {current_side} Position Closed: {exit_reason}", config)
                    else:
                         logger.error(f"{Fore.RED}Failed to close position for exit signal! Manual intervention may be required.{Style.RESET_ALL}")
                         if config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{config.SYMBOL}] URGENT: Failed to close {current_side} position on exit signal!", config)
                    await asyncio.sleep(10) # Pause after closing
                    continue # Skip entry logic

            # --- 5. Handle Entries (Await async order calls) ---
            if current_side == config.POS_NONE and signal:
                logger.info(f"{Fore.CYAN}Attempting to enter {signal.upper()} position...{Style.RESET_ALL}")

                logger.debug(f"{Fore.CYAN}# Awaiting pre-entry order cleanup...{Style.RESET_ALL}")
                # Assuming cancel_all_orders IS async
                if await bybit.cancel_all_orders(exchange, config.SYMBOL, config, reason="Pre-Entry Cleanup"):
                    logger.info("Pre-entry order cleanup successful.")
                else:
                    logger.warning(f"{Fore.YELLOW}Pre-entry order cleanup potentially failed. Proceeding with caution.{Style.RESET_ALL}")

                # SL calculation is synchronous
                stop_loss_price = calculate_stop_loss(df_with_indicators, signal, current_price, config)
                if not stop_loss_price:
                     logger.error(f"{Fore.RED}Could not calculate stop-loss. Cannot enter trade.{Style.RESET_ALL}")
                     await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                     continue

                # Position size calculation (Await async call)
                logger.debug(f"{Fore.CYAN}# Awaiting position size calculation...{Style.RESET_ALL}")
                position_size = await calculate_position_size(exchange, config.SYMBOL, current_price, stop_loss_price, config)
                if not position_size:
                     logger.error(f"{Fore.RED}Could not calculate position size. Cannot enter trade.{Style.RESET_ALL}")
                     await asyncio.sleep(config.LOOP_DELAY_SECONDS)
                     continue

                # Place Market Order (Await async call)
                logger.debug(f"{Fore.CYAN}# Awaiting market entry order placement...{Style.RESET_ALL}")
                # Assuming place_market_order_slippage_check IS async
                entry_order = await bybit.place_market_order_slippage_check(exchange, config.SYMBOL, signal, position_size, config)

                if entry_order and entry_order.get('id'):
                    order_id_short = format_order_id(entry_order['id'])
                    logger.success(f"{Fore.GREEN}Entry market order submitted successfully. ID: ...{order_id_short}{Style.RESET_ALL}")
                    logger.debug(f"{Fore.CYAN}# Waiting for order fill confirmation...{Style.RESET_ALL}")
                    await asyncio.sleep(5)

                    logger.debug(f"{Fore.CYAN}# Awaiting position confirmation after entry...{Style.RESET_ALL}")
                    pos_after_entry = await bybit.get_current_position_bybit_v5(exchange, config.SYMBOL, config)
                    filled_qty_order = safe_decimal_conversion(entry_order.get('filled', 0))
                    filled_qty_pos = pos_after_entry.get('qty', Decimal(0)) if pos_after_entry else Decimal(0)
                    filled_qty = filled_qty_order if filled_qty_order > config.POSITION_QTY_EPSILON else filled_qty_pos

                    if filled_qty <= config.POSITION_QTY_EPSILON and pos_after_entry and pos_after_entry['side'] == signal.upper():
                        logger.warning(f"{Fore.YELLOW}Entry order filled quantity seems zero ({filled_qty_order} / {filled_qty_pos}). Using target size {position_size} for SL based on position side confirmation.{Style.RESET_ALL}")
                        filled_qty = position_size

                    if pos_after_entry and pos_after_entry['side'] == signal.upper() and filled_qty > config.POSITION_QTY_EPSILON:
                        pos_qty_formatted = format_amount(exchange, config.SYMBOL, pos_after_entry['qty'])
                        logger.info(f"{Fore.GREEN}Position confirmed open: {pos_after_entry['side']} {pos_qty_formatted}{Style.RESET_ALL}")

                        # Place Stop Loss (Await async call)
                        sl_side = config.SIDE_SELL if signal == config.SIDE_BUY else config.SIDE_BUY
                        logger.debug(f"{Fore.CYAN}# Awaiting native stop-loss placement...{Style.RESET_ALL}")
                        # Assuming place_native_stop_loss IS async
                        sl_order = await bybit.place_native_stop_loss(exchange, config.SYMBOL, sl_side, filled_qty, stop_loss_price, config)

                        if sl_order and sl_order.get('id'):
                             sl_id_short = format_order_id(sl_order['id'])
                             logger.success(f"{Fore.GREEN}Native stop-loss order placed successfully. ID: ...{sl_id_short}{Style.RESET_ALL}")
                             stop_loss_orders[config.SYMBOL] = sl_order['id']
                             if config.ENABLE_SMS_ALERTS:
                                 entry_price_fmt = format_price(exchange, config.SYMBOL, current_price)
                                 sl_price_fmt = format_price(exchange, config.SYMBOL, stop_loss_price)
                                 filled_qty_fmt = format_amount(exchange, config.SYMBOL, filled_qty)
                                 send_sms_alert(f"[{config.SYMBOL}] Entered {signal.upper()} {filled_qty_fmt} @ {entry_price_fmt}. SL @ {sl_price_fmt}", config)
                        else:
                             logger.error(f"{Back.RED}{Fore.WHITE}Failed to place stop-loss order after entry! Attempting to close position immediately.{Style.RESET_ALL}")
                             if config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{config.SYMBOL}] URGENT: Failed to place SL after {signal.upper()} entry! Closing position.", config)
                             logger.debug(f"{Fore.CYAN}# Awaiting emergency position close...{Style.RESET_ALL}")
                             close_order = await bybit.close_position_reduce_only(exchange, config.SYMBOL, config, reason="Failed SL Placement")
                             if close_order: logger.warning(f"{Fore.YELLOW}Position closed due to failed SL placement.{Style.RESET_ALL}")
                             else: logger.critical(f"{Back.RED}{Fore.WHITE}CRITICAL: FAILED TO CLOSE POSITION AFTER FAILED SL PLACEMENT!{Style.RESET_ALL}")
                    else:
                        pos_side_report = pos_after_entry['side'] if pos_after_entry else 'N/A'
                        logger.error(f"{Fore.RED}Entry order submitted ({order_id_short}) but position confirmation failed or quantity is zero. Pos Side: {pos_side_report}, Filled Qty from order: {filled_qty_order}, Pos Qty: {filled_qty_pos}{Style.RESET_ALL}")
                        if config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{config.SYMBOL}] URGENT: Entry order {order_id_short} confirmation failed!", config)

                else:
                    logger.error(f"{Fore.RED}Entry market order placement failed.{Style.RESET_ALL}")

            # --- 6. Wait for next cycle (Use asyncio.sleep) ---
            logger.info(f"Cycle complete. Waiting {config.LOOP_DELAY_SECONDS} seconds...")
            await asyncio.sleep(config.LOOP_DELAY_SECONDS)

        except ccxt.NetworkError as e:
            logger.warning(f"{Fore.YELLOW}Network Error occurred in main loop: {e}. Retrying after delay...{Style.RESET_ALL}")
            await asyncio.sleep(config.LOOP_DELAY_SECONDS * 2)
        except ccxt.ExchangeError as e:
            logger.error(f"{Fore.RED}Exchange Error occurred in main loop: {e}. Retrying after delay...{Style.RESET_ALL}")
            if config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{config.SYMBOL}] Exchange Error: {e}", config)
            await asyncio.sleep(config.LOOP_DELAY_SECONDS)
        except KeyboardInterrupt:
            logger.warning(f"{Fore.YELLOW}Keyboard interrupt received. Shutting down gracefully...{Style.RESET_ALL}")
            break
        except Exception as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}!!! UNEXPECTED CRITICAL ERROR IN MAIN LOOP !!!{Style.RESET_ALL}", exc_info=True)
            logger.critical(f"{Back.RED}{Fore.WHITE}Error Type: {type(e).__name__}, Message: {e}{Style.RESET_ALL}")
            if config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{config.SYMBOL}] CRITICAL ERROR: {type(e).__name__}. Check logs!", config)
            logger.info(f"{Fore.YELLOW}Attempting to continue after critical error... pausing for {config.LOOP_DELAY_SECONDS * 3}s{Style.RESET_ALL}")
            await asyncio.sleep(config.LOOP_DELAY_SECONDS * 3)

    logger.info(f"{Fore.MAGENTA}--- Ehlers Volumetric Strategy Stopping ---{Style.RESET_ALL}")


# --- Asynchronous Main Function ---
async def main():
    global logger, exchange, CONFIG # Make globals accessible

    # --- Initialize Logger ---
    # Ensure LOG_FILE_PATH env var is set or default is used
    log_file_path = os.getenv("LOG_FILE_PATH", "ehlers_strategy.log")
    logger = setup_logger(
        logger_name="EhlersStrategy",
        log_file=log_file_path,
        console_level=logging.getLevelName(os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()),
        file_level=logging.getLevelName(os.getenv("LOG_FILE_LEVEL", "DEBUG").upper()),
        third_party_log_level=logging.WARNING
    )

    # --- Load Configuration ---
    CONFIG = Config()

    # --- Validate Config ---
    if not CONFIG.API_KEY or not CONFIG.API_SECRET:
        logger.critical(f"{Back.RED}{Fore.WHITE}API Key or Secret not found. Grant the script access! Exiting.{Style.RESET_ALL}")
        sys.exit(1)
    logger.info(f"Configuration loaded. Testnet: {CONFIG.TESTNET_MODE}, Symbol: {CONFIG.SYMBOL}")

    # --- Initialize Exchange (Synchronous) ---
    exchange = bybit.initialize_bybit(CONFIG)
    if not exchange:
        logger.critical(f"{Back.RED}{Fore.WHITE}Failed to initialize Bybit exchange. Connection failed. Exiting.{Style.RESET_ALL}")
        if CONFIG.ENABLE_SMS_ALERTS: send_sms_alert("Strategy exit: Bybit exchange initialization failed.", CONFIG)
        sys.exit(1)

    # --- Set Leverage and Validate Market ---
    setup_success = False
    try:
        # --- Set Leverage (Assume Synchronous or handle return value) ---
        logger.debug(f"{Fore.CYAN}# Setting leverage...{Style.RESET_ALL}")

        # --- IMPORTANT: Check if bybit.set_leverage is async or sync ---
        # If it's async def set_leverage(...): use await
        # If it's def set_leverage(...): do NOT use await

        # OPTION 1: Assuming set_leverage IS ASYNC and returns bool/dict or raises error
        # leverage_set_result = await bybit.set_leverage(exchange, CONFIG.SYMBOL, CONFIG.DEFAULT_LEVERAGE, CONFIG)

        # OPTION 2: Assuming set_leverage IS SYNC and returns bool/dict
        leverage_set_result = bybit.set_leverage(exchange, CONFIG.SYMBOL, CONFIG.DEFAULT_LEVERAGE, CONFIG)

        # Check the result regardless of sync/async
        if not leverage_set_result:
            # Log critical error, SMS sent within helper or here based on return
            logger.critical(f"{Back.RED}{Fore.WHITE}Leverage setting failed for {CONFIG.SYMBOL} to {CONFIG.DEFAULT_LEVERAGE}x (check helper logs for details). Exiting.{Style.RESET_ALL}")
            if CONFIG.ENABLE_SMS_ALERTS: send_sms_alert(f"Strategy exit: Failed to set leverage {CONFIG.DEFAULT_LEVERAGE}x for {CONFIG.SYMBOL}.", CONFIG)
            # Do not proceed, let finally block handle close
            raise ccxt.ExchangeError("Leverage setting failed") # Raise specific error to exit try block

        logger.info(f"{Fore.GREEN}Leverage set to {CONFIG.DEFAULT_LEVERAGE}x for {CONFIG.SYMBOL}.{Style.RESET_ALL}")

        # --- Validate Market (Assume synchronous) ---
        logger.debug(f"{Fore.CYAN}# Validating market {CONFIG.SYMBOL}...{Style.RESET_ALL}")
        market_details = bybit.validate_market(exchange, CONFIG.SYMBOL, CONFIG)
        if not market_details:
             logger.critical(f"{Back.RED}{Fore.WHITE}Market validation failed for {CONFIG.SYMBOL}. Exiting.{Style.RESET_ALL}")
             if CONFIG.ENABLE_SMS_ALERTS: send_sms_alert(f"Strategy exit: Market validation failed for {CONFIG.SYMBOL}.", CONFIG)
             raise ccxt.ExchangeError("Market validation failed") # Raise specific error

        logger.info(f"{Fore.GREEN}Market {CONFIG.SYMBOL} validated successfully.{Style.RESET_ALL}")
        setup_success = True # Mark setup as successful

    except ccxt.AuthenticationError as e:
         logger.critical(f"{Back.RED}{Fore.WHITE}Authentication Error during setup: {e}. Check API keys. Exiting.{Style.RESET_ALL}", exc_info=True)
         if CONFIG.ENABLE_SMS_ALERTS: send_sms_alert(f"Strategy exit: Authentication error during setup.", CONFIG)
    except ccxt.ExchangeError as e:
         # Catches failures from leverage or market validation if they raise ExchangeError
         logger.critical(f"{Back.RED}{Fore.WHITE}Exchange Error during setup: {e}. Exiting.{Style.RESET_ALL}", exc_info=True)
         # SMS alert likely sent already for specific failure (leverage/market)
    except TypeError as e:
        # Catch the specific error if await was used incorrectly on set_leverage
        logger.critical(f"{Back.RED}{Fore.WHITE}TypeError during setup (likely awaiting a non-async function like set_leverage?): {e}. Exiting.{Style.RESET_ALL}", exc_info=True)
        if CONFIG.ENABLE_SMS_ALERTS: send_sms_alert(f"Strategy exit: TypeError during setup.", CONFIG)
    except Exception as e:
        # Catch any other unexpected errors during setup
        logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected error during setup: {e}. Exiting.{Style.RESET_ALL}", exc_info=True)
        if CONFIG.ENABLE_SMS_ALERTS: send_sms_alert(f"Strategy exit: Unexpected error during setup.", CONFIG)

    # --- Start Strategy only if setup succeeded ---
    if setup_success:
        try:
            await run_strategy(CONFIG, exchange) # Pass the validated exchange object
        except NameError as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}A NameError occurred: {e}.{Style.RESET_ALL}", exc_info=True)
            if "Fore" in str(e) or "Back" in str(e) or "Style" in str(e):
                 logger.critical(f"{Fore.YELLOW}Ensure {Style.BRIGHT}'pip install colorama'{Style.RESET_ALL}{Fore.YELLOW} is done.{Style.RESET_ALL}")
            # SMS not needed here, critical log is sufficient
        except Exception as e:
            logger.critical(f"{Back.RED}{Fore.WHITE}Strategy execution failed with unhandled exception: {e}{Style.RESET_ALL}", exc_info=True)
            if CONFIG.ENABLE_SMS_ALERTS: send_sms_alert(f"[{CONFIG.SYMBOL}] CRITICAL FAILURE: Strategy terminated unexpectedly. Check logs!", CONFIG)
        # The finally block below will handle closing the exchange after run_strategy finishes or errors out.

    # --- Cleanup: Close Exchange Connection ---
    # This block runs regardless of whether setup_success was True or False,
    # ensuring the connection is closed if it was opened.
    logger.info(f"{Fore.CYAN}# Performing cleanup...{Style.RESET_ALL}")
    if exchange and hasattr(exchange, 'close'):
        try:
            logger.info(f"{Fore.CYAN}# Closing connection to the exchange realm...{Style.RESET_ALL}")
            await exchange.close() # exchange.close() IS typically async
            logger.info(f"{Fore.GREEN}Exchange connection closed.{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error closing exchange connection during cleanup: {e}{Style.RESET_ALL}", exc_info=True)
    else:
        logger.info(f"{Fore.YELLOW}Exchange object not available or has no close method; skipping close.{Style.RESET_ALL}")

    if not setup_success:
        logger.warning(f"{Fore.YELLOW}Exiting due to setup failure.{Style.RESET_ALL}")
        sys.exit(1) # Ensure exit code reflects failure if setup didn't complete

    logger.info(f"{Fore.MAGENTA}Strategy shutdown complete.{Style.RESET_ALL}")


if __name__ == "__main__":
    # Pyrmethus initiates the asynchronous ritual...
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Asyncio loop interrupted. Exiting.{Style.RESET_ALL}")
        # Allow finally block in main() to attempt cleanup
    except Exception as e:
        # Catch errors during asyncio.run itself if main() fails spectacularly early
        print(f"{Back.RED}{Fore.WHITE}Fatal error during asyncio execution: {e}{Style.RESET_ALL}")
        # Attempt to log if logger was initialized
        if logger:
            logger.critical(f"Fatal error during asyncio execution", exc_info=True)
        sys.exit(1)
```

**Key Changes:**

1.  **Leverage Call:** The `await` before `bybit.set_leverage` is commented out (Option 2). **You MUST verify if `set_leverage` in your `bybit_helpers.py` is defined with `async def` or just `def`.** If it's `async def`, uncomment Option 1 and remove Option 2. If it's `def`, keep Option 2.
2.  **Robust Setup Logic:** The `try...except` block in `main` now specifically handles the flow:
    *   It attempts leverage setting and market validation.
    *   If `set_leverage` returns a "falsy" value (like `False` or `None`), it logs the failure and raises a specific `ccxt.ExchangeError` to cleanly exit the `try` block.
    *   It catches specific `ccxt` errors, the potential `TypeError` from incorrect `await`, and general `Exception`.
    *   A `setup_success` flag is used to track if initialization completed successfully.
3.  **Conditional Strategy Start:** `run_strategy` is only called if `setup_success` is `True`.
4.  **Unified Cleanup:** The `exchange.close()` call is moved to a final block *after* the setup `try...except` and the `run_strategy` call. This ensures it *always* attempts to close the connection if `exchange` was initialized, regardless of where a failure occurred (setup or runtime). It also includes a check (`if exchange and hasattr(exchange, 'close')`) before attempting the close.
5.  **Safe Globals:** Added `Optional` typing to global `exchange` and `CONFIG` and safety checks where they are used before being potentially assigned (like in `calculate_stop_loss`).

**Action Required:**

1.  **Verify `bybit.set_leverage`:** Check its definition in `bybit_helpers.py`. Is it `async def` or `def`? Adjust the call in `main()` accordingly (use or remove `await`).
2.  **Replace Code:** Update your script with this new version.
3.  **Test:** Run the script again. Observe the logs carefully during the setup phase.

This refined spell should correctly navigate the setup process, handle potential synchronous/asynchronous mismatches in the leverage setting, and ensure a clean exit path, banishing the `TypeError`.