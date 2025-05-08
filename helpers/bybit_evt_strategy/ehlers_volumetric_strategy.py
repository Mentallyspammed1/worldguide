# --- START OF FILE ehlers_volumetric_strategy.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ehlers Volumetric Trend Strategy for Bybit V5 (v1.3 - Class, TP, Order Mgmt)

This script implements a trading strategy based on the Ehlers Volumetric Trend
indicator using the Bybit V5 API via CCXT. It leverages custom helper modules
for exchange interaction, indicator calculation, logging, and utilities.

Strategy Logic:
- Uses Ehlers Volumetric Trend (EVT) for primary entry signals.
- Enters LONG on EVT bullish trend initiation.
- Enters SHORT on EVT bearish trend initiation.
- Exits positions when the EVT trend reverses.
- Uses ATR-based stop-loss and take-profit orders (placed as reduce-only limit orders).
- Manages position size based on risk percentage.
- Includes error handling, retries, and rate limit awareness via helper modules.
- Encapsulated within an EhlersStrategy class.
"""

import os
import sys
import time
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP # Import ROUND_UP for TP
from typing import Optional, Dict, Tuple, Any, Literal

# Third-party libraries
try:
    import ccxt
except ImportError: print("FATAL: CCXT library not found.", file=sys.stderr); sys.exit(1)
try:
    import pandas as pd
except ImportError: print("FATAL: pandas library not found.", file=sys.stderr); sys.exit(1)
try:
    from dotenv import load_dotenv
except ImportError: print("Warning: python-dotenv not found. Cannot load .env file.", file=sys.stderr); load_dotenv = lambda: None # Dummy function

# --- Import Colorama for main script logging ---
try:
    from colorama import Fore, Style, Back, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    print("Warning: 'colorama' library not found. Main script logs will not be colored.", file=sys.stderr)
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor() # type: ignore


# --- Import Custom Modules ---
try:
    from neon_logger import setup_logger
    import bybit_helper_functions as bybit_helpers # Import the module itself
    import indicators as ind
    from bybit_utils import (
        safe_decimal_conversion, format_price, format_amount,
        format_order_id, send_sms_alert # Sync SMS alert here
    )
    # Import config models
    from config_models import AppConfig, APIConfig, StrategyConfig # Import specific models needed
except ImportError as e:
    print(f"FATAL: Error importing helper modules: {e}", file=sys.stderr)
    print("Ensure all .py files (config_models, neon_logger, bybit_utils, bybit_helper_functions, indicators) are present.", file=sys.stderr)
    sys.exit(1)

# --- Load Environment Variables ---
# load_dotenv() # Called in main.py usually, after logger setup

# --- Logger Placeholder ---
# Logger configured in main block
logger: logging.Logger = logging.getLogger(__name__) # Get logger by name

# --- Strategy Class ---
class EhlersStrategy:
    """Encapsulates the Ehlers Volumetric Trend trading strategy logic."""

    def __init__(self, config: AppConfig): # Use AppConfig type hint
        self.app_config = config
        self.config = config.to_legacy_config_dict() # Convert to legacy dict for internal use IF NEEDED
                                                     # Better: Update class to use AppConfig directly
        self.api_config: APIConfig = config.api_config # Direct access
        self.strategy_config: StrategyConfig = config.strategy_config # Direct access

        self.symbol = self.api_config.symbol
        self.timeframe = self.strategy_config.timeframe
        self.exchange: Optional[ccxt.bybit] = None # Will be initialized (Sync version)
        self.bybit_helpers = bybit_helpers # Store module for access
        self.is_initialized = False
        self.is_running = False

        # Position State
        self.current_side: str = self.api_config.pos_none
        self.current_qty: Decimal = Decimal("0.0")
        self.entry_price: Optional[Decimal] = None
        self.sl_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None

        # Market details
        self.min_qty: Optional[Decimal] = None
        self.qty_step: Optional[Decimal] = None
        self.price_tick: Optional[Decimal] = None

        logger.info(f"EhlersStrategy initialized for {self.symbol} on {self.timeframe}.")

    def _initialize(self) -> bool:
        """Connects to the exchange, validates market, sets config, fetches initial state."""
        logger.info(f"{Fore.CYAN}--- Strategy Initialization Phase ---{Style.RESET_ALL}")
        try:
            # Pass the main AppConfig object to helpers
            self.exchange = self.bybit_helpers.initialize_bybit(self.app_config)
            if not self.exchange: return False

            market_details = self.bybit_helpers.validate_market(self.exchange, self.symbol, self.app_config)
            if not market_details: return False
            self._extract_market_details(market_details)

            logger.info(f"Setting leverage for {self.symbol} to {self.strategy_config.leverage}x...")
            # Pass AppConfig
            if not self.bybit_helpers.set_leverage(self.exchange, self.symbol, self.strategy_config.leverage, self.app_config):
                 logger.critical(f"{Back.RED}Failed to set leverage.{Style.RESET_ALL}")
                 return False
            logger.success("Leverage set/confirmed.")

            # Set Position Mode (One-Way) - Optional but recommended for clarity
            pos_mode = self.strategy_config.default_position_mode
            logger.info(f"Attempting to set position mode to '{pos_mode}'...")
            # Pass AppConfig
            mode_set = self.bybit_helpers.set_position_mode_bybit_v5(self.exchange, self.symbol, pos_mode, self.app_config)
            if not mode_set:
                 logger.warning(f"{Fore.YELLOW}Could not explicitly set position mode to '{pos_mode}'. Ensure it's set correctly in Bybit UI.{Style.RESET_ALL}")
            else:
                 logger.info(f"Position mode confirmed/set to '{pos_mode}'.")

            logger.info("Fetching initial account state (position, orders, balance)...")
            # Pass AppConfig
            if not self._update_state():
                 logger.error("Failed to fetch initial state.")
                 return False

            logger.info(f"Initial Position: Side={self.current_side}, Qty={self.current_qty}")

            logger.info("Performing initial cleanup: cancelling existing orders...")
            # Pass AppConfig
            if not self._cancel_open_orders("Initialization Cleanup"):
                 logger.warning("Initial order cancellation failed or encountered issues.")

            self.is_initialized = True
            logger.success(f"{Fore.GREEN}--- Strategy Initialization Complete ---{Style.RESET_ALL}")
            return True

        except Exception as e:
            logger.critical(f"{Back.RED}Critical error during strategy initialization: {e}{Style.RESET_ALL}", exc_info=True)
            # Clean up exchange connection if partially initialized
            if self.exchange and hasattr(self.exchange, 'close'):
                try: self.exchange.close()
                except Exception: pass # Ignore errors during cleanup close
            return False

    def _extract_market_details(self, market: Dict):
        """Extracts and stores relevant market limits and precision."""
        limits = market.get('limits', {})
        precision = market.get('precision', {})

        self.min_qty = safe_decimal_conversion(limits.get('amount', {}).get('min'))
        amount_precision = precision.get('amount') # Number of decimal places for amount
        # Qty step is usually 10^-precision
        self.qty_step = (Decimal('1') / (Decimal('10') ** int(amount_precision))) if amount_precision is not None else None

        price_precision = precision.get('price') # Number of decimal places for price
        # Price tick is usually 10^-precision
        self.price_tick = (Decimal('1') / (Decimal('10') ** int(price_precision))) if price_precision is not None else None

        logger.info(f"Market Details Set: Min Qty={self.min_qty}, Qty Step={self.qty_step}, Price Tick={self.price_tick}")

    def _update_state(self) -> bool:
        """Fetches and updates the current position, balance, and open orders."""
        logger.debug("Updating strategy state...")
        try:
            # Fetch Position - Pass AppConfig
            pos_data = self.bybit_helpers.get_current_position_bybit_v5(self.exchange, self.symbol, self.app_config)
            if pos_data is None: logger.error("Failed to fetch position data."); return False

            self.current_side = pos_data['side']
            self.current_qty = pos_data['qty']
            self.entry_price = pos_data.get('entry_price') # Can be None if no position

            # Fetch Balance - Pass AppConfig
            balance_info = self.bybit_helpers.fetch_usdt_balance(self.exchange, self.app_config)
            if balance_info is None: logger.error("Failed to fetch balance data."); return False
            _, available_balance = balance_info # Unpack equity, available
            logger.info(f"Available Balance: {available_balance:.4f} {self.api_config.usdt_symbol}")

            # If not in position, reset tracked orders
            if self.current_side == self.api_config.pos_none:
                if self.sl_order_id or self.tp_order_id:
                     logger.debug("Not in position, clearing tracked SL/TP order IDs.")
                     self.sl_order_id = None
                     self.tp_order_id = None
            # Optional: If in position, verify tracked SL/TP orders still exist and are open
            # else: self._verify_open_sl_tp()

            logger.debug("State update complete.")
            return True

        except Exception as e:
            logger.error(f"Unexpected error during state update: {e}", exc_info=True)
            return False

    # Optional verification function
    # def _verify_open_sl_tp(self):
    #     """Checks if tracked SL/TP orders are still open."""
    #     if self.sl_order_id:
    #         sl_order = self.bybit_helpers.fetch_order(self.exchange, self.symbol, self.sl_order_id, self.app_config, order_filter='StopOrder') # Adjust filter if needed
    #         if not sl_order or sl_order.get('status') != 'open':
    #              logger.warning(f"Tracked SL order {self.sl_order_id} is no longer open/found. Clearing ID.")
    #              self.sl_order_id = None
    #     if self.tp_order_id:
    #          tp_order = self.bybit_helpers.fetch_order(self.exchange, self.symbol, self.tp_order_id, self.app_config, order_filter='Order') # Assuming TP is limit
    #          if not tp_order or tp_order.get('status') != 'open':
    #              logger.warning(f"Tracked TP order {self.tp_order_id} is no longer open/found. Clearing ID.")
    #              self.tp_order_id = None


    def _fetch_data(self) -> Tuple[Optional[pd.DataFrame], Optional[Decimal]]:
        """Fetches OHLCV data and the latest ticker price."""
        logger.debug("Fetching market data...")
        # Pass AppConfig
        ohlcv_df = self.bybit_helpers.fetch_ohlcv_paginated(
            exchange=self.exchange,
            symbol=self.symbol,
            timeframe=self.timeframe,
            config=self.app_config, # Pass main config
            max_total_candles=self.strategy_config.ohlcv_limit
        )
        if ohlcv_df is None or not isinstance(ohlcv_df, pd.DataFrame) or ohlcv_df.empty:
            logger.warning("Could not fetch sufficient OHLCV data.")
            return None, None

        # Pass AppConfig
        ticker = self.bybit_helpers.fetch_ticker_validated(self.exchange, self.symbol, self.app_config)
        if ticker is None:
            logger.warning("Could not fetch valid ticker data.")
            return ohlcv_df, None # Return OHLCV if available, but no price

        current_price = ticker.get('last') # Already Decimal from helper
        if current_price is None:
            logger.warning("Ticker data retrieved but missing 'last' price.")
            return ohlcv_df, None

        logger.debug(f"Data fetched: {len(ohlcv_df)} candles, Last Price: {current_price}")
        return ohlcv_df, current_price

    def _calculate_indicators(self, ohlcv_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates indicators based on the config."""
        if ohlcv_df is None or ohlcv_df.empty: return None
        logger.debug("Calculating indicators...")
        # Prepare config dict expected by indicators module (if it uses dict)
        # Or pass AppConfig directly if indicators module supports it
        indicator_config_dict = {
            "indicator_settings": self.strategy_config.indicator_settings.model_dump(),
            "analysis_flags": self.strategy_config.analysis_flags.model_dump(),
            "strategy_params": self.strategy_config.strategy_params,
            "strategy": self.strategy_config.strategy_info
        }
        # df_with_indicators = ind.calculate_all_indicators(ohlcv_df, self.app_config) # If module accepts AppConfig
        df_with_indicators = ind.calculate_all_indicators(ohlcv_df, indicator_config_dict) # If module expects dict


        # Validate necessary columns exist
        evt_len = self.strategy_config.indicator_settings.evt_length
        atr_len = self.strategy_config.indicator_settings.atr_period
        evt_trend_col = f'evt_trend_{evt_len}'
        atr_col = f'ATRr_{atr_len}' # pandas_ta default name

        if df_with_indicators is None:
            logger.error("Indicator calculation returned None.")
            return None
        if evt_trend_col not in df_with_indicators.columns:
            logger.error(f"Required EVT trend column '{evt_trend_col}' not found after calculation.")
            return None
        if self.strategy_config.analysis_flags.use_atr and atr_col not in df_with_indicators.columns:
             logger.error(f"Required ATR column '{atr_col}' not found after calculation (use_atr is True).")
             return None

        logger.debug("Indicators calculated successfully.")
        return df_with_indicators

    def _generate_signals(self, df_ind: pd.DataFrame) -> Optional[Literal['buy', 'sell']]:
        """Generates trading signals based on the last indicator data point."""
        if df_ind is None or df_ind.empty: return None
        logger.debug("Generating trading signals...")
        try:
            latest = df_ind.iloc[-1]
            evt_len = self.strategy_config.indicator_settings.evt_length
            trend_col = f'evt_trend_{evt_len}'
            buy_col = f'evt_buy_{evt_len}'
            sell_col = f'evt_sell_{evt_len}'

            if not all(col in latest.index and pd.notna(latest[col]) for col in [trend_col, buy_col, sell_col]):
                 logger.warning(f"EVT signal columns missing or NaN in latest data: {latest[[trend_col, buy_col, sell_col]].to_dict()}")
                 return None

            buy_signal = latest[buy_col]
            sell_signal = latest[sell_col]

            # Return 'buy'/'sell' string consistent with helper functions
            if buy_signal:
                logger.info(f"{Fore.GREEN}BUY signal generated based on EVT Buy flag.{Style.RESET_ALL}")
                return self.api_config.side_buy # 'buy'
            elif sell_signal:
                logger.info(f"{Fore.RED}SELL signal generated based on EVT Sell flag.{Style.RESET_ALL}")
                return self.api_config.side_sell # 'sell'
            else:
                # Check for exit signal based purely on trend reversal
                current_trend = int(latest[trend_col])
                if self.current_side == self.api_config.pos_long and current_trend == -1:
                     logger.info(f"{Fore.YELLOW}EXIT LONG signal generated (Trend flipped Short).{Style.RESET_ALL}")
                     # Strategy logic handles exit separately, signal generator focuses on entry
                elif self.current_side == self.api_config.pos_short and current_trend == 1:
                     logger.info(f"{Fore.YELLOW}EXIT SHORT signal generated (Trend flipped Long).{Style.RESET_ALL}")
                     # Strategy logic handles exit separately

                logger.debug("No new entry signal generated.")
                return None

        except IndexError:
            logger.warning("IndexError generating signals (DataFrame likely too short).")
            return None
        except Exception as e:
            logger.error(f"Error generating signals: {e}", exc_info=True)
            return None

    def _calculate_sl_tp(self, df_ind: pd.DataFrame, side: Literal['buy', 'sell'], entry_price: Decimal) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculates initial stop-loss and take-profit prices."""
        if df_ind is None or df_ind.empty: return None, None
        logger.debug(f"Calculating SL/TP for {side} entry at {entry_price}...")
        try:
            atr_len = self.strategy_config.indicator_settings.atr_period
            atr_col = f'ATRr_{atr_len}'
            if atr_col not in df_ind.columns: logger.error(f"ATR column '{atr_col}' not found."); return None, None

            latest_atr = safe_decimal_conversion(df_ind.iloc[-1][atr_col])
            if latest_atr is None or latest_atr <= Decimal(0):
                logger.warning(f"Invalid ATR value ({latest_atr}) for SL/TP calculation.")
                return None, None # Require valid ATR

            # Stop Loss Calculation
            sl_multiplier = self.strategy_config.stop_loss_atr_multiplier
            sl_offset = latest_atr * sl_multiplier
            stop_loss_price = (entry_price - sl_offset) if side == self.api_config.side_buy else (entry_price + sl_offset)

            # --- Price Tick Adjustment ---
            # Adjust SL/TP to the nearest valid price tick
            if self.price_tick is None:
                 logger.warning("Price tick size unknown. Cannot adjust SL/TP precisely.")
                 sl_price_adjusted = stop_loss_price
            else:
                 # Round SL "away" from entry (more conservative)
                 rounding_mode = ROUND_DOWN if side == self.api_config.side_buy else ROUND_UP
                 sl_price_adjusted = (stop_loss_price / self.price_tick).quantize(Decimal('1'), rounding=rounding_mode) * self.price_tick

            # Ensure SL didn't cross entry after rounding
            if side == self.api_config.side_buy and sl_price_adjusted >= entry_price:
                 sl_price_adjusted = entry_price - self.price_tick if self.price_tick else entry_price * Decimal("0.999")
                 logger.warning(f"Adjusted Buy SL >= entry. Setting SL just below entry: {sl_price_adjusted}")
            elif side == self.api_config.side_sell and sl_price_adjusted <= entry_price:
                 sl_price_adjusted = entry_price + self.price_tick if self.price_tick else entry_price * Decimal("1.001")
                 logger.warning(f"Adjusted Sell SL <= entry. Setting SL just above entry: {sl_price_adjusted}")


            # Take Profit Calculation
            tp_multiplier = self.strategy_config.take_profit_atr_multiplier
            tp_price_adjusted = None
            if tp_multiplier > 0:
                tp_offset = latest_atr * tp_multiplier
                take_profit_price = (entry_price + tp_offset) if side == self.api_config.side_buy else (entry_price - tp_offset)

                # Ensure TP is logical relative to entry BEFORE rounding
                if side == self.api_config.side_buy and take_profit_price <= entry_price:
                    logger.warning(f"Calculated Buy TP ({take_profit_price}) <= entry ({entry_price}). Skipping TP.")
                elif side == self.api_config.side_sell and take_profit_price >= entry_price:
                    logger.warning(f"Calculated Sell TP ({take_profit_price}) >= entry ({entry_price}). Skipping TP.")
                else:
                    # Adjust TP to nearest tick, round "towards" entry (more conservative fill chance)
                    if self.price_tick is None:
                        logger.warning("Price tick size unknown. Cannot adjust TP precisely.")
                        tp_price_adjusted = take_profit_price
                    else:
                        rounding_mode = ROUND_DOWN if side == self.api_config.side_buy else ROUND_UP
                        tp_price_adjusted = (take_profit_price / self.price_tick).quantize(Decimal('1'), rounding=rounding_mode) * self.price_tick

                    # Ensure TP didn't cross entry after rounding
                    if side == self.api_config.side_buy and tp_price_adjusted <= entry_price:
                         tp_price_adjusted = entry_price + self.price_tick if self.price_tick else entry_price * Decimal("1.001")
                         logger.warning(f"Adjusted Buy TP <= entry. Setting TP just above entry: {tp_price_adjusted}")
                    elif side == self.api_config.side_sell and tp_price_adjusted >= entry_price:
                         tp_price_adjusted = entry_price - self.price_tick if self.price_tick else entry_price * Decimal("0.999")
                         logger.warning(f"Adjusted Sell TP >= entry. Setting TP just below entry: {tp_price_adjusted}")
            else:
                logger.info("Take Profit multiplier is zero or less. Skipping TP calculation.")


            logger.info(f"Calculated SL: {format_price(self.exchange, self.symbol, sl_price_adjusted)}, "
                        f"TP: {format_price(self.exchange, self.symbol, tp_price_adjusted) or 'None'} (ATR: {latest_atr:.4f})")
            return sl_price_adjusted, tp_price_adjusted

        except IndexError: logger.warning("IndexError calculating SL/TP."); return None, None
        except Exception as e: logger.error(f"Error calculating SL/TP: {e}", exc_info=True); return None, None

    def _calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal) -> Optional[Decimal]:
        """Calculates position size based on risk percentage and stop-loss distance."""
        logger.debug("Calculating position size...")
        try:
            # Pass AppConfig
            balance_info = self.bybit_helpers.fetch_usdt_balance(self.exchange, self.app_config)
            if balance_info is None: logger.error("Cannot calc size: Failed fetch balance."); return None
            _, available_balance = balance_info

            if available_balance is None or available_balance <= Decimal("0"):
                logger.error("Cannot calculate position size: Zero or invalid available balance.")
                return None

            risk_amount_usd = available_balance * self.strategy_config.risk_per_trade
            price_diff = abs(entry_price - stop_loss_price)

            if price_diff <= Decimal("0"):
                logger.error(f"Cannot calculate size: Entry price ({entry_price}) and SL price ({stop_loss_price}) invalid or equal.")
                return None

            # Calculate size based on risk amount and price difference per unit
            # For linear contracts (Value = Amount * Price), Size = Risk / PriceDiff
            # For inverse contracts (Value = Amount / Price), need careful derivation
            market = self.exchange.market(self.symbol)
            is_inverse = market.get('inverse', False)

            position_size_base: Decimal
            if is_inverse:
                 # Risk = Size * abs(1/Entry - 1/SL) => Size = Risk / abs(1/Entry - 1/SL)
                 if entry_price <= 0 or stop_loss_price <= 0: raise ValueError("Prices must be positive for inverse size calc.")
                 size_denominator = abs(Decimal(1)/entry_price - Decimal(1)/stop_loss_price)
                 if size_denominator <= 0: raise ValueError("Inverse size denominator is zero.")
                 position_size_base = risk_amount_usd / size_denominator
            else: # Linear contract
                 position_size_base = risk_amount_usd / price_diff

            # Apply market precision/step size constraints
            if self.qty_step is None:
                 logger.warning("Quantity step size unknown, cannot adjust size precisely.")
                 position_size_adjusted = position_size_base # Use raw value
            else:
                 # Round down to the nearest step size increment
                 position_size_adjusted = (position_size_base // self.qty_step) * self.qty_step

            if position_size_adjusted <= Decimal(0):
                 logger.warning(f"Adjusted position size is zero or negative. Step: {self.qty_step}, Orig: {position_size_base}")
                 return None

            if self.min_qty is not None and position_size_adjusted < self.min_qty:
                logger.warning(f"Calculated size ({position_size_adjusted}) < Min Qty ({self.min_qty}). Cannot trade this size.")
                # Option: Round up to min_qty if desired, but this increases risk.
                # position_size_adjusted = self.min_qty
                # logger.warning(f"Adjusting size up to Min Qty ({self.min_qty}). Risk will be higher.")
                return None # Default: Don't trade if calculated size is too small

            logger.info(f"Calculated position size: {format_amount(self.exchange, self.symbol, position_size_adjusted)} "
                        f"(Risk: {risk_amount_usd:.2f} USDT, Balance: {available_balance:.2f} USDT)")
            return position_size_adjusted

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return None

    def _cancel_open_orders(self, reason: str = "Strategy Action") -> bool:
        """Cancels tracked SL and TP orders."""
        cancelled_sl, cancelled_tp = True, True # Assume success if no ID tracked
        all_success = True

        # Determine order filter based on placement type
        sl_filter = 'StopOrder' if not self.strategy_config.place_tpsl_as_limit else 'Order'
        tp_filter = 'Order' # TP is always limit

        if self.sl_order_id:
            logger.info(f"Cancelling existing SL order {format_order_id(self.sl_order_id)} ({reason})...")
            try:
                # Pass AppConfig and filter
                cancelled_sl = self.bybit_helpers.cancel_order(self.exchange, self.symbol, self.sl_order_id, self.app_config, order_filter=sl_filter)
                if cancelled_sl: logger.info("SL order cancelled successfully or already gone.")
                else: logger.warning("Failed attempt to cancel SL order."); all_success = False
            except Exception as e:
                 logger.error(f"Error cancelling SL order {self.sl_order_id}: {e}", exc_info=True)
                 cancelled_sl = False; all_success = False
            finally:
                 self.sl_order_id = None # Always clear tracked ID after attempt

        if self.tp_order_id:
             logger.info(f"Cancelling existing TP order {format_order_id(self.tp_order_id)} ({reason})...")
             try:
                 # Pass AppConfig and filter
                 cancelled_tp = self.bybit_helpers.cancel_order(self.exchange, self.symbol, self.tp_order_id, self.app_config, order_filter=tp_filter)
                 if cancelled_tp: logger.info("TP order cancelled successfully or already gone.")
                 else: logger.warning("Failed attempt to cancel TP order."); all_success = False
             except Exception as e:
                  logger.error(f"Error cancelling TP order {self.tp_order_id}: {e}", exc_info=True)
                  cancelled_tp = False; all_success = False
             finally:
                  self.tp_order_id = None # Always clear tracked ID after attempt

        return all_success # Return overall success

    def _handle_exit(self, df_ind: pd.DataFrame) -> bool:
        """Checks exit conditions and closes the position if necessary."""
        if self.current_side == self.api_config.pos_none: return False # Not in position

        logger.debug("Checking exit conditions...")
        should_exit = False
        exit_reason = ""
        try:
            evt_len = self.strategy_config.indicator_settings.evt_length
            latest_trend = df_ind.iloc[-1].get(f'evt_trend_{evt_len}')

            if latest_trend is not None:
                latest_trend = int(latest_trend) # Ensure integer comparison
                if self.current_side == self.api_config.pos_long and latest_trend == -1:
                    should_exit = True; exit_reason = "EVT Trend flipped Short"
                elif self.current_side == self.api_config.pos_short and latest_trend == 1:
                    should_exit = True; exit_reason = "EVT Trend flipped Long"
            else:
                logger.warning("Cannot determine latest EVT trend for exit check.")

            # --- Add check for SL/TP Hit ---
            # This requires fetching order status, adds latency. Only do if essential.
            # Example (needs refinement based on order types):
            # if not should_exit and self.sl_order_id:
            #    sl_filter = 'StopOrder' if not self.strategy_config.place_tpsl_as_limit else 'Order'
            #    sl_status = self.bybit_helpers.fetch_order(self.exchange, self.symbol, self.sl_order_id, self.app_config, order_filter=sl_filter)
            #    if sl_status and sl_status.get
