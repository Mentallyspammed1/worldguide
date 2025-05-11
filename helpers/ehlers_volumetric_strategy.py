# --- START OF FILE ehlers_volumetric_strategy.py --- continuation

# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ehlers Volumetric Trend Strategy for Bybit V5 (v1.3 - Class, TP, Order Mgmt)

This script implements a trading strategy based on the Ehlers Volumetric Trend
indicator using the Bybit V5 API via CCXT. It leverages custom helper modules
for exchange interaction, indicator calculation, logging, and utilities.

Strategy Logic:
- Uses Ehlers Volumetric Trend (EVT) for primary entry signals.
- Enters LONG on EVT bullish trend initiation.
- Enters SHORT on EVT bearish trend initiation.
- Exits positions when the EVT trend reverses.
- Uses ATR-based stop-loss and take-profit orders (placed as reduce-only limit orders or native stops).
- Manages position size based on risk percentage.
- Includes error handling, retries, and rate limit awareness via helper modules.
- Encapsulated within an EhlersStrategy class.
"""

import logging
import sys
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal, getcontext
from typing import Literal

# Third-party libraries
try:
    import ccxt
except ImportError:
    print("FATAL: CCXT library not found.", file=sys.stderr)
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    print("FATAL: pandas library not found.", file=sys.stderr)
    sys.exit(1)
try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not found. Cannot load .env file.", file=sys.stderr)

    def load_dotenv(*args, **kwargs):
        return False  # Dummy function


# --- Import Colorama for main script logging ---
try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init

    colorama_init(autoreset=True)
except ImportError:
    print(
        "Warning: 'colorama' library not found. Main script logs will not be colored.",
        file=sys.stderr,
    )

    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = Back = DummyColor()  # type: ignore


# --- Import Custom Modules ---
try:
    import bybit_helper_functions as bybit_helpers  # Import the module itself
    import indicators as ind
    from bybit_utils import (
        format_amount,
        format_order_id,  # Sync SMS alert here
        format_price,
        safe_decimal_conversion,
        send_sms_alert,
    )

    # Import config models
    from config_models import (  # Import specific models needed + loader
        APIConfig,
        AppConfig,
        StrategyConfig,
        load_config,
    )
    from neon_logger import setup_logger
except ImportError as e:
    print(f"FATAL: Error importing helper modules: {e}", file=sys.stderr)
    print(
        "Ensure all .py files (config_models, neon_logger, bybit_utils, bybit_helper_functions, indicators) are present.",
        file=sys.stderr,
    )
    sys.exit(1)

# Set Decimal context precision
getcontext().prec = 28

# --- Logger Placeholder ---
# Logger configured in main block
logger: logging.Logger = logging.getLogger(__name__)  # Get logger by name


# --- Strategy Class ---
class EhlersStrategy:
    """Encapsulates the Ehlers Volumetric Trend trading strategy logic."""

    def __init__(self, config: AppConfig):  # Use AppConfig type hint
        self.app_config = config
        # Direct access to nested config models is preferred
        self.api_config: APIConfig = config.api_config
        self.strategy_config: StrategyConfig = config.strategy_config
        self.sms_config = config.sms_config  # For SMS alerts

        self.symbol = self.api_config.symbol
        self.timeframe = self.strategy_config.timeframe
        self.exchange: ccxt.bybit | None = None  # Will be initialized (Sync version)
        self.bybit_helpers = bybit_helpers  # Store module for access
        self.is_initialized = False
        self.is_running = False

        # Position State
        self.current_side: str = self.api_config.pos_none
        self.current_qty: Decimal = Decimal("0.0")
        self.entry_price: Decimal | None = None
        self.sl_order_id: str | None = None
        self.tp_order_id: str | None = None

        # Market details
        self.min_qty: Decimal | None = None
        self.qty_step: Decimal | None = None
        self.price_tick: Decimal | None = None

        logger.info(
            f"EhlersStrategy initialized for {self.symbol} on {self.timeframe}."
        )

    def _initialize(self) -> bool:
        """Connects to the exchange, validates market, sets config, fetches initial state."""
        logger.info(
            f"{Fore.CYAN}--- Strategy Initialization Phase ---{Style.RESET_ALL}"
        )
        try:
            # Pass the main AppConfig object to helpers
            self.exchange = self.bybit_helpers.initialize_bybit(self.app_config)
            if not self.exchange:
                return False

            market_details = self.bybit_helpers.validate_market(
                self.exchange, self.symbol, self.app_config
            )
            if not market_details:
                return False
            self._extract_market_details(market_details)

            logger.info(
                f"Setting leverage for {self.symbol} to {self.strategy_config.leverage}x..."
            )
            # Pass AppConfig
            if not self.bybit_helpers.set_leverage(
                self.exchange,
                self.symbol,
                self.strategy_config.leverage,
                self.app_config,
            ):
                logger.critical(f"{Back.RED}Failed to set leverage.{Style.RESET_ALL}")
                return False
            logger.success("Leverage set/confirmed.")

            # Set Position Mode (One-Way) - Optional but recommended for clarity
            pos_mode = self.strategy_config.default_position_mode
            logger.info(f"Attempting to set position mode to '{pos_mode}'...")
            # Pass AppConfig
            mode_set = self.bybit_helpers.set_position_mode_bybit_v5(
                self.exchange, self.symbol, pos_mode, self.app_config
            )
            if not mode_set:
                logger.warning(
                    f"{Fore.YELLOW}Could not explicitly set position mode to '{pos_mode}'. Ensure it's set correctly in Bybit UI.{Style.RESET_ALL}"
                )
            else:
                logger.info(f"Position mode confirmed/set to '{pos_mode}'.")

            # Set Margin Mode if required (e.g., isolated)
            if self.strategy_config.default_margin_mode == "isolated":
                logger.info(
                    f"Attempting to set ISOLATED margin mode for {self.symbol}..."
                )
                # Pass AppConfig
                margin_set = self.bybit_helpers.set_isolated_margin_bybit_v5(
                    self.exchange,
                    self.symbol,
                    self.strategy_config.leverage,
                    self.app_config,
                )
                if not margin_set:
                    logger.warning(
                        f"{Fore.YELLOW}Could not set ISOLATED margin mode. Ensure account allows it and no conflicting settings.{Style.RESET_ALL}"
                    )
                else:
                    logger.info(
                        f"Isolated margin mode confirmed/set for {self.symbol}."
                    )
            else:
                logger.info(
                    f"Using default margin mode: {self.strategy_config.default_margin_mode}"
                )

            logger.info("Fetching initial account state (position, orders, balance)...")
            # Pass AppConfig
            if not self._update_state():
                logger.error("Failed to fetch initial state.")
                return False

            logger.info(
                f"Initial Position: Side={self.current_side}, Qty={self.current_qty}"
            )

            logger.info("Performing initial cleanup: cancelling existing orders...")
            # Pass AppConfig
            if not self._cancel_all_open_orders("Initialization Cleanup"):
                logger.warning(
                    "Initial order cancellation failed or encountered issues."
                )
            else:
                logger.info("Initial order cleanup successful.")

            self.is_initialized = True
            logger.success(
                f"{Fore.GREEN}--- Strategy Initialization Complete ---{Style.RESET_ALL}"
            )
            return True

        except Exception as e:
            logger.critical(
                f"{Back.RED}Critical error during strategy initialization: {e}{Style.RESET_ALL}",
                exc_info=True,
            )
            # Clean up exchange connection if partially initialized
            if self.exchange and hasattr(self.exchange, "close"):
                try:
                    self.exchange.close()
                except Exception:
                    pass  # Ignore errors during cleanup close
            return False

    def _extract_market_details(self, market: dict):
        """Extracts and stores relevant market limits and precision."""
        limits = market.get("limits", {})
        precision = market.get("precision", {})

        self.min_qty = safe_decimal_conversion(limits.get("amount", {}).get("min"))
        amount_precision = precision.get(
            "amount"
        )  # Number of decimal places for amount
        # Qty step is usually 10^-precision
        self.qty_step = (
            (Decimal("1") / (Decimal("10") ** int(amount_precision)))
            if amount_precision is not None
            else None
        )

        price_precision = precision.get("price")  # Number of decimal places for price
        # Price tick is usually 10^-precision
        self.price_tick = (
            (Decimal("1") / (Decimal("10") ** int(price_precision)))
            if price_precision is not None
            else None
        )

        logger.info(
            f"Market Details Set: Min Qty={self.min_qty}, Qty Step={self.qty_step}, Price Tick={self.price_tick}"
        )

    def _update_state(self) -> bool:
        """Fetches and updates the current position, balance, and open orders."""
        logger.debug("Updating strategy state...")
        if not self.exchange:
            return False  # Ensure exchange is initialized
        try:
            # Fetch Position - Pass AppConfig
            pos_data = self.bybit_helpers.get_current_position_bybit_v5(
                self.exchange, self.symbol, self.app_config
            )
            if pos_data is None:
                logger.error("Failed to fetch position data.")
                return False

            self.current_side = pos_data["side"]
            self.current_qty = pos_data["qty"]
            self.entry_price = pos_data.get("entry_price")  # Can be None if no position

            # Fetch Balance - Pass AppConfig
            balance_info = self.bybit_helpers.fetch_usdt_balance(
                self.exchange, self.app_config
            )
            if balance_info is None:
                logger.error("Failed to fetch balance data.")
                return False
            _, available_balance = balance_info  # Unpack equity, available
            logger.info(
                f"Available Balance: {available_balance:.4f} {self.api_config.usdt_symbol}"
            )

            # If not in position, reset tracked orders
            if self.current_side == self.api_config.pos_none:
                if self.sl_order_id or self.tp_order_id:
                    logger.debug("Not in position, clearing tracked SL/TP order IDs.")
                    self.sl_order_id = None
                    self.tp_order_id = None
            else:  # If in position, verify tracked SL/TP orders still exist and are open
                self._verify_open_sl_tp()

            logger.debug("State update complete.")
            return True

        except Exception as e:
            logger.error(f"Unexpected error during state update: {e}", exc_info=True)
            return False

    def _verify_open_sl_tp(self):
        """Checks if tracked SL/TP orders are still open."""
        if not self.exchange:
            return
        # Determine order filter based on placement type
        sl_filter = (
            "StopOrder" if not self.strategy_config.place_tpsl_as_limit else "Order"
        )
        tp_filter = "Order"  # TP is always limit

        if self.sl_order_id:
            sl_order = self.bybit_helpers.fetch_order(
                self.exchange,
                self.symbol,
                self.sl_order_id,
                self.app_config,
                order_filter=sl_filter,
            )
            if not sl_order or sl_order.get("status") not in [
                "open",
                "new",
                "untriggered",
            ]:  # Check for open/pending states
                logger.warning(
                    f"Tracked SL order {self.sl_order_id} no longer active (Status: {sl_order.get('status') if sl_order else 'Not Found'}). Clearing ID."
                )
                self.sl_order_id = None
            else:
                logger.debug(
                    f"Tracked SL order {self.sl_order_id} status: {sl_order.get('status')}"
                )

        if self.tp_order_id:
            tp_order = self.bybit_helpers.fetch_order(
                self.exchange,
                self.symbol,
                self.tp_order_id,
                self.app_config,
                order_filter=tp_filter,
            )
            if not tp_order or tp_order.get("status") not in [
                "open",
                "new",
            ]:  # TP is limit, check open/new
                logger.warning(
                    f"Tracked TP order {self.tp_order_id} no longer active (Status: {tp_order.get('status') if tp_order else 'Not Found'}). Clearing ID."
                )
                self.tp_order_id = None
            else:
                logger.debug(
                    f"Tracked TP order {self.tp_order_id} status: {tp_order.get('status')}"
                )

    def _fetch_data(self) -> tuple[pd.DataFrame | None, Decimal | None]:
        """Fetches OHLCV data and the latest ticker price."""
        if not self.exchange:
            return None, None
        logger.debug("Fetching market data...")
        # Pass AppConfig
        ohlcv_df = self.bybit_helpers.fetch_ohlcv_paginated(
            exchange=self.exchange,
            symbol=self.symbol,
            timeframe=self.timeframe,
            config=self.app_config,  # Pass main config
            max_total_candles=self.strategy_config.ohlcv_limit
            + 5,  # Fetch slightly more for lookbacks
        )
        if ohlcv_df is None or not isinstance(ohlcv_df, pd.DataFrame) or ohlcv_df.empty:
            logger.warning("Could not fetch sufficient OHLCV data.")
            return None, None

        # Pass AppConfig
        ticker = self.bybit_helpers.fetch_ticker_validated(
            self.exchange, self.symbol, self.app_config
        )
        if ticker is None:
            logger.warning("Could not fetch valid ticker data.")
            return ohlcv_df, None  # Return OHLCV if available, but no price

        current_price = ticker.get("last")  # Already Decimal from helper
        if current_price is None:
            logger.warning("Ticker data retrieved but missing 'last' price.")
            return ohlcv_df, None

        logger.debug(
            f"Data fetched: {len(ohlcv_df)} candles, Last Price: {current_price}"
        )
        return ohlcv_df, current_price

    def _calculate_indicators(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame | None:
        """Calculates indicators based on the config."""
        if ohlcv_df is None or ohlcv_df.empty:
            return None
        logger.debug("Calculating indicators...")
        # Prepare config dict expected by indicators module
        indicator_config_dict = {
            "indicator_settings": self.strategy_config.indicator_settings.model_dump(),
            "analysis_flags": self.strategy_config.analysis_flags.model_dump(),
            "strategy_params": self.strategy_config.strategy_params,
            "strategy": self.strategy_config.strategy_info,
        }
        df_with_indicators = ind.calculate_all_indicators(
            ohlcv_df, indicator_config_dict
        )

        # Validate necessary columns exist
        evt_len = self.strategy_config.indicator_settings.evt_length
        atr_len = self.strategy_config.indicator_settings.atr_period
        evt_trend_col = f"evt_trend_{evt_len}"
        atr_col = f"ATRr_{atr_len}"  # pandas_ta default name

        if df_with_indicators is None:
            logger.error("Indicator calculation returned None.")
            return None
        if evt_trend_col not in df_with_indicators.columns:
            logger.error(
                f"Required EVT trend column '{evt_trend_col}' not found after calculation."
            )
            return None
        if (
            self.strategy_config.analysis_flags.use_atr
            and atr_col not in df_with_indicators.columns
        ):
            logger.error(
                f"Required ATR column '{atr_col}' not found after calculation (use_atr is True)."
            )
            return None

        logger.debug("Indicators calculated successfully.")
        return df_with_indicators

    def _generate_signals(self, df_ind: pd.DataFrame) -> Literal["buy", "sell"] | None:
        """Generates trading signals based on the last indicator data point."""
        if df_ind is None or df_ind.empty:
            return None
        logger.debug("Generating trading signals...")
        try:
            latest = df_ind.iloc[-1]
            evt_len = self.strategy_config.indicator_settings.evt_length
            trend_col = f"evt_trend_{evt_len}"
            buy_col = f"evt_buy_{evt_len}"
            sell_col = f"evt_sell_{evt_len}"

            if not all(
                col in latest.index and pd.notna(latest[col])
                for col in [trend_col, buy_col, sell_col]
            ):
                logger.warning(
                    f"EVT signal columns missing or NaN in latest data: {latest[[trend_col, buy_col, sell_col]].to_dict()}"
                )
                return None

            buy_signal = latest[buy_col]
            sell_signal = latest[sell_col]

            # Return 'buy'/'sell' string consistent with helper functions
            if buy_signal:
                logger.info(
                    f"{Fore.GREEN}BUY signal generated based on EVT Buy flag.{Style.RESET_ALL}"
                )
                return self.api_config.side_buy  # 'buy'
            elif sell_signal:
                logger.info(
                    f"{Fore.RED}SELL signal generated based on EVT Sell flag.{Style.RESET_ALL}"
                )
                return self.api_config.side_sell  # 'sell'
            else:
                # Check for exit signal based purely on trend reversal
                current_trend = int(latest[trend_col])
                if (
                    self.current_side == self.api_config.pos_long
                    and current_trend == -1
                ):
                    logger.info(
                        f"{Fore.YELLOW}EXIT LONG signal generated (Trend flipped Short).{Style.RESET_ALL}"
                    )
                    # Strategy logic handles exit separately, signal generator focuses on entry
                elif (
                    self.current_side == self.api_config.pos_short
                    and current_trend == 1
                ):
                    logger.info(
                        f"{Fore.YELLOW}EXIT SHORT signal generated (Trend flipped Long).{Style.RESET_ALL}"
                    )
                    # Strategy logic handles exit separately

                logger.debug("No new entry signal generated.")
                return None

        except IndexError:
            logger.warning(
                "IndexError generating signals (DataFrame likely too short)."
            )
            return None
        except Exception as e:
            logger.error(f"Error generating signals: {e}", exc_info=True)
            return None

    def _calculate_sl_tp(
        self, df_ind: pd.DataFrame, side: Literal["buy", "sell"], entry_price: Decimal
    ) -> tuple[Decimal | None, Decimal | None]:
        """Calculates initial stop-loss and take-profit prices."""
        if df_ind is None or df_ind.empty:
            return None, None
        logger.debug(f"Calculating SL/TP for {side} entry at {entry_price}...")
        if not self.exchange:
            return None, None
        try:
            atr_len = self.strategy_config.indicator_settings.atr_period
            atr_col = f"ATRr_{atr_len}"
            if atr_col not in df_ind.columns:
                logger.error(f"ATR column '{atr_col}' not found.")
                return None, None

            latest_atr = safe_decimal_conversion(df_ind.iloc[-1][atr_col])
            if latest_atr is None or latest_atr <= Decimal(0):
                logger.warning(
                    f"Invalid ATR value ({latest_atr}) for SL/TP calculation."
                )
                return None, None  # Require valid ATR

            # Stop Loss Calculation
            sl_multiplier = self.strategy_config.stop_loss_atr_multiplier
            sl_offset = latest_atr * sl_multiplier
            stop_loss_price = (
                (entry_price - sl_offset)
                if side == self.api_config.side_buy
                else (entry_price + sl_offset)
            )

            # --- Price Tick Adjustment ---
            # Adjust SL/TP to the nearest valid price tick
            if self.price_tick is None:
                logger.warning(
                    "Price tick size unknown. Cannot adjust SL/TP precisely."
                )
                sl_price_adjusted = stop_loss_price
            else:
                # Round SL "away" from entry (more conservative)
                rounding_mode = (
                    ROUND_DOWN if side == self.api_config.side_buy else ROUND_UP
                )
                sl_price_adjusted = (stop_loss_price / self.price_tick).quantize(
                    Decimal("1"), rounding=rounding_mode
                ) * self.price_tick

            # Ensure SL didn't cross entry after rounding
            if side == self.api_config.side_buy and sl_price_adjusted >= entry_price:
                sl_price_adjusted = (
                    entry_price - self.price_tick
                    if self.price_tick
                    else entry_price * Decimal("0.999")
                )
                logger.warning(
                    f"Adjusted Buy SL >= entry. Setting SL just below entry: {sl_price_adjusted}"
                )
            elif side == self.api_config.side_sell and sl_price_adjusted <= entry_price:
                sl_price_adjusted = (
                    entry_price + self.price_tick
                    if self.price_tick
                    else entry_price * Decimal("1.001")
                )
                logger.warning(
                    f"Adjusted Sell SL <= entry. Setting SL just above entry: {sl_price_adjusted}"
                )

            # Take Profit Calculation
            tp_multiplier = self.strategy_config.take_profit_atr_multiplier
            tp_price_adjusted = None
            if tp_multiplier > 0:
                tp_offset = latest_atr * tp_multiplier
                take_profit_price = (
                    (entry_price + tp_offset)
                    if side == self.api_config.side_buy
                    else (entry_price - tp_offset)
                )

                # Ensure TP is logical relative to entry BEFORE rounding
                if (
                    side == self.api_config.side_buy
                    and take_profit_price <= entry_price
                ):
                    logger.warning(
                        f"Calculated Buy TP ({take_profit_price}) <= entry ({entry_price}). Skipping TP."
                    )
                elif (
                    side == self.api_config.side_sell
                    and take_profit_price >= entry_price
                ):
                    logger.warning(
                        f"Calculated Sell TP ({take_profit_price}) >= entry ({entry_price}). Skipping TP."
                    )
                else:
                    # Adjust TP to nearest tick, round "towards" entry (more conservative fill chance)
                    if self.price_tick is None:
                        logger.warning(
                            "Price tick size unknown. Cannot adjust TP precisely."
                        )
                        tp_price_adjusted = take_profit_price
                    else:
                        rounding_mode = (
                            ROUND_DOWN if side == self.api_config.side_buy else ROUND_UP
                        )
                        tp_price_adjusted = (
                            take_profit_price / self.price_tick
                        ).quantize(
                            Decimal("1"), rounding=rounding_mode
                        ) * self.price_tick

                    # Ensure TP didn't cross entry after rounding
                    if (
                        side == self.api_config.side_buy
                        and tp_price_adjusted <= entry_price
                    ):
                        tp_price_adjusted = (
                            entry_price + self.price_tick
                            if self.price_tick
                            else entry_price * Decimal("1.001")
                        )
                        logger.warning(
                            f"Adjusted Buy TP <= entry. Setting TP just above entry: {tp_price_adjusted}"
                        )
                    elif (
                        side == self.api_config.side_sell
                        and tp_price_adjusted >= entry_price
                    ):
                        tp_price_adjusted = (
                            entry_price - self.price_tick
                            if self.price_tick
                            else entry_price * Decimal("0.999")
                        )
                        logger.warning(
                            f"Adjusted Sell TP >= entry. Setting TP just below entry: {tp_price_adjusted}"
                        )
            else:
                logger.info(
                    "Take Profit multiplier is zero or less. Skipping TP calculation."
                )

            logger.info(
                f"Calculated SL: {format_price(self.exchange, self.symbol, sl_price_adjusted)}, "
                f"TP: {format_price(self.exchange, self.symbol, tp_price_adjusted) or 'None'} (ATR: {latest_atr:.4f})"
            )
            return sl_price_adjusted, tp_price_adjusted

        except IndexError:
            logger.warning("IndexError calculating SL/TP.")
            return None, None
        except Exception as e:
            logger.error(f"Error calculating SL/TP: {e}", exc_info=True)
            return None, None

    def _calculate_position_size(
        self, entry_price: Decimal, stop_loss_price: Decimal
    ) -> Decimal | None:
        """Calculates position size based on risk percentage and stop-loss distance."""
        if not self.exchange:
            return None
        logger.debug("Calculating position size...")
        try:
            # Pass AppConfig
            balance_info = self.bybit_helpers.fetch_usdt_balance(
                self.exchange, self.app_config
            )
            if balance_info is None:
                logger.error("Cannot calc size: Failed fetch balance.")
                return None
            _, available_balance = balance_info

            if available_balance is None or available_balance <= Decimal("0"):
                logger.error(
                    "Cannot calculate position size: Zero or invalid available balance."
                )
                return None

            risk_amount_usd = available_balance * self.strategy_config.risk_per_trade
            price_diff = abs(entry_price - stop_loss_price)

            if price_diff <= Decimal("0"):
                logger.error(
                    f"Cannot calculate size: Entry price ({entry_price}) and SL price ({stop_loss_price}) invalid or equal."
                )
                return None

            # Calculate size based on risk amount and price difference per unit
            market = self.exchange.market(self.symbol)
            is_inverse = market.get("inverse", False)

            position_size_base: Decimal
            if is_inverse:
                # Risk = Size * abs(1/Entry - 1/SL) => Size = Risk / abs(1/Entry - 1/SL)
                if entry_price <= 0 or stop_loss_price <= 0:
                    raise ValueError("Prices must be positive for inverse size calc.")
                size_denominator = abs(
                    Decimal(1) / entry_price - Decimal(1) / stop_loss_price
                )
                if size_denominator <= 0:
                    raise ValueError("Inverse size denominator is zero.")
                position_size_base = risk_amount_usd / size_denominator
            else:  # Linear contract
                position_size_base = risk_amount_usd / price_diff

            # Apply market precision/step size constraints
            if self.qty_step is None:
                logger.warning(
                    "Quantity step size unknown, cannot adjust size precisely."
                )
                position_size_adjusted = position_size_base  # Use raw value
            else:
                # Round down to the nearest step size increment
                position_size_adjusted = (
                    position_size_base // self.qty_step
                ) * self.qty_step

            if position_size_adjusted <= Decimal(0):
                logger.warning(
                    f"Adjusted position size is zero or negative. Step: {self.qty_step}, Orig: {position_size_base}"
                )
                return None

            if self.min_qty is not None and position_size_adjusted < self.min_qty:
                logger.warning(
                    f"Calculated size ({position_size_adjusted}) < Min Qty ({self.min_qty}). Cannot trade this size."
                )
                return None  # Default: Don't trade if calculated size is too small

            logger.info(
                f"Calculated position size: {format_amount(self.exchange, self.symbol, position_size_adjusted)} "
                f"(Risk: {risk_amount_usd:.2f} USDT, Balance: {available_balance:.2f} USDT)"
            )
            return position_size_adjusted

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return None

    def _cancel_open_orders(self, reason: str = "Strategy Action") -> bool:
        """Cancels tracked SL and TP orders."""
        if not self.exchange:
            return False
        cancelled_sl, cancelled_tp = True, True  # Assume success if no ID tracked
        all_success = True

        # Determine order filter based on placement type
        sl_filter = (
            "StopOrder" if not self.strategy_config.place_tpsl_as_limit else "Order"
        )
        tp_filter = "Order"  # TP is always limit

        if self.sl_order_id:
            logger.info(
                f"Cancelling existing SL order {format_order_id(self.sl_order_id)} ({reason})..."
            )
            try:
                # Pass AppConfig and filter
                cancelled_sl = self.bybit_helpers.cancel_order(
                    self.exchange,
                    self.symbol,
                    self.sl_order_id,
                    self.app_config,
                    order_filter=sl_filter,
                )
                if cancelled_sl:
                    logger.info("SL order cancelled successfully or already gone.")
                else:
                    logger.warning("Failed attempt to cancel SL order.")
                    all_success = False
            except Exception as e:
                logger.error(
                    f"Error cancelling SL order {self.sl_order_id}: {e}", exc_info=True
                )
                cancelled_sl = False
                all_success = False
            finally:
                self.sl_order_id = None  # Always clear tracked ID after attempt

        if self.tp_order_id:
            logger.info(
                f"Cancelling existing TP order {format_order_id(self.tp_order_id)} ({reason})..."
            )
            try:
                # Pass AppConfig and filter
                cancelled_tp = self.bybit_helpers.cancel_order(
                    self.exchange,
                    self.symbol,
                    self.tp_order_id,
                    self.app_config,
                    order_filter=tp_filter,
                )
                if cancelled_tp:
                    logger.info("TP order cancelled successfully or already gone.")
                else:
                    logger.warning("Failed attempt to cancel TP order.")
                    all_success = False
            except Exception as e:
                logger.error(
                    f"Error cancelling TP order {self.tp_order_id}: {e}", exc_info=True
                )
                cancelled_tp = False
                all_success = False
            finally:
                self.tp_order_id = None  # Always clear tracked ID after attempt

        return all_success  # Return overall success

    def _cancel_all_open_orders(self, reason: str = "Strategy Action") -> bool:
        """Cancels ALL open orders for the symbol (more robust cleanup)."""
        if not self.exchange:
            return False
        logger.info(f"Cancelling ALL open orders for {self.symbol} ({reason})...")
        success = True
        # Cancel regular orders first
        try:
            cancelled_reg = self.bybit_helpers.cancel_all_orders(
                self.exchange,
                self.symbol,
                self.app_config,
                reason=f"{reason} (Regular)",
                order_filter="Order",
            )
            if not cancelled_reg:
                logger.warning("Failed to cancel all regular orders.")
                success = False
            else:
                logger.info("Regular orders cancelled successfully or none found.")
        except Exception as e:
            logger.error(f"Error cancelling regular orders: {e}", exc_info=True)
            success = False

        # Cancel stop/conditional orders
        try:
            cancelled_stop = self.bybit_helpers.cancel_all_orders(
                self.exchange,
                self.symbol,
                self.app_config,
                reason=f"{reason} (Stop)",
                order_filter="StopOrder",
            )
            if not cancelled_stop:
                logger.warning("Failed to cancel all stop orders.")
                success = False
            else:
                logger.info("Stop orders cancelled successfully or none found.")
        except Exception as e:
            logger.error(f"Error cancelling stop orders: {e}", exc_info=True)
            success = False

        # Clear tracked IDs regardless of cancellation outcome
        self.sl_order_id = None
        self.tp_order_id = None
        return success

    def _close_position(self, reason: str) -> bool:
        """Closes the current position using the helper function."""
        if not self.exchange:
            return False
        logger.info(f"Attempting to close position for {self.symbol}. Reason: {reason}")
        # Use the helper, pass AppConfig
        close_order_result = self.bybit_helpers.close_position_reduce_only(
            exchange=self.exchange,
            symbol=self.symbol,
            config=self.app_config,  # Pass main AppConfig
            reason=reason,
        )
        if close_order_result:
            logger.success(
                f"Position close order submitted successfully. Reason: {reason}"
            )
            # Optionally wait briefly and confirm position is closed via _update_state
            time.sleep(2)  # Small delay for order to potentially fill/update
            self._update_state()
            if self.current_side == self.api_config.pos_none:
                logger.success("Position confirmed closed after order submission.")
                return True
            else:
                logger.warning(
                    f"Position still showing active after close order submission (Side: {self.current_side}, Qty: {self.current_qty}). Manual check may be needed."
                )
                return False  # Indicate position might still be open
        else:
            # Helper function logs errors, just indicate failure here
            logger.error(f"Failed to submit position close order. Reason: {reason}")
            return False

    def _handle_exit(self, df_ind: pd.DataFrame) -> bool:
        """Checks exit conditions and closes the position if necessary."""
        if not self.exchange or self.current_side == self.api_config.pos_none:
            return False  # Not initialized or not in position

        logger.debug("Checking exit conditions...")
        should_exit = False
        exit_reason = ""
        try:
            evt_len = self.strategy_config.indicator_settings.evt_length
            latest_trend_col = f"evt_trend_{evt_len}"
            if latest_trend_col not in df_ind.columns:
                logger.warning(
                    f"Cannot check exit: EVT Trend column '{latest_trend_col}' missing."
                )
                return False

            latest_trend_val = df_ind.iloc[-1].get(latest_trend_col)

            if latest_trend_val is not None and pd.notna(latest_trend_val):
                latest_trend = int(latest_trend_val)  # Ensure integer comparison
                if self.current_side == self.api_config.pos_long and latest_trend == -1:
                    should_exit = True
                    exit_reason = "EVT Trend flipped Short"
                elif (
                    self.current_side == self.api_config.pos_short and latest_trend == 1
                ):
                    should_exit = True
                    exit_reason = "EVT Trend flipped Long"
            else:
                logger.warning(
                    "Cannot determine latest EVT trend for exit check (NaN)."
                )

            # --- Add check for SL/TP Hit ---
            # We rely on _verify_open_sl_tp clearing the IDs if orders are filled/cancelled.
            # If either ID is None while in a position, it suggests the order might have been hit.
            # This is an indirect check.
            if not should_exit:
                if (
                    self.sl_order_id is None
                    and self.current_side != self.api_config.pos_none
                ):
                    should_exit = True
                    exit_reason = "Stop Loss Order likely hit (ID missing)"
                    logger.warning(
                        f"{Fore.YELLOW}Assuming SL hit for {self.current_side} position as SL order ID is missing.{Style.RESET_ALL}"
                    )
                elif (
                    self.tp_order_id is None
                    and self.strategy_config.take_profit_atr_multiplier > 0
                    and self.current_side != self.api_config.pos_none
                ):
                    # Only consider TP hit if TP was actually configured
                    should_exit = True
                    exit_reason = "Take Profit Order likely hit (ID missing)"
                    logger.warning(
                        f"{Fore.YELLOW}Assuming TP hit for {self.current_side} position as TP order ID is missing.{Style.RESET_ALL}"
                    )

            # --- Execute Exit ---
            if should_exit:
                logger.warning(
                    f"{Fore.YELLOW}EXIT Triggered for {self.current_side} {self.current_qty} {self.symbol}. Reason: {exit_reason}{Style.RESET_ALL}"
                )
                # 1. Cancel remaining SL/TP orders first (important!)
                self._cancel_open_orders(f"Pre-Exit Cleanup ({exit_reason})")

                # 2. Close the position
                close_success = self._close_position(exit_reason)
                if close_success:
                    logger.success(
                        f"Exit process initiated successfully for {self.symbol}."
                    )
                    # State will be updated in the next loop iteration
                    return True  # Indicate exit was actioned
                else:
                    logger.error(
                        f"Exit process failed for {self.symbol}. Manual intervention might be needed."
                    )
                    send_sms_alert(
                        f"CRITICAL: Failed to close {self.symbol} position ({exit_reason}). Manual check needed!",
                        self.sms_config,
                    )
                    return False  # Indicate exit failed
            else:
                logger.debug("No exit condition met.")
                return False  # No exit action taken

        except Exception as e:
            logger.error(f"Error during exit check: {e}", exc_info=True)
            return False

    def _place_sl_tp(
        self,
        position_qty: Decimal,
        position_side: Literal["buy", "sell"],
        stop_loss_price: Decimal,
        take_profit_price: Decimal | None,
    ) -> bool:
        """Places the stop-loss and take-profit orders after entry."""
        if not self.exchange:
            return False
        logger.info(
            f"Placing SL/TP for {position_side} {position_qty} {self.symbol}..."
        )
        sl_success, tp_success = False, True  # TP success is True if not placing
        sl_order_details, tp_order_details = None, None
        order_side = (
            self.api_config.side_sell
            if position_side == self.api_config.pos_long
            else self.api_config.side_buy
        )

        # Place Stop Loss
        if stop_loss_price is not None:
            try:
                sl_client_oid = f"sl_{self.symbol.split('/')[0]}_{int(time.time())}"[
                    -36:
                ]
                if self.strategy_config.place_tpsl_as_limit:
                    logger.info(
                        f"Placing SL as Reduce-Only LIMIT Order: Side={order_side}, Qty={position_qty}, Px={stop_loss_price}"
                    )
                    sl_order_details = self.bybit_helpers.place_limit_order_tif(
                        exchange=self.exchange,
                        symbol=self.symbol,
                        side=order_side,
                        amount=position_qty,
                        price=stop_loss_price,
                        config=self.app_config,
                        is_reduce_only=True,
                        client_order_id=sl_client_oid,
                        time_in_force="GTC",  # GTC is typical for SL/TP limit
                    )
                else:  # Native Stop Order
                    logger.info(
                        f"Placing SL as NATIVE Stop Market Order: Side={order_side}, Qty={position_qty}, TriggerPx={stop_loss_price}"
                    )
                    sl_order_details = self.bybit_helpers.place_native_stop_loss(
                        exchange=self.exchange,
                        symbol=self.symbol,
                        side=order_side,
                        amount=position_qty,
                        stop_price=stop_loss_price,
                        config=self.app_config,
                        client_order_id=sl_client_oid,
                    )

                if sl_order_details and sl_order_details.get("id"):
                    self.sl_order_id = sl_order_details["id"]
                    logger.success(
                        f"Stop Loss order placed successfully. ID: {format_order_id(self.sl_order_id)}"
                    )
                    sl_success = True
                else:
                    logger.error("Failed to place Stop Loss order.")
            except Exception as e:
                logger.error(f"Exception placing Stop Loss order: {e}", exc_info=True)
        else:
            logger.error("Stop Loss price is None, cannot place SL order.")

        # Place Take Profit (always as reduce-only limit order)
        if take_profit_price is not None:
            try:
                tp_client_oid = f"tp_{self.symbol.split('/')[0]}_{int(time.time())}"[
                    -36:
                ]
                logger.info(
                    f"Placing TP as Reduce-Only LIMIT Order: Side={order_side}, Qty={position_qty}, Px={take_profit_price}"
                )
                tp_order_details = self.bybit_helpers.place_limit_order_tif(
                    exchange=self.exchange,
                    symbol=self.symbol,
                    side=order_side,
                    amount=position_qty,
                    price=take_profit_price,
                    config=self.app_config,
                    is_reduce_only=True,
                    client_order_id=tp_client_oid,
                    time_in_force="GTC",
                )
                if tp_order_details and tp_order_details.get("id"):
                    self.tp_order_id = tp_order_details["id"]
                    logger.success(
                        f"Take Profit order placed successfully. ID: {format_order_id(self.tp_order_id)}"
                    )
                    tp_success = True
                else:
                    logger.error("Failed to place Take Profit order.")
                    tp_success = False  # Mark failure if TP was intended but failed
            except Exception as e:
                logger.error(f"Exception placing Take Profit order: {e}", exc_info=True)
                tp_success = False
        else:
            logger.info("Take Profit price is None, skipping TP order placement.")
            tp_success = True  # Not placing TP is not a failure in this context

        if not sl_success:
            logger.critical(
                f"{Back.RED}FAILED TO PLACE STOP LOSS ORDER! POSITION IS OPEN WITHOUT SL.{Style.RESET_ALL}"
            )
            send_sms_alert(
                f"CRITICAL: Failed place SL for {self.symbol}. Pos OPEN!",
                self.sms_config,
            )
            # Decide on action: close position immediately? Or just warn?
            # self._close_position("Failed SL Placement") # Potentially close immediately
            return False  # Indicate critical failure

        if not tp_success:
            logger.warning(
                f"{Fore.YELLOW}Failed to place Take Profit order. Position remains open with SL only.{Style.RESET_ALL}"
            )
            # Continue running, but TP is missing

        return sl_success  # Overall success depends mainly on SL placement

    def _handle_entry(
        self,
        entry_signal: Literal["buy", "sell"],
        current_price: Decimal,
        df_ind: pd.DataFrame,
    ):
        """Handles the logic for entering a new position."""
        if not self.exchange:
            return
        if self.current_side != self.api_config.pos_none:
            logger.info(
                f"Entry signal '{entry_signal}' received, but already in {self.current_side} position. Ignoring."
            )
            return

        logger.warning(
            f"{Style.BRIGHT}--- Attempting {entry_signal.upper()} Entry for {self.symbol} ---{Style.RESET_ALL}"
        )

        # 1. Calculate SL/TP
        # Use current_price as the estimated entry price for initial calculation
        sl_price, tp_price = self._calculate_sl_tp(df_ind, entry_signal, current_price)
        if sl_price is None:
            logger.error("Entry aborted: Could not calculate valid Stop Loss price.")
            return

        # 2. Calculate Position Size
        position_size = self._calculate_position_size(current_price, sl_price)
        if position_size is None or position_size <= Decimal(0):
            logger.error("Entry aborted: Could not calculate valid position size.")
            return

        # 3. Cancel any potentially stray orders (shouldn't be needed if state is clean)
        self._cancel_all_open_orders("Pre-Entry Cleanup")

        # 4. Place Market Entry Order
        entry_client_oid = f"entry_{self.symbol.split('/')[0]}_{int(time.time())}"[-36:]
        entry_order = self.bybit_helpers.place_market_order_slippage_check(
            exchange=self.exchange,
            symbol=self.symbol,
            side=entry_signal,
            amount=position_size,
            config=self.app_config,
            client_order_id=entry_client_oid,
        )

        # 5. Handle Entry Order Result
        if not entry_order or not entry_order.get("id"):
            logger.error("Entry aborted: Market entry order failed.")
            return

        entry_order_id = entry_order["id"]
        logger.success(
            f"Entry order submitted OK. ID: {format_order_id(entry_order_id)}"
        )

        # 6. Fetch Actual Entry Details (CRITICAL for SL/TP accuracy)
        # Wait a short time for the order to likely fill and update state
        time.sleep(3)  # Adjust delay as needed
        if not self._update_state():  # Fetch position again
            logger.error(
                "Failed to update state after entry order. Cannot place SL/TP accurately."
            )
            send_sms_alert(
                f"CRITICAL: Failed state update after {self.symbol} entry. SL/TP NOT PLACED!",
                self.sms_config,
            )
            # Position is open without SL/TP - very risky!
            return

        # Check if position was actually opened successfully
        if self.current_side == self.api_config.pos_none or self.current_qty <= Decimal(
            0
        ):
            logger.error(
                f"Entry FAILED: Position not detected after market order {format_order_id(entry_order_id)} submission."
            )
            # Maybe the order was rejected or filled zero?
            return

        # Verify the opened position matches the intended signal and size (approximately)
        if self.current_side != (
            self.api_config.pos_long
            if entry_signal == "buy"
            else self.api_config.pos_short
        ):
            logger.error(
                f"Entry MISMATCH: Intended {entry_signal}, but position is {self.current_side}. Aborting SL/TP."
            )
            # Close the unexpected position?
            self._close_position("Unexpected position side after entry")
            return

        # Use actual filled quantity and entry price from the updated state
        actual_entry_price = self.entry_price
        actual_position_qty = self.current_qty
        logger.info(
            f"Confirmed Entry: Side={self.current_side}, Qty={actual_position_qty}, AvgPrice={actual_entry_price}"
        )

        # 7. Recalculate SL/TP based on ACTUAL entry price
        sl_price_final, tp_price_final = self._calculate_sl_tp(
            df_ind, entry_signal, actual_entry_price
        )
        if sl_price_final is None:
            logger.error(
                "Entry Failed: Could not calculate FINAL Stop Loss price after entry confirmation."
            )
            send_sms_alert(
                f"CRITICAL: Failed FINAL SL calc for {self.symbol}. Pos OPEN! Closing...",
                self.sms_config,
            )
            self._close_position("Failed Final SL Calculation")
            return

        # 8. Place SL/TP Orders
        if not self._place_sl_tp(
            actual_position_qty, entry_signal, sl_price_final, tp_price_final
        ):
            # Error placing SL/TP logged by _place_sl_tp, potentially position closed if SL failed
            logger.error("Entry process completed with issues (SL/TP placement).")
        else:
            logger.success(
                f"--- {entry_signal.upper()} Entry Process Complete for {self.symbol} ---"
            )

    def run_loop(self):
        """Main strategy execution loop."""
        if not self._initialize():
            logger.critical("Strategy initialization failed. Exiting.")
            return

        self.is_running = True
        logger.info(
            f"{Fore.GREEN}{Style.BRIGHT}>>> Strategy Loop Started ({self.symbol} / {self.timeframe}) <<< {Style.RESET_ALL}"
        )

        while self.is_running:
            loop_start_time = time.time()
            try:
                logger.info(f"{Style.DIM}--- Loop Iteration ---{Style.RESET_ALL}")

                # 1. Update State
                if not self._update_state():
                    logger.warning("Failed to update state in loop. Skipping cycle.")
                    time.sleep(
                        self.strategy_config.loop_delay_seconds // 2
                    )  # Shorter sleep on error
                    continue

                # 2. Fetch Data
                ohlcv_df, current_price = self._fetch_data()
                if ohlcv_df is None or current_price is None:
                    logger.warning("Failed to fetch data in loop. Skipping cycle.")
                    time.sleep(self.strategy_config.loop_delay_seconds // 2)
                    continue

                # 3. Calculate Indicators
                df_with_indicators = self._calculate_indicators(ohlcv_df)
                if df_with_indicators is None:
                    logger.warning("Failed indicator calculation. Skipping cycle.")
                    time.sleep(self.strategy_config.loop_delay_seconds // 2)
                    continue

                # 4. Check for Exit (if in position)
                exit_actioned = False
                if self.current_side != self.api_config.pos_none:
                    exit_actioned = self._handle_exit(df_with_indicators)
                    if exit_actioned:
                        logger.info(
                            "Exit actioned this cycle. State will update next loop."
                        )
                        # Skip entry logic if we just exited
                        # Continue to sleep calculation

                # 5. Generate & Handle Entry Signals (if not in position and didn't just exit)
                if self.current_side == self.api_config.pos_none and not exit_actioned:
                    entry_signal = self._generate_signals(df_with_indicators)
                    if entry_signal:
                        self._handle_entry(
                            entry_signal, current_price, df_with_indicators
                        )
                        # Entry handling places orders, state updates next loop

                # 6. Calculate Sleep Time
                loop_end_time = time.time()
                execution_time = loop_end_time - loop_start_time
                sleep_time = max(
                    0, self.strategy_config.loop_delay_seconds - execution_time
                )
                logger.info(
                    f"Loop finished in {execution_time:.2f}s. Sleeping for {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received. Stopping strategy...")
                self.is_running = False  # Signal loop to stop
            except ccxt.NetworkError as e:
                logger.error(f"NetworkError in main loop: {e}. Retrying after delay...")
                time.sleep(
                    self.strategy_config.loop_delay_seconds
                )  # Longer sleep on network error
            except Exception as e:
                logger.critical(
                    f"{Back.RED}CRITICAL UNHANDLED EXCEPTION in main loop: {e}{Style.RESET_ALL}",
                    exc_info=True,
                )
                send_sms_alert(
                    f"CRITICAL ERROR in {self.symbol} strategy loop: {type(e).__name__}",
                    self.sms_config,
                )
                # Consider stopping the bot on critical errors
                # self.is_running = False
                time.sleep(
                    self.strategy_config.loop_delay_seconds * 2
                )  # Longer sleep on critical error

        logger.warning("--- Strategy Loop Stopped ---")
        self.stop()  # Ensure cleanup is called

    def start(self):
        """Starts the strategy loop."""
        self.run_loop()

    def stop(self):
        """Stops the strategy, cancels orders, and cleans up."""
        logger.warning("--- Initiating Strategy Shutdown ---")
        self.is_running = False  # Ensure loop flag is False

        if self.exchange:
            # Cancel all orders first
            logger.info("Cancelling any remaining open orders...")
            # Use the more robust cancel_all function here
            self._cancel_all_open_orders("Strategy Shutdown")

            # Optional: Close open position on stop
            # if self.current_side != self.api_config.pos_none:
            #     logger.warning("Closing open position due to strategy stop...")
            #     self._close_position("Strategy Shutdown")

            # Close exchange connection
            try:
                logger.info("Closing exchange connection...")
                self.exchange.close()
                logger.info("Exchange connection closed.")
            except Exception as e:
                logger.error(f"Error closing exchange connection: {e}", exc_info=True)
        else:
            logger.info("Exchange not initialized, nothing to close.")

        logger.warning("--- Strategy Shutdown Complete ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Load Configuration ---
    try:
        # Load from .env file and environment variables
        app_config = load_config()
    except SystemExit:
        # load_config prints detailed errors and exits
        sys.exit(1)  # Exit if config loading failed
    except Exception as e:
        # Catch any other unexpected error during config load
        print(
            f"\n{Back.RED}FATAL: Unexpected error during configuration loading: {e}{Style.RESET_ALL}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Setup Logger ---
    # Use configuration from AppConfig
    log_conf = app_config.logging_config
    logger = setup_logger(
        logger_name=log_conf.logger_name,
        log_file=log_conf.log_file,
        console_level_str=log_conf.console_level_str,
        file_level_str=log_conf.file_level_str,
        log_rotation_bytes=log_conf.log_rotation_bytes,
        log_backup_count=log_conf.log_backup_count,
        third_party_log_level_str=log_conf.third_party_log_level_str,
    )
    logger.info("=" * 60)
    logger.info(f"Logger '{log_conf.logger_name}' initialized.")
    logger.info(
        f"Using Project Config: Testnet={app_config.api_config.testnet_mode}, Symbol={app_config.api_config.symbol}"
    )
    logger.info("=" * 60)

    # --- Initialize and Run Strategy ---
    strategy = EhlersStrategy(app_config)
    try:
        strategy.start()  # This enters the main run_loop
    except Exception as e:
        logger.critical(
            f"{Back.RED}Unhandled exception at strategy top level: {e}{Style.RESET_ALL}",
            exc_info=True,
        )
    finally:
        # Ensure stop sequence is called even if start() fails or loop exits unexpectedly
        logger.info("Ensuring strategy shutdown sequence...")
        strategy.stop()
        logger.info("Main script finished.")
