# -*- coding: utf-8 -*-
"""
Strategy Definition: Enhanced Volumatic Trend + OB (for CCXT Async Bot)

This strategy combines a trend-following mechanism based on smoothed EMAs
with order block identification using pivot points. It generates BUY/SELL
signals when price enters a relevant order block while the trend aligns.
"""
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import pandas_ta as ta  # Ensure pandas_ta is installed
from colorama import Fore, Style

# --- Constants ---
ATR_MULTIPLIER_MAIN = 3.0  # Multiplier for main trend bands based on ATR
ATR_MULTIPLIER_VOL = 4.0   # Multiplier for volatility bands (offset from opposite main band)
PIVOT_CHECK_LOOKBACK_BUFFER = 5  # Extra bars to look back for potential new pivots confirmation
DEFAULT_PRICE_PRECISION = 8
DEFAULT_AMOUNT_PRECISION = 8
DEFAULT_PRICE_TICK_STR = '0.00000001'
DEFAULT_AMOUNT_TICK_STR = '0.00000001'
# Minimum OB range relative to price tick size to be considered valid (prevents tiny OBs due to precision noise)
MIN_OB_RANGE_FACTOR = Decimal('0.1')
# Small epsilon for float comparisons, especially involving ATR or zero checks
FLOAT_EPSILON = 1e-12

log = logging.getLogger(__name__)  # Gets logger configured in main.py
# Set decimal precision for calculations involving Decimal.
# Note: This sets precision globally for this module's context.
# Consider local context management if this module interacts heavily with others using Decimal.
getcontext().prec = 28  # Sufficient precision for typical crypto price calculations

# --- Type Definitions (consistent with main.py if applicable) ---
class OrderBlock(TypedDict):
    """Represents a detected Order Block."""
    id: int          # Integer index of the candle where the OB's pivot formed (relative to df)
    type: str        # 'bull' or 'bear'
    left_idx: int    # Integer index of bar where OB pivot formed
    right_idx: int   # Integer index of last bar OB is valid for (updated if active)
    top: float       # Top price boundary of the OB
    bottom: float    # Bottom price boundary of the OB
    active: bool     # Still considered valid (not invalidated by price action)?
    closed_idx: Optional[int]  # Integer index where it was invalidated, None if active


class AnalysisResults(TypedDict):
    """Structure for returning strategy analysis results."""
    dataframe: pd.DataFrame        # DataFrame with calculated indicators
    last_signal: str               # 'BUY', 'SELL', 'HOLD', 'EXIT_LONG', 'EXIT_SHORT'
    active_bull_boxes: List[OrderBlock] # List of currently active bullish OBs
    active_bear_boxes: List[OrderBlock] # List of currently active bearish OBs
    last_close: float              # Last closing price used for analysis
    current_trend: Optional[bool]  # True for UP, False for DOWN, None if undetermined
    trend_changed: bool            # Did the trend change on the last candle?
    last_atr: Optional[float]      # Last calculated ATR value


# --- Strategy Class ---
class VolumaticOBStrategy:
    """
    Encapsulates the logic for the 'Enhanced Volumatic Trend + OB' strategy.

    Calculates technical indicators (smoothed EMAs, ATR, pivots, volume normalization)
    and generates trading signals based on OHLCV data. Uses CCXT market structure
    for handling price and amount precision correctly.
    """
    def __init__(self, market: Dict[str, Any], **params: Any):
        """
        Initializes the strategy instance.

        Args:
            market: CCXT market structure dictionary containing precision info.
            **params: Strategy parameters loaded from configuration. Expected keys:
                'length', 'vol_atr_period', 'vol_percentile_len', 'vol_percentile',
                'ob_source', 'pivot_left_h', 'pivot_right_h', 'pivot_left_l',
                'pivot_right_l', 'max_boxes'.
        """
        if not market:
            log.error("Market data is required for initialization.")
            raise ValueError("Market data cannot be None or empty.")

        self.market = market
        self._parse_params(params)
        self._validate_params()
        self._set_precision(market)

        # --- Strategy State Variables ---
        # These are updated during the `update` method call.
        self.upper: Optional[float] = None          # Upper main trend band
        self.lower: Optional[float] = None          # Lower main trend band
        self.lower_vol: Optional[float] = None      # Lower volatility band
        self.upper_vol: Optional[float] = None      # Upper volatility band
        self.bull_boxes: List[OrderBlock] = []      # All identified bullish OBs (active and inactive)
        self.bear_boxes: List[OrderBlock] = []      # All identified bearish OBs (active and inactive)
        # Tracks the *intended* state based on signals ('BUY' means aiming for long, 'SELL' for short)
        # Helps prevent duplicate entry signals if the entry condition persists over multiple candles.
        self.last_signal_state: str = "HOLD"        # Internal state: HOLD, BUY (want long), SELL (want short)
        self.current_trend: Optional[bool] = None   # True = UP, False = DOWN, None = Undetermined

        # Calculate minimum required data length based on the largest lookback period needed by indicators.
        self.min_data_len = max(
            self.length + 3,  # For _ema_swma: SWMA(4) needs 4 bars, EMA(length) needs 'length' SWMA values. Min len = length + 3.
            self.vol_atr_period,
            self.vol_percentile_len,
            self.pivot_left_h + self.pivot_right_h + 1, # Lookback + current + lookforward for high pivots
            self.pivot_left_l + self.pivot_right_l + 1, # Lookback + current + lookforward for low pivots
            5 # Minimum reasonable length
        )

        log.info(f"{Fore.MAGENTA}Initializing VolumaticOB Strategy Engine...{Style.RESET_ALL}")
        log.info(f"Symbol: {market.get('symbol', 'N/A')}")
        log.info(f"Params: TrendLen={self.length}, Pivots={self.pivot_left_h}/{self.pivot_right_h}(H) "
                 f"{self.pivot_left_l}/{self.pivot_right_l}(L), MaxBoxes={self.max_boxes}, OB Source={self.ob_source}")
        log.info(f"Minimum data points required: {self.min_data_len}")
        log.debug(f"Price Precision: {self.price_precision}, Amount Precision: {self.amount_precision}")
        log.debug(f"Price Tick: {self.price_tick}, Amount Tick: {self.amount_tick}")

    def _parse_params(self, params: Dict[str, Any]):
        """Loads and type-casts parameters from the config dictionary."""
        try:
            self.length = int(params.get('length', 40))
            self.vol_atr_period = int(params.get('vol_atr_period', 200))
            self.vol_percentile_len = int(params.get('vol_percentile_len', 1000))
            self.vol_percentile = int(params.get('vol_percentile', 95))
            self.ob_source = str(params.get('ob_source', "Wicks")).capitalize() # Standardize capitalization
            self.pivot_left_h = int(params.get('pivot_left_h', 10))
            self.pivot_right_h = int(params.get('pivot_right_h', 10))
            self.pivot_left_l = int(params.get('pivot_left_l', 10))
            self.pivot_right_l = int(params.get('pivot_right_l', 10))
            self.max_boxes = int(params.get('max_boxes', 5))
        except (ValueError, TypeError) as e:
            log.error(f"Invalid strategy parameter type in configuration: {e}")
            raise ValueError(f"Invalid strategy parameter type in configuration: {e}") from e

    def _validate_params(self):
        """Performs basic validation of strategy parameters."""
        if self.ob_source not in ["Wicks", "Bodys"]:
            raise ValueError("Invalid 'ob_source' parameter. Must be 'Wicks' or 'Bodys'.")
        lengths = [self.length, self.vol_atr_period, self.vol_percentile_len,
                   self.pivot_left_h, self.pivot_right_h, self.pivot_left_l, self.pivot_right_l]
        if not all(isinstance(x, int) and x > 0 for x in lengths):
             raise ValueError("All length/period parameters must be positive integers.")
        if not isinstance(self.max_boxes, int) or self.max_boxes <= 0:
             raise ValueError("'max_boxes' must be a positive integer.")
        if not 0 < self.vol_percentile <= 100:
             raise ValueError("'vol_percentile' must be between 1 and 100 (inclusive of 100).")

    def _set_precision(self, market: Dict[str, Any]):
        """Extracts and sets price/amount precision details from CCXT market structure."""
        try:
            precision = market.get('precision', {})
            limits = market.get('limits', {})
            amount_limits = limits.get('amount', {})
            price_limits = limits.get('price', {})

            # Determine precision digits (number of decimal places)
            self.price_precision = int(precision.get('price', DEFAULT_PRICE_PRECISION))
            self.amount_precision = int(precision.get('amount', DEFAULT_AMOUNT_PRECISION))

            # Determine tick size (smallest increment) using Decimal for accuracy
            # Prefer market['precision']['tickSize'] if available, fallback to limits['price/amount']['min'] or defaults
            # Note: CCXT v4+ might use 'tickSize' in precision dict. Older versions rely more on limits['min'].
            # We check multiple potential sources for robustness.
            price_tick_str = (precision.get('tickSize') # Check newer precision dict first
                             or precision.get('price') # Sometimes precision dict has the tick directly
                             or price_limits.get('min') # Fallback to limits
                             or DEFAULT_PRICE_TICK_STR)
            amount_tick_str = (precision.get('amount') # Amount tick often in precision dict
                              or amount_limits.get('min') # Fallback to limits
                              or DEFAULT_AMOUNT_TICK_STR)

            self.price_tick = Decimal(str(price_tick_str))
            self.amount_tick = Decimal(str(amount_tick_str))

            if self.price_tick <= 0 or self.amount_tick <= 0:
                log.error("Market precision ticks must be positive. Check market data.")
                raise ValueError("Market precision ticks must be positive.")

        except (ValueError, TypeError, KeyError, InvalidOperation) as e:
            log.warning(f"Could not parse market precision/limits: {e}. Using defaults.")
            self.price_precision = DEFAULT_PRICE_PRECISION
            self.amount_precision = DEFAULT_AMOUNT_PRECISION
            self.price_tick = Decimal(DEFAULT_PRICE_TICK_STR)
            self.amount_tick = Decimal(DEFAULT_AMOUNT_TICK_STR)

    # --- Precision Handling ---
    def format_price(self, price: float) -> str:
        """Formats price according to market precision for display/logging."""
        if pd.isna(price): return "NaN"
        try:
            # Use quantize with the number of decimal places for standard formatting.
            price_decimal = Decimal(str(price))
            # Create a quantizer like Decimal('0.00000001') based on price_precision
            quantizer = Decimal('1e-' + str(self.price_precision))
            rounded_price = price_decimal.quantize(quantizer, rounding=ROUND_HALF_UP)
            # Ensure the output string matches the precision exactly (e.g., trailing zeros)
            return f"{rounded_price:.{self.price_precision}f}"
        except (InvalidOperation, TypeError, ValueError):
            log.warning(f"Could not format price: {price}")
            return str(price) # Fallback

    def format_amount(self, amount: float) -> str:
        """Formats amount according to market precision for display/logging."""
        if pd.isna(amount): return "NaN"
        try:
            amount_decimal = Decimal(str(amount))
            # Create a quantizer like Decimal('0.0001') based on amount_precision
            quantizer = Decimal('1e-' + str(self.amount_precision))
            # Usually round down amount for orders to avoid exceeding limits/funds
            rounded_amount = amount_decimal.quantize(quantizer, rounding=ROUND_DOWN)
            return f"{rounded_amount:.{self.amount_precision}f}"
        except (InvalidOperation, TypeError, ValueError):
            log.warning(f"Could not format amount: {amount}")
            return str(amount) # Fallback

    def round_price(self, price: float) -> float:
        """
        Rounds price to the nearest valid price tick using ROUND_HALF_UP (standard rounding).

        Note: Specific order placement (SL/TP) might require context-specific
        rounding (e.g., ROUND_UP for sell SL, ROUND_DOWN for buy SL) which should
        be handled in the order execution logic if needed, not here.
        """
        if pd.isna(price) or self.price_tick <= 0: return np.nan
        try:
            price_decimal = Decimal(str(price))
            # Quantize based on the tick size using standard rounding
            rounded = (price_decimal / self.price_tick).quantize(Decimal('0'), rounding=ROUND_HALF_UP) * self.price_tick
            return float(rounded)
        except (InvalidOperation, TypeError, ValueError):
            log.warning(f"Could not round price: {price}")
            return price # Fallback

    def round_amount(self, amount: float) -> float:
        """
        Rounds amount DOWN to the nearest valid amount tick (conservative approach).
        This helps avoid insufficient funds or exceeding order size limits.
        """
        if pd.isna(amount) or self.amount_tick <= 0: return np.nan
        try:
            amount_decimal = Decimal(str(amount))
            # Quantize based on the tick size, rounding down
            rounded = (amount_decimal / self.amount_tick).quantize(Decimal('0'), rounding=ROUND_DOWN) * self.amount_tick
            return float(rounded)
        except (InvalidOperation, TypeError, ValueError):
            log.warning(f"Could not round amount: {amount}")
            return amount # Fallback
    # --- End Precision Handling ---

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculates the Exponential Moving Average (EMA) of a 4-period
        Smoothed Moving Average (SWMA) of the input series.
        Provides a smoother trend line compared to a simple EMA.

        Args:
            series: The pandas Series (e.g., 'close' prices) to calculate on.
            length: The length parameter for the final EMA.

        Returns:
            A pandas Series containing the EMA(SWMA(4)) values.
        """
        # Need at least length + 3 bars for the calculation to be valid
        # SWMA(4) needs 4 bars, EMA(length) needs 'length' values of SWMA output.
        if series.empty or len(series) < length + 3:
            return pd.Series(np.nan, index=series.index, dtype=float) # Ensure float dtype

        try:
            # Calculate 4-period Smoothed Moving Average (SWMA)
            # SWMA is similar to Hull Moving Average's weighted moving average component.
            # Let pandas_ta handle NaN filling by default (usually NaN propagation)
            swma4 = ta.swma(series.astype(float), length=4)

            # Calculate EMA of the SWMA result
            # Let pandas_ta handle NaN filling by default
            ema_of_swma = ta.ema(swma4, length=length)
            return ema_of_swma # NaNs will propagate correctly
        except Exception as e:
            log.error(f"Error calculating EMA(SWMA): {e}", exc_info=True)
            return pd.Series(np.nan, index=series.index, dtype=float) # Ensure float dtype

    def _find_pivots(self, df: pd.DataFrame, left: int, right: int, is_high: bool) -> pd.Series:
        """
        Vectorized calculation of pivot points based on the Pine Script definition:
        - Pivot High: A value higher than `left` bars to the left and higher than or equal to `right` bars to the right.
        - Pivot Low: A value lower than `left` bars to the left and lower than or equal to `right` bars to the right.

        Args:
            df: DataFrame containing OHLC data.
            left: Number of bars to the left for comparison (strict inequality).
            right: Number of bars to the right for comparison (non-strict inequality).
            is_high: True to find pivot highs, False for pivot lows.

        Returns:
            A pandas Series with pivot prices at the pivot bar index, NaN otherwise.
        """
        if self.ob_source == "Wicks":
            source_col = 'high' if is_high else 'low'
        else: # Bodys
            # Common interpretation for body-based pivots:
            # Use close for high pivot (peak of buying pressure), open for low pivot (peak of selling pressure).
            source_col = 'close' if is_high else 'open'

        if source_col not in df.columns:
            log.error(f"Source column '{source_col}' not found for pivot calculation based on ob_source='{self.ob_source}'.")
            return pd.Series(np.nan, index=df.index, dtype=float)

        source = df[source_col].astype(float) # Ensure float type for comparisons

        # --- Vectorized Pivot Logic ---
        # Create boolean masks for left and right comparisons, initialized to True
        # Using numpy arrays for potentially faster boolean operations
        source_np = source.to_numpy()
        left_check_np = np.ones(len(source_np), dtype=bool)
        right_check_np = np.ones(len(source_np), dtype=bool)

        # Check left side (strict inequality)
        for k in range(1, left + 1):
            shifted_source = source.shift(k).to_numpy()
            if is_high:
                left_check_np &= (source_np > shifted_source)
            else:
                left_check_np &= (source_np < shifted_source)

        # Check right side (non-strict inequality)
        for k in range(1, right + 1):
            shifted_source = source.shift(-k).to_numpy()
            if is_high:
                right_check_np &= (source_np >= shifted_source)
            else:
                right_check_np &= (source_np <= shifted_source)

        # A pivot exists where both left and right conditions are met.
        # Also ensure the source value itself is not NaN.
        is_pivot_np = left_check_np & right_check_np & ~np.isnan(source_np)

        # Invalidate pivots too close to the start/end of the series where lookback/lookforward is incomplete
        is_pivot_np[:left] = False
        is_pivot_np[-right:] = False

        # Return the source value where a pivot is detected, NaN otherwise
        pivots_np = np.where(is_pivot_np, source_np, np.nan)

        return pd.Series(pivots_np, index=df.index, name=f'pivot_{"high" if is_high else "low"}')

    def _update_trend_levels(self, df: pd.DataFrame, current_trend_up: bool, trend_just_changed: bool):
        """
        Updates the dynamic trend levels (upper, lower, vol_bands) based on the latest ATR and EMA.
        This is called when the trend is first identified or when it changes.

        Args:
            df: The DataFrame containing indicator data.
            current_trend_up: The current trend direction (True=UP, False=DOWN).
            trend_just_changed: Boolean indicating if the trend changed on the last candle.
        """
        # Update if trend is identified for the first time OR if it changed on this candle
        if self.current_trend is None or trend_just_changed:
            is_initial_trend = self.current_trend is None
            previous_trend = self.current_trend
            self.current_trend = current_trend_up # Update strategy's trend state

            # Log only if the trend actually changed or is being initialized
            if is_initial_trend or (previous_trend is not None and previous_trend != current_trend_up):
                trend_status_str = 'UP' if current_trend_up else 'DOWN'
                log.info(f"{Fore.CYAN}{'Initial Trend Detected' if is_initial_trend else 'Trend Changed'}! "
                         f"New Trend: {trend_status_str}. Updating levels...{Style.RESET_ALL}")

                # Use the EMA1 and ATR from the *current* candle where change is confirmed.
                last_row = df.iloc[-1]
                current_ema1 = last_row.get('ema1') # Use .get for safety
                current_atr = last_row.get('atr')

                # Ensure EMA1 and ATR are valid numbers and ATR is positive
                if pd.notna(current_ema1) and pd.notna(current_atr) and current_atr > FLOAT_EPSILON:
                    self.upper = self.round_price(current_ema1 + current_atr * ATR_MULTIPLIER_MAIN)
                    self.lower = self.round_price(current_ema1 - current_atr * ATR_MULTIPLIER_MAIN)
                    # Vol bands are offset from the *opposite* main band
                    # Round these as well for consistent comparisons
                    self.lower_vol = self.round_price(self.lower + current_atr * ATR_MULTIPLIER_VOL)
                    self.upper_vol = self.round_price(self.upper - current_atr * ATR_MULTIPLIER_VOL)
                    # Ensure vol bands don't cross main bands (clamp them)
                    self.lower_vol = max(self.lower_vol, self.lower) # Lower vol cannot be below lower main
                    self.upper_vol = min(self.upper_vol, self.upper) # Upper vol cannot be above upper main

                    log.info(f"Levels Updated @ {df.index[-1]}: U={self.format_price(self.upper)}, L={self.format_price(self.lower)}, "
                             f"U_Vol={self.format_price(self.upper_vol)}, L_Vol={self.format_price(self.lower_vol)}")
                else:
                    log.warning(f"Could not update levels @ {df.index[-1]} due to NaN/zero EMA1({current_ema1})/ATR({current_atr}). Levels reset.")
                    self.upper, self.lower, self.lower_vol, self.upper_vol = None, None, None, None
            # else: Trend confirmed but didn't change from previous state, no need to update levels again.

    def _create_order_blocks(self, df: pd.DataFrame):
        """
        Identifies new pivot points from df['ph']/df['pl'] in recent history
        and creates corresponding OrderBlock dictionary entries.
        Appends new OBs to self.bull_boxes or self.bear_boxes.

        Args:
            df: DataFrame containing pivot data ('ph', 'pl').
        """
        new_boxes_created_count = 0
        # Use integer index for OB tracking and referencing df.iloc

        # Only need to check recent bars for *new* pivots that could form OBs.
        # A pivot at index 'i' is confirmed after 'right' bars pass. We look for non-NaN values in df['ph'] / df['pl'].
        # Check bars from (current - right_lookback - buffer) index for pivots.
        # The pivot itself must have occurred within this window.
        check_start_idx = max(0, len(df) - max(self.pivot_right_h, self.pivot_right_l) - PIVOT_CHECK_LOOKBACK_BUFFER)
        current_bar_int_idx = len(df) - 1

        # Iterate through potential pivot locations in the recent part of the dataframe
        # Using iloc for integer-based indexing
        for i in range(check_start_idx, len(df)):
            pivot_high_price = df['ph'].iloc[i]
            pivot_low_price = df['pl'].iloc[i]

            # --- Check for Bearish Box from Pivot High ---
            if pd.notna(pivot_high_price):
                pivot_occur_int_idx = i # The integer index where the pivot high occurred
                # Check if an OB (active or inactive) already exists for this exact pivot index to avoid duplicates
                if not any(b['id'] == pivot_occur_int_idx for b in self.bear_boxes):
                    ob_candle = df.iloc[pivot_occur_int_idx] # Get the candle data at the pivot index
                    top_price, bottom_price = np.nan, np.nan

                    # Define OB boundaries based on source ('Wicks' or 'Bodys')
                    if self.ob_source == "Wicks":
                        # Bear OB (Wicks): High to Close of the pivot candle. Often an up candle before a down move.
                        top_price = ob_candle.get('high')
                        bottom_price = ob_candle.get('close') # Common definition
                    else: # Bodys
                        # Bear OB (Bodys): Body of the pivot candle. Typically Open to Close for an up candle.
                        open_price = ob_candle.get('open')
                        close_price = ob_candle.get('close')
                        if pd.notna(open_price) and pd.notna(close_price):
                            top_price = max(open_price, close_price)
                            bottom_price = min(open_price, close_price)

                    # Ensure valid prices and top > bottom
                    if pd.notna(top_price) and pd.notna(bottom_price) and top_price > bottom_price:
                        # Check if OB has a meaningful range (e.g., > tick size * factor)
                        try:
                            if Decimal(str(top_price)) - Decimal(str(bottom_price)) >= self.price_tick * MIN_OB_RANGE_FACTOR:
                                new_box = OrderBlock(id=pivot_occur_int_idx, type='bear', left_idx=pivot_occur_int_idx,
                                                     right_idx=current_bar_int_idx, # Initially valid until current bar
                                                     top=self.round_price(top_price), # Round OB boundaries
                                                     bottom=self.round_price(bottom_price),
                                                     active=True, closed_idx=None)
                                self.bear_boxes.append(new_box)
                                new_boxes_created_count += 1
                                log.info(f"{Fore.RED}New Bear OB [{pivot_occur_int_idx}] @ {df.index[pivot_occur_int_idx]}: "
                                         f"T={self.format_price(new_box['top'])}, B={self.format_price(new_box['bottom'])}{Style.RESET_ALL}")
                            # else: log.debug(f"Skipping Bear OB [{pivot_occur_int_idx}]: Range too small.")
                        except InvalidOperation:
                             log.warning(f"Could not create Bear OB [{pivot_occur_int_idx}] due to invalid price for Decimal: T={top_price}, B={bottom_price}")

            # --- Check for Bullish Box from Pivot Low ---
            if pd.notna(pivot_low_price):
                pivot_occur_int_idx = i # The integer index where the pivot low occurred
                # Check if an OB already exists for this exact pivot index
                if not any(b['id'] == pivot_occur_int_idx for b in self.bull_boxes):
                    ob_candle = df.iloc[pivot_occur_int_idx] # Get the candle data at the pivot index
                    top_price, bottom_price = np.nan, np.nan

                    # Define OB boundaries based on source
                    if self.ob_source == "Wicks":
                        # Bull OB (Wicks): Open to Low of the pivot candle. Often a down candle before an up move.
                        top_price = ob_candle.get('open') # Common definition
                        bottom_price = ob_candle.get('low')
                    else: # Bodys
                        # Bull OB (Bodys): Body of the pivot candle. Typically Open to Close for a down candle.
                        open_price = ob_candle.get('open')
                        close_price = ob_candle.get('close')
                        if pd.notna(open_price) and pd.notna(close_price):
                            top_price = max(open_price, close_price)
                            bottom_price = min(open_price, close_price)

                    # Ensure valid prices and top > bottom
                    if pd.notna(top_price) and pd.notna(bottom_price) and top_price > bottom_price:
                        # Check if OB has a meaningful range
                         try:
                            if Decimal(str(top_price)) - Decimal(str(bottom_price)) >= self.price_tick * MIN_OB_RANGE_FACTOR:
                                new_box = OrderBlock(id=pivot_occur_int_idx, type='bull', left_idx=pivot_occur_int_idx,
                                                     right_idx=current_bar_int_idx, # Initially valid until current bar
                                                     top=self.round_price(top_price), # Round OB boundaries
                                                     bottom=self.round_price(bottom_price),
                                                     active=True, closed_idx=None)
                                self.bull_boxes.append(new_box)
                                new_boxes_created_count += 1
                                log.info(f"{Fore.GREEN}New Bull OB [{pivot_occur_int_idx}] @ {df.index[pivot_occur_int_idx]}: "
                                         f"T={self.format_price(new_box['top'])}, B={self.format_price(new_box['bottom'])}{Style.RESET_ALL}")
                            # else: log.debug(f"Skipping Bull OB [{pivot_occur_int_idx}]: Range too small.")
                         except InvalidOperation:
                             log.warning(f"Could not create Bull OB [{pivot_occur_int_idx}] due to invalid price for Decimal: T={top_price}, B={bottom_price}")

        # Log if any boxes were created in this update cycle
        # if new_boxes_created_count > 0:
        #     log.debug(f"Created {new_boxes_created_count} new order block(s). Total: {len(self.bull_boxes)} bull, {len(self.bear_boxes)} bear.")

    def _manage_order_blocks(self, df: pd.DataFrame):
        """
        Invalidates Order Blocks based on the latest price action (close price)
        and prunes the lists (self.bull_boxes, self.bear_boxes) to keep
        only the most recent active and a limited history of inactive ones.

        Args:
            df: DataFrame containing the latest candle data.
        """
        if not self.bull_boxes and not self.bear_boxes:
            return # Nothing to manage

        if df.empty:
            log.warning("Cannot manage OBs: DataFrame is empty.")
            return

        last_row = df.iloc[-1]
        current_close = last_row.get('close')
        # Using close for invalidation is common. Could use high/low for stricter rules.
        current_bar_int_idx = len(df) - 1 # Get integer index of the current bar

        if pd.isna(current_close):
            log.warning(f"Cannot manage OBs @ {df.index[-1]}: Current close price is NaN.")
            return # Cannot manage OBs without a valid close price

        closed_bull_count, closed_bear_count = 0, 0

        # --- Invalidate existing boxes ---
        # Check Bull Boxes: Invalidate if the candle's close goes below the bottom of the OB.
        for box in self.bull_boxes:
            if box['active']:
                # Invalidation condition: Candle close breaks below the OB low
                # Use a small epsilon tolerance for floating point comparison robustness
                if current_close < box['bottom'] - FLOAT_EPSILON:
                    box['active'] = False
                    box['closed_idx'] = current_bar_int_idx
                    closed_bull_count += 1
                    log.debug(f"Closing Bull OB [{box['id']}] (B={self.format_price(box['bottom'])}) due to close {self.format_price(current_close)}")
                else:
                    # Keep extending the validity (right_idx) if still active
                    box['right_idx'] = current_bar_int_idx

        # Check Bear Boxes: Invalidate if the candle's close goes above the top of the OB.
        for box in self.bear_boxes:
            if box['active']:
                # Invalidation condition: Candle close breaks above the OB high
                # Use a small epsilon tolerance
                if current_close > box['top'] + FLOAT_EPSILON:
                    box['active'] = False
                    box['closed_idx'] = current_bar_int_idx
                    closed_bear_count += 1
                    log.debug(f"Closing Bear OB [{box['id']}] (T={self.format_price(box['top'])}) due to close {self.format_price(current_close)}")
                else:
                    # Keep extending the validity (right_idx) if still active
                    box['right_idx'] = current_bar_int_idx

        if closed_bull_count: log.info(f"{Fore.YELLOW}Closed {closed_bull_count} Bull OB(s) @ {df.index[-1]} due to price ({self.format_price(current_close)}) breaking below.{Style.RESET_ALL}")
        if closed_bear_count: log.info(f"{Fore.YELLOW}Closed {closed_bear_count} Bear OB(s) @ {df.index[-1]} due to price ({self.format_price(current_close)}) breaking above.{Style.RESET_ALL}")

        # --- Prune Order Blocks ---
        # Keep only the 'max_boxes' most recent *active* boxes, plus a limited history
        # of recent *inactive* ones (e.g., for visualization or debugging).
        # Sort by ID (which corresponds to the bar index) descending, so newest are first.

        # Prune Bull Boxes
        active_bull = sorted([b for b in self.bull_boxes if b['active']], key=lambda x: x['id'], reverse=True)
        inactive_bull = sorted([b for b in self.bull_boxes if not b['active']], key=lambda x: x['id'], reverse=True)
        # Keep max_boxes active, and maybe max_boxes * 2 inactive for potential debugging/visualization
        self.bull_boxes = active_bull[:self.max_boxes] + inactive_bull[:self.max_boxes * 2]

        # Prune Bear Boxes
        active_bear = sorted([b for b in self.bear_boxes if b['active']], key=lambda x: x['id'], reverse=True)
        inactive_bear = sorted([b for b in self.bear_boxes if not b['active']], key=lambda x: x['id'], reverse=True)
        self.bear_boxes = active_bear[:self.max_boxes] + inactive_bear[:self.max_boxes * 2]

        # Optional: Log pruning action if lists were actually shortened significantly
        # log.debug(f"Pruned OB lists: {len(self.bull_boxes)} bull ({len(active_bull)} active), "
        #           f"{len(self.bear_boxes)} bear ({len(active_bear)} active) remaining.")


    def _generate_signal(self, df: pd.DataFrame) -> str:
        """
        Generates the trading signal ('BUY', 'SELL', 'EXIT_LONG', 'EXIT_SHORT', 'HOLD')
        based on the current trend, active order blocks, and the last signal state.

        Signal Logic:
        1. Priority Exit: If trend just changed against the current desired state, signal exit.
        2. Entry: If trend is UP and price enters an active Bull OB, signal BUY (if not already BUY state).
        3. Entry: If trend is DOWN and price enters an active Bear OB, signal SELL (if not already SELL state).
        4. Hold: Otherwise, signal HOLD.

        Args:
            df: DataFrame with the latest candle data.

        Returns:
            The generated signal string.
        """
        signal = "HOLD" # Default signal

        if df.empty: return signal # Cannot generate signal on empty data

        last_row = df.iloc[-1]
        current_close = last_row.get('close')
        # trend_changed reflects change *on this candle* compared to previous
        trend_just_changed = bool(last_row.get('trend_changed', False)) # Default to False if column missing

        # Get currently active boxes after management/pruning
        active_bull_boxes = [b for b in self.bull_boxes if b['active']]
        active_bear_boxes = [b for b in self.bear_boxes if b['active']]

        # Only generate signals if trend is defined and close price is valid
        if self.current_trend is None or pd.isna(current_close):
            log.debug(f"Signal: HOLD (Trend={self.current_trend}, Close={current_close} unavailable)")
            return "HOLD" # Cannot determine signal without trend or price

        trend_str = 'UP' if self.current_trend else 'DOWN'

        # --- Priority 1: Exit signal if trend just changed against the desired position state ---
        if trend_just_changed:
            if not self.current_trend and self.last_signal_state == "BUY": # Trend changed DOWN while wanting LONG
                signal = "EXIT_LONG"
                log.warning(f"{Fore.YELLOW}{Style.BRIGHT}*** EXIT_LONG Signal: Trend changed to DOWN while in BUY state. ***{Style.RESET_ALL}")
            elif self.current_trend and self.last_signal_state == "SELL": # Trend changed UP while wanting SHORT
                signal = "EXIT_SHORT"
                log.warning(f"{Fore.YELLOW}{Style.BRIGHT}*** EXIT_SHORT Signal: Trend changed to UP while in SELL state. ***{Style.RESET_ALL}")

        # --- Priority 2: Entry signals if trend allows and not already in desired state ---
        if signal == "HOLD": # Only check for entry if no exit signal was generated
            if self.current_trend: # Trend is UP -> Look for Long Entry
                if self.last_signal_state != "BUY": # Only enter if not already trying to be long
                    # Check if price entered any *active* Bullish OB
                    for box in active_bull_boxes:
                        # Entry condition: Current close is within the Bull OB range (inclusive)
                        # Use epsilon for float comparison robustness
                        if box['bottom'] - FLOAT_EPSILON <= current_close <= box['top'] + FLOAT_EPSILON:
                            signal = "BUY"
                            log.warning(f"{Fore.GREEN}{Style.BRIGHT}*** BUY Signal trigger: Price {self.format_price(current_close)} entered Bull OB "
                                     f"[{box['id']}] ({self.format_price(box['bottom'])} - {self.format_price(box['top'])}) (Trend: {trend_str}) ***{Style.RESET_ALL}")
                            break # Take the first valid OB entry found (newest active first due to pruning sort)
            elif not self.current_trend: # Trend is DOWN -> Look for Short Entry
                if self.last_signal_state != "SELL": # Only enter if not already trying to be short
                    # Check if price entered any *active* Bearish OB
                    for box in active_bear_boxes:
                        # Entry condition: Current close is within the Bear OB range (inclusive)
                        # Use epsilon for float comparison robustness
                        if box['bottom'] - FLOAT_EPSILON <= current_close <= box['top'] + FLOAT_EPSILON:
                            signal = "SELL"
                            log.warning(f"{Fore.RED}{Style.BRIGHT}*** SELL Signal trigger: Price {self.format_price(current_close)} entered Bear OB "
                                     f"[{box['id']}] ({self.format_price(box['bottom'])} - {self.format_price(box['top'])}) (Trend: {trend_str}) ***{Style.RESET_ALL}")
                            break # Take the first valid OB entry found (newest active first due to pruning sort)

        # Update internal state based on the generated signal
        if signal == "BUY":
            self.last_signal_state = "BUY"
        elif signal == "SELL":
            self.last_signal_state = "SELL"
        elif signal in ["EXIT_LONG", "EXIT_SHORT"]:
            self.last_signal_state = "HOLD" # Reset state after exit signal is generated

        # Log the final signal if it's HOLD (for debugging clarity)
        if signal == "HOLD":
             log.debug(f"Signal: HOLD (Internal State: {self.last_signal_state}, Trend: {trend_str}, Close: {self.format_price(current_close)})")

        return signal


    def update(self, df_input: pd.DataFrame) -> AnalysisResults:
        """
        Calculates indicators and generates signals based on the input DataFrame.
        This is the main method called to get the latest strategy analysis.

        Args:
            df_input: pandas DataFrame with OHLCV columns ('open', 'high', 'low', 'close', 'volume')
                      and a DatetimeIndex. Assumes the last row is the most recent *closed* candle.

        Returns:
            AnalysisResults dictionary containing the updated dataframe, the latest signal,
            active order blocks, last close price, current trend status, trend change flag,
            and last ATR value.
        """
        # Basic validation of input DataFrame
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if df_input.empty or not all(col in df_input.columns for col in required_cols):
            log.error(f"Input DataFrame is empty or missing required columns ({required_cols}). Cannot analyze.")
            # Return a default state indicating failure/no analysis
            return AnalysisResults(
                dataframe=df_input, # Return original potentially incomplete dataframe
                last_signal="HOLD",
                active_bull_boxes=[b for b in self.bull_boxes if b.get('active', False)], # Safely get active boxes
                active_bear_boxes=[b for b in self.bear_boxes if b.get('active', False)],
                last_close=np.nan,
                current_trend=self.current_trend, # Keep previous trend state
                trend_changed=False,
                last_atr=None
            )

        if len(df_input) < self.min_data_len:
            log.warning(f"Not enough data ({len(df_input)}/{self.min_data_len}) for analysis. Need more historical bars.")
            # Return current state but indicate HOLD signal as analysis is incomplete
            last_close = df_input['close'].iloc[-1] if not df_input.empty else np.nan
            return AnalysisResults(
                dataframe=df_input, # Return original dataframe
                last_signal="HOLD",
                active_bull_boxes=[b for b in self.bull_boxes if b.get('active', False)], # Return current known active boxes
                active_bear_boxes=[b for b in self.bear_boxes if b.get('active', False)],
                last_close=last_close,
                current_trend=self.current_trend,
                trend_changed=False, # No change detectable
                last_atr=None # Cannot calculate ATR reliably
            )

        # Work on a copy to avoid modifying the original DataFrame passed to the function
        df = df_input.copy()
        log.debug(f"Analyzing {len(df)} candles ending {df.index[-1]}...")

        # Ensure correct numeric types, coercing errors to NaN
        for col in required_cols:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                 log.warning(f"Required column '{col}' missing from input DataFrame during type conversion.")
                 df[col] = np.nan # Add column as NaN if missing

        # --- Indicator Calculations ---
        # ATR for trend levels and potentially risk management
        # Let pandas_ta handle NaN filling (default is NaN propagation)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.vol_atr_period)

        # Trend Indicators: Smoothed EMA (ema1) vs Regular EMA (ema2)
        df['ema1'] = self._ema_swma(df['close'], length=self.length) # Smoothed EMA = EMA(SWMA(4))
        df['ema2'] = ta.ema(df['close'], length=self.length) # Regular EMA

        # Determine trend direction based on EMA comparison
        # True = UP (smoothed ema < regular ema -> recent price action accelerating upwards)
        # False = DOWN (smoothed ema >= regular ema)
        # Using >= for DOWN ensures a defined state even if EMAs are equal.
        # Ensure comparison handles NaNs gracefully -> result is NaN if either EMA is NaN
        df['trend_up'] = np.where(df['ema1'] < df['ema2'], True,
                         np.where(df['ema1'] >= df['ema2'], False, np.nan)) # Use np.nan where EMAs are NaN

        # Convert 'trend_up' to nullable boolean type for consistency if needed, though ffill handles float NaNs fine
        # df['trend_up'] = df['trend_up'].astype('boolean') # Optional: Use pandas nullable boolean

        # Forward fill the trend to handle initial NaNs where EMAs haven't converged yet.
        # This assumes the first calculated trend persists until a change is detected.
        df['trend_up'] = df['trend_up'].ffill()

        # Detect trend changes: True if trend_up is different from the previous non-NaN trend_up value.
        # Ensure we compare boolean/float values correctly after ffill
        trend_shifted = df['trend_up'].shift(1)
        df['trend_changed'] = (df['trend_up'] != trend_shifted) & df['trend_up'].notna() & trend_shifted.notna()

        # Get values from the last (most recent closed) candle
        last_row = df.iloc[-1]
        # Handle potential NaN at start if EMAs haven't converged yet
        current_trend_up_on_last_candle = last_row.get('trend_up') # Use .get for safety
        trend_just_changed_on_last_candle = bool(last_row.get('trend_changed', False))
        last_atr_value = last_row.get('atr')

        # Update dynamic trend levels if trend is determined and either initializing or just changed
        if pd.notna(current_trend_up_on_last_candle):
            self._update_trend_levels(df, bool(current_trend_up_on_last_candle), trend_just_changed_on_last_candle)
        else:
             # If trend is still NaN on the last candle, ensure levels are reset
             if self.current_trend is not None:
                 log.warning(f"Trend became undetermined (NaN) on last candle {df.index[-1]}. Resetting levels.")
                 self.upper, self.lower, self.lower_vol, self.upper_vol = None, None, None, None
                 self.current_trend = None # Reset internal trend state


        # --- Volume Normalization (Optional - for context/potential future use) ---
        # Calculates volume relative to a rolling percentile of recent volume.
        roll_window = min(self.vol_percentile_len, len(df))
        min_periods_vol = max(1, min(roll_window // 2, 50)) # Heuristic for min periods in rolling calculation
        if roll_window > min_periods_vol and 'volume' in df.columns and df['volume'].notna().any():
            try:
                # Calculate the percentile value using rolling apply (can be slow on very large DFs)
                # Ensure we handle slices with only NaNs or zeros gracefully inside the lambda
                df['vol_percentile_val'] = df['volume'].rolling(window=roll_window, min_periods=min_periods_vol).apply(
                    lambda x: np.nanpercentile(x[x > FLOAT_EPSILON], self.vol_percentile) if np.any(x > FLOAT_EPSILON) else np.nan,
                    raw=True # Use raw=True for potential speedup
                )
                # Normalize volume against the calculated percentile value
                # Ensure division by zero or NaN is handled
                df['vol_norm'] = np.where(
                    (df['vol_percentile_val'].notna()) & (df['vol_percentile_val'] > FLOAT_EPSILON), # Check percentile is valid & non-zero
                    (df['volume'].fillna(0.0) / df['vol_percentile_val'] * 100.0), # Normalize to percentage of percentile, fillna volume just in case
                    0.0 # Assign 0 if percentile is NaN/zero, or volume is zero/NaN
                )
                # Fill any remaining NaNs (e.g., from initial rolling period) with 0
                # FIX: Avoid inplace=True on slice/copy
                df['vol_norm'] = df['vol_norm'].fillna(0.0)
            except Exception as e:
                 log.error(f"Error calculating volume normalization: {e}", exc_info=True)
                 df['vol_percentile_val'] = np.nan
                 df['vol_norm'] = 0.0
        else:
            df['vol_percentile_val'] = np.nan
            df['vol_norm'] = 0.0 # Assign default if calculation skipped

        # --- Pivot & Order Block Calculations ---
        # Calculate pivot highs and lows based on the configured parameters
        df['ph'] = self._find_pivots(df, self.pivot_left_h, self.pivot_right_h, is_high=True)
        df['pl'] = self._find_pivots(df, self.pivot_left_l, self.pivot_right_l, is_high=False)

        # Identify new OBs based on the calculated pivots in recent history
        self._create_order_blocks(df)

        # Invalidate/prune existing OBs based on the latest candle's close price
        self._manage_order_blocks(df)

        # --- Signal Generation ---
        # Generate the final trading signal based on trend, OBs, and internal state
        signal = self._generate_signal(df)

        # --- Prepare Results ---
        last_close_price = last_row.get('close', np.nan) # Safely get last close
        active_bull_boxes = [b for b in self.bull_boxes if b.get('active', False)] # Use .get for safety
        active_bear_boxes = [b for b in self.bear_boxes if b.get('active', False)]

        # Optional: Clean up intermediate indicator columns from the returned DataFrame if desired for production
        # Consider keeping them for debugging/analysis purposes
        # cols_to_drop = ['ema1', 'ema2', 'trend_up', 'trend_changed', 'ph', 'pl', 'vol_percentile_val', 'vol_norm']
        # df_results = df.drop(columns=cols_to_drop, errors='ignore')
        df_results = df # Return the full dataframe with all calculations

        return AnalysisResults(
            dataframe=df_results, # Return the updated dataframe
            last_signal=signal,
            active_bull_boxes=active_bull_boxes,
            active_bear_boxes=active_bear_boxes,
            last_close=last_close_price,
            current_trend=self.current_trend, # The strategy's current understanding of the trend
            trend_changed=trend_just_changed_on_last_candle, # Whether the trend changed on this specific candle
            last_atr=last_atr_value if pd.notna(last_atr_value) else None # Return last ATR or None
        )
