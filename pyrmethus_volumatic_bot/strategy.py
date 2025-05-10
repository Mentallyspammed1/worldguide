# -*- coding: utf-8 -*-
"""
Strategy Definition: Enhanced Volumatic Trend + OB
Calculates indicators, identifies order blocks, and generates trading signals.
"""

import logging
from typing import List, Dict, Optional, Any, TypedDict
from decimal import Decimal, ROUND_DOWN

import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style


# --- Type Definitions ---
# Consistent type hints improve readability and allow static analysis.
class OrderBlock(TypedDict):
    id: int  # Unique identifier (e.g., index of the candle where OB formed)
    type: str  # 'bull' or 'bear'
    left_idx: int  # Index of bar where OB formed
    right_idx: int  # Index of last bar OB is valid for (updated if active)
    top: float  # Top price of the OB
    bottom: float  # Bottom price of the OB
    active: bool  # Still considered valid?
    closed_idx: Optional[int]  # Index where it was invalidated/closed


class AnalysisResults(TypedDict):
    dataframe: pd.DataFrame  # DataFrame with indicators
    last_signal: str  # 'BUY', 'SELL', 'HOLD', 'EXIT_LONG', 'EXIT_SHORT'
    active_bull_boxes: List[OrderBlock]
    active_bear_boxes: List[OrderBlock]
    last_close: float  # Last closing price
    current_trend: Optional[bool]  # True for UP, False for DOWN, None if undetermined
    trend_changed: bool  # True if trend changed on the last candle
    last_atr: Optional[float]  # Last calculated ATR value (for SL/TP)


# --- Strategy Class ---
class VolumaticOBStrategy:
    """
    Encapsulates the logic from the 'Enhanced Volumatic Trend + OB' concept.
    Calculates indicators and generates trading signals based on market data.
    Accepts market_info for precision handling during calculations and rounding.
    """

    def __init__(self, market_info: Dict[str, Any], **params):
        """
        Initializes the strategy engine.

        Args:
            market_info: Dictionary containing instrument details (tick size, qty step)
                         fetched from the exchange API (e.g., Bybit).
            **params: Strategy parameters loaded from the configuration file.
                      See _parse_params for expected keys.
        """
        self.log = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )  # Specific logger
        if not market_info:
            raise ValueError(
                "Market info (tickSize, qtyStep) is required for strategy initialization."
            )
        self.market_info = market_info
        self._parse_params(params)
        self._validate_params()

        # State Variables - These persist across calls to update()
        self.upper: Optional[float] = None
        self.lower: Optional[float] = None
        self.lower_vol: Optional[float] = None
        self.upper_vol: Optional[float] = None
        self.step_up: Optional[float] = None
        self.step_dn: Optional[float] = None
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []
        # last_signal_state tracks the *intended* state based on signals generated.
        # 'BUY' means the strategy generated a buy signal (aiming to be long).
        # 'SELL' means the strategy generated a sell signal (aiming to be short).
        # 'HOLD' means neutral or after an exit signal.
        self.last_signal_state: str = "HOLD"
        self.current_trend: Optional[bool] = None  # True=UP, False=DOWN

        # Calculate minimum required data length based on indicator periods
        self.min_data_len = max(
            self.length + 4,  # For _ema_swma shift
            self.vol_atr_period,
            self.vol_percentile_len,
            self.pivot_left_h + self.pivot_right_h + 1,  # Pivot lookback/forward
            self.pivot_left_l + self.pivot_right_l + 1,
        )

        # Extract precision details from market_info using Decimal for accuracy
        try:
            self.tick_size = Decimal(self.market_info["priceFilter"]["tickSize"])
            self.qty_step = Decimal(self.market_info["lotSizeFilter"]["qtyStep"])
        except (KeyError, TypeError) as e:
            self.log.error(f"Failed to extract tickSize/qtyStep from market_info: {e}")
            raise ValueError(
                "Market info missing required price/lot filter details."
            ) from e

        self.price_precision = self._get_decimal_places(self.tick_size)
        self.qty_precision = self._get_decimal_places(self.qty_step)

        self.log.info(
            f"{Fore.MAGENTA}Initializing VolumaticOB Strategy Engine...{Style.RESET_ALL}"
        )
        self.log.info(
            f"Params: TrendLen={self.length}, Pivots={self.pivot_left_h}/{self.pivot_right_h}, "
            f"MaxBoxes={self.max_boxes}, OB Source={self.ob_source}"
        )
        self.log.info(f"Minimum data points required: {self.min_data_len}")
        self.log.debug(f"Tick Size: {self.tick_size}, Qty Step: {self.qty_step}")
        self.log.debug(
            f"Price Precision: {self.price_precision}, Qty Precision: {self.qty_precision}"
        )

    def _parse_params(self, params: Dict[str, Any]):
        """Load and type-cast parameters from the config dictionary."""
        self.length = int(params.get("length", 40))
        self.vol_atr_period = int(params.get("vol_atr_period", 200))
        self.vol_percentile_len = int(params.get("vol_percentile_len", 1000))
        self.vol_percentile = int(params.get("vol_percentile", 100))
        self.ob_source = str(params.get("ob_source", "Wicks"))
        self.pivot_left_h = int(params.get("pivot_left_h", 10))
        self.pivot_right_h = int(params.get("pivot_right_h", 10))
        self.pivot_left_l = int(params.get("pivot_left_l", 10))
        self.pivot_right_l = int(params.get("pivot_right_l", 10))
        self.max_boxes = int(params.get("max_boxes", 5))
        # Load SL params too
        sl_config = params.get("stop_loss", {})
        self.sl_method = str(sl_config.get("method", "ATR"))
        self.sl_atr_multiplier = float(sl_config.get("atr_multiplier", 2.0))

    def _validate_params(self):
        """Perform basic validation of strategy parameters."""
        if self.ob_source not in ["Wicks", "Bodys"]:
            raise ValueError("Invalid 'ob_source'. Must be 'Wicks' or 'Bodys'.")
        lengths = [
            self.length,
            self.vol_atr_period,
            self.vol_percentile_len,
            self.pivot_left_h,
            self.pivot_right_h,
            self.pivot_left_l,
            self.pivot_right_l,
        ]
        if not all(isinstance(x, int) and x > 0 for x in lengths):
            raise ValueError("All length/period parameters must be positive integers.")
        if not isinstance(self.max_boxes, int) or self.max_boxes <= 0:
            raise ValueError("'max_boxes' must be a positive integer.")
        if not 0 < self.vol_percentile <= 100:
            raise ValueError("'vol_percentile' must be between 1 and 100.")
        if self.sl_method not in ["ATR", "OB"]:
            raise ValueError("Invalid 'stop_loss.method'. Must be 'ATR' or 'OB'.")
        if self.sl_atr_multiplier <= 0:
            raise ValueError("'stop_loss.atr_multiplier' must be positive.")

    def _get_decimal_places(self, decimal_val: Decimal) -> int:
        """Calculates the number of decimal places from a Decimal object."""
        # Using abs() handles cases like Decimal('1e-8') correctly
        return (
            abs(decimal_val.as_tuple().exponent)
            if decimal_val.is_finite() and decimal_val.as_tuple().exponent < 0
            else 0
        )

    def round_price(self, price: float, rounding_mode=ROUND_DOWN) -> float:
        """
        Rounds a price according to the market's tickSize.
        Uses ROUND_DOWN by default, suitable for sell SL/TP or conservative entries.
        Use ROUND_UP for buy SL/TP.
        """
        if not isinstance(price, (float, int)) or not np.isfinite(price):
            self.log.warning(
                f"Invalid price value for rounding: {price}. Returning NaN."
            )
            return np.nan
        try:
            # Use Decimal for precise rounding based on tick_size
            return float(
                Decimal(str(price)).quantize(self.tick_size, rounding=rounding_mode)
            )
        except Exception as e:
            self.log.error(
                f"Error rounding price {price} with tick_size {self.tick_size}: {e}"
            )
            return np.nan  # Return NaN on error

    def round_qty(self, qty: float) -> float:
        """
        Rounds quantity DOWN according to the market's qtyStep.
        Ensures the quantity is a valid multiple of the minimum step.
        """
        if not isinstance(qty, (float, int)) or not np.isfinite(qty) or qty < 0:
            self.log.warning(
                f"Invalid quantity value for rounding: {qty}. Returning 0."
            )
            return 0.0
        try:
            # Use Decimal division and multiplication to round down to the nearest step
            qty_decimal = Decimal(str(qty))
            rounded_qty = (qty_decimal // self.qty_step) * self.qty_step
            return float(rounded_qty)
        except Exception as e:
            self.log.error(
                f"Error rounding quantity {qty} with qty_step {self.qty_step}: {e}"
            )
            return 0.0  # Return 0 on error

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculates a smoothed EMA variant (SWMA of price, then EMA of SWMA).
        Handles potential NaNs in the input series.
        """
        if series.isnull().all() or len(series) < 4 or length <= 0:
            return pd.Series(np.nan, index=series.index)

        # Symmetric Weighted Moving Average (SWMA) weights [1, 2, 2, 1] / 6
        weights = np.array([1, 2, 2, 1]) / 6.0
        # Use rolling apply with raw=True for potential speedup
        swma = series.rolling(window=4, min_periods=4).apply(
            lambda x: np.dot(x, weights) if not np.isnan(x).any() else np.nan, raw=True
        )

        # Calculate EMA of the SWMA result
        # Use adjust=False for behavior closer to some TA libraries/platforms
        ema_of_swma = ta.ema(swma.dropna(), length=length, adjust=False)

        # Reindex to match original series index, filling gaps with NaN
        return ema_of_swma.reindex(series.index)

    def _find_pivots(
        self, df: pd.DataFrame, left: int, right: int, is_high: bool
    ) -> pd.Series:
        """
        Finds pivot high or pivot low points in a DataFrame series.
        Mimics the logic of Pine Script's ta.pivothigh/low functions.

        Args:
            df: DataFrame containing price data.
            left: Number of bars to the left to check.
            right: Number of bars to the right to check.
            is_high: True to find pivot highs, False for pivot lows.

        Returns:
            A pandas Series with pivot values at the index where they occur, NaN otherwise.
        """
        if self.ob_source == "Wicks":
            # Use high for pivot highs, low for pivot lows
            source_col = "high" if is_high else "low"
        else:  # Bodys
            # Use close for pivot highs, open for pivot lows (common interpretation)
            source_col = "close" if is_high else "open"

        if source_col not in df.columns:
            self.log.error(
                f"Source column '{source_col}' not found in DataFrame for pivot calculation."
            )
            return pd.Series(np.nan, index=df.index)

        source_series = df[source_col]
        pivots = pd.Series(np.nan, index=df.index)

        # Efficient vectorized approach (faster than Python loop for large data)
        # Pad series to handle boundary conditions during shifts easily
        padded_source = pd.concat(
            [pd.Series([np.nan] * left), source_series, pd.Series([np.nan] * right)]
        )

        # Check left side: current value must be strictly greater/less than left neighbors
        left_check = True
        for i in range(1, left + 1):
            shifted = padded_source.shift(i)
            if is_high:
                left_check &= padded_source > shifted
            else:
                left_check &= padded_source < shifted

        # Check right side: current value must be greater/less than or equal to right neighbors
        right_check = True
        for i in range(1, right + 1):
            shifted = padded_source.shift(-i)
            if is_high:
                right_check &= padded_source >= shifted
            else:
                right_check &= padded_source <= shifted

        # Combine checks and align back to original index
        is_pivot = (left_check & right_check).iloc[left:-right]  # Remove padding
        is_pivot.index = df.index  # Align index

        # Assign the source value where a pivot is detected
        pivots[is_pivot] = source_series[is_pivot]

        return pivots

    def update(self, df_input: pd.DataFrame) -> AnalysisResults:
        """
        Calculates indicators, manages order blocks, and generates trading signals
        based on the input DataFrame.

        Args:
            df_input: pandas DataFrame with ['open', 'high', 'low', 'close', 'volume'] columns.
                      Index must be a DatetimeIndex, sorted chronologically.

        Returns:
            An AnalysisResults dictionary containing the updated DataFrame, signals,
            active order blocks, and other relevant state information.
        """
        if df_input.empty or len(df_input) < self.min_data_len:
            self.log.warning(
                f"Not enough data ({len(df_input)}/{self.min_data_len}) for analysis. Returning current state."
            )
            # Return current state without modifying the input DataFrame
            return AnalysisResults(
                dataframe=df_input,  # Return original df
                last_signal="HOLD",
                active_bull_boxes=[
                    b for b in self.bull_boxes if b["active"]
                ],  # Return copies of active boxes
                active_bear_boxes=[b for b in self.bear_boxes if b["active"]],
                last_close=df_input["close"].iloc[-1] if not df_input.empty else np.nan,
                current_trend=self.current_trend,
                trend_changed=False,
                last_atr=None,  # Cannot calculate ATR with insufficient data
            )

        # Work on a copy to avoid modifying the original DataFrame passed in
        df = df_input.copy()
        self.log.debug(f"Analyzing {len(df)} candles ending {df.index[-1]}...")

        # --- Volumatic Trend Calculations ---
        df["atr"] = ta.atr(
            df["high"], df["low"], df["close"], length=self.vol_atr_period
        )
        df["ema1"] = self._ema_swma(df["close"], length=self.length)
        df["ema2"] = ta.ema(
            df["close"], length=self.length, adjust=False
        )  # Use adjust=False

        # Determine trend direction (True=UP, False=DOWN)
        # Use np.select for clarity, ffill to handle initial NaNs
        conditions = [df["ema1"] < df["ema2"], df["ema1"] >= df["ema2"]]
        choices = [True, False]
        df["trend_up"] = np.select(conditions, choices, default=np.nan)
        df["trend_up"] = df[
            "trend_up"
        ].ffill()  # Forward fill trend after initial calculation

        # Detect trend change, ignoring NaNs and the very first valid trend
        df["trend_changed"] = (
            (df["trend_up"] != df["trend_up"].shift(1))
            & df["trend_up"].notna()
            & df["trend_up"].shift(1).notna()
        )

        # --- Update Levels on Trend Change ---
        last_row = df.iloc[-1]
        current_trend_up = last_row["trend_up"]
        trend_just_changed = last_row["trend_changed"]
        last_atr_value = last_row["atr"]
        current_ema1 = last_row["ema1"]

        # Update persistent trend state if the trend value is valid
        if pd.notna(current_trend_up):
            is_initial_trend = self.current_trend is None
            if is_initial_trend:
                self.current_trend = current_trend_up
                self.log.info(
                    f"Initial Trend detected: {'UP' if self.current_trend else 'DOWN'}"
                )
                trend_just_changed = True  # Force level update on first detection

            elif trend_just_changed and current_trend_up != self.current_trend:
                self.current_trend = current_trend_up  # Update internal trend state
                self.log.info(
                    f"{Fore.MAGENTA}Trend Changed! New Trend: {'UP' if current_trend_up else 'DOWN'}. Updating levels...{Style.RESET_ALL}"
                )
                # Level update logic moved inside the trend change block

            # Update levels IF trend just changed (or it's the initial trend) AND necessary values are valid
            if (
                trend_just_changed
                and pd.notna(current_ema1)
                and pd.notna(last_atr_value)
                and last_atr_value > 1e-9
            ):
                self.upper = current_ema1 + last_atr_value * 3
                self.lower = current_ema1 - last_atr_value * 3
                self.lower_vol = self.lower + last_atr_value * 4
                self.upper_vol = self.upper - last_atr_value * 4
                # Prevent levels from crossing due to large ATR or calculation quirks
                self.lower_vol = max(self.lower_vol, self.lower)
                self.upper_vol = min(self.upper_vol, self.upper)

                self.step_up = (
                    (self.lower_vol - self.lower) / 100
                    if self.lower_vol > self.lower
                    else 0
                )
                self.step_dn = (
                    (self.upper - self.upper_vol) / 100
                    if self.upper > self.upper_vol
                    else 0
                )

                self.log.info(
                    f"Levels Updated @ {df.index[-1]}: U={self.upper:.{self.price_precision}f}, L={self.lower:.{self.price_precision}f}"
                )
            elif trend_just_changed:
                self.log.warning(
                    f"Could not update levels at {df.index[-1]} due to NaN/zero values (EMA1={current_ema1}, ATR={last_atr_value}). Levels remain unchanged or reset."
                )
                # Optionally reset levels if update fails:
                # self.upper, self.lower, self.lower_vol, self.upper_vol, self.step_up, self.step_dn = [None] * 6

        # --- Volume Normalization ---
        # Calculate rolling percentile of volume
        roll_window = min(self.vol_percentile_len, len(df))
        min_periods_vol = max(
            1, roll_window // 2
        )  # Require a reasonable number of periods
        # Use np.nanpercentile for robustness against NaNs within the window
        # Rolling apply can be slow; consider alternatives if needed
        df["vol_percentile_val"] = (
            df["volume"]
            .rolling(window=roll_window, min_periods=min_periods_vol)
            .apply(
                lambda x: np.nanpercentile(x[x > 0], self.vol_percentile)
                if np.any(x > 0)
                else np.nan,
                raw=True,  # raw=True might be faster
            )
        )

        # Normalize volume based on the percentile value
        df["vol_norm"] = np.where(
            (df["vol_percentile_val"].notna())
            & (df["vol_percentile_val"] > 1e-9),  # Avoid division by zero/NaN
            (df["volume"] / df["vol_percentile_val"] * 100),
            0,  # Assign 0 if percentile is NaN or (near) zero
        )
        df["vol_norm"] = (
            df["vol_norm"].fillna(0).astype(float)
        )  # Ensure float type and fill any remaining NaNs

        # --- Pivot Order Block Calculations ---
        df["ph"] = self._find_pivots(
            df, self.pivot_left_h, self.pivot_right_h, is_high=True
        )
        df["pl"] = self._find_pivots(
            df, self.pivot_left_l, self.pivot_right_l, is_high=False
        )

        # --- Create and Manage Order Blocks ---
        # Use integer indices for easier list management and box IDs
        df["int_index"] = range(len(df))
        current_bar_int_idx = len(df) - 1

        # Check for newly formed pivots in the recent data
        # A pivot at index `p` is confirmed `right` bars later (at index `p + right`).
        # We only need to check bars where pivots might have just been confirmed.
        check_start_idx = max(
            0, len(df) - max(self.pivot_right_h, self.pivot_right_l) - 5
        )  # Check recent bars + buffer
        new_boxes_created_count = 0

        for i in range(check_start_idx, len(df)):
            # Check for Bearish Box (Pivot High confirmation)
            # Pivot occurred at index `i - pivot_right_h`
            bear_pivot_occur_idx = i - self.pivot_right_h
            if bear_pivot_occur_idx >= 0 and pd.notna(
                df["ph"].iloc[bear_pivot_occur_idx]
            ):
                # Check if a box for this pivot index already exists
                if not any(b["id"] == bear_pivot_occur_idx for b in self.bear_boxes):
                    ob_candle = df.iloc[bear_pivot_occur_idx]
                    top_price, bottom_price = np.nan, np.nan

                    if self.ob_source == "Wicks":
                        top_price = ob_candle["high"]
                        # Common definition: use close for bear OB bottom (body top)
                        bottom_price = max(ob_candle["open"], ob_candle["close"])
                    else:  # Bodys
                        top_price = max(ob_candle["open"], ob_candle["close"])
                        bottom_price = min(ob_candle["open"], ob_candle["close"])

                    # Ensure valid prices and top >= bottom
                    if (
                        pd.notna(top_price)
                        and pd.notna(bottom_price)
                        and top_price >= bottom_price
                    ):
                        # Check for zero-range OB (e.g., doji) - allow small range based on tick size
                        if abs(top_price - bottom_price) > float(
                            self.tick_size / 10
                        ):  # Allow OB if range > 1/10th tick
                            new_box = OrderBlock(
                                id=bear_pivot_occur_idx,
                                type="bear",
                                left_idx=bear_pivot_occur_idx,
                                right_idx=current_bar_int_idx,  # Valid up to current bar
                                top=top_price,
                                bottom=bottom_price,
                                active=True,
                                closed_idx=None,
                            )
                            self.bear_boxes.append(new_box)
                            new_boxes_created_count += 1
                            self.log.info(
                                f"{Fore.RED}New Bear OB {bear_pivot_occur_idx} @ {df.index[bear_pivot_occur_idx]}: T={top_price:.{self.price_precision}f}, B={bottom_price:.{self.price_precision}f}{Style.RESET_ALL}"
                            )

            # Check for Bullish Box (Pivot Low confirmation)
            # Pivot occurred at index `i - pivot_right_l`
            bull_pivot_occur_idx = i - self.pivot_right_l
            if bull_pivot_occur_idx >= 0 and pd.notna(
                df["pl"].iloc[bull_pivot_occur_idx]
            ):
                # Check if a box for this pivot index already exists
                if not any(b["id"] == bull_pivot_occur_idx for b in self.bull_boxes):
                    ob_candle = df.iloc[bull_pivot_occur_idx]
                    top_price, bottom_price = np.nan, np.nan

                    if self.ob_source == "Wicks":
                        # Common definition: use open for bull OB top (body bottom)
                        top_price = min(ob_candle["open"], ob_candle["close"])
                        bottom_price = ob_candle["low"]
                    else:  # Bodys
                        top_price = max(ob_candle["open"], ob_candle["close"])
                        bottom_price = min(ob_candle["open"], ob_candle["close"])

                    if (
                        pd.notna(top_price)
                        and pd.notna(bottom_price)
                        and top_price >= bottom_price
                    ):
                        if abs(top_price - bottom_price) > float(self.tick_size / 10):
                            new_box = OrderBlock(
                                id=bull_pivot_occur_idx,
                                type="bull",
                                left_idx=bull_pivot_occur_idx,
                                right_idx=current_bar_int_idx,
                                top=top_price,
                                bottom=bottom_price,
                                active=True,
                                closed_idx=None,
                            )
                            self.bull_boxes.append(new_box)
                            new_boxes_created_count += 1
                            self.log.info(
                                f"{Fore.GREEN}New Bull OB {bull_pivot_occur_idx} @ {df.index[bull_pivot_occur_idx]}: T={top_price:.{self.price_precision}f}, B={bottom_price:.{self.price_precision}f}{Style.RESET_ALL}"
                            )

        if new_boxes_created_count > 0:
            self.log.debug(f"Created {new_boxes_created_count} new order blocks.")

        # --- Manage existing boxes (close or extend validity) ---
        current_close = last_row["close"]
        closed_bull_count = 0
        closed_bear_count = 0

        if pd.notna(current_close):  # Only manage boxes if close price is valid
            for box in self.bull_boxes:
                if box["active"]:
                    # Invalidate Bull OB if price closes below its bottom
                    if current_close < box["bottom"]:
                        box["active"] = False
                        box["closed_idx"] = current_bar_int_idx
                        closed_bull_count += 1
                    else:
                        # Otherwise, extend its validity to the current bar
                        box["right_idx"] = current_bar_int_idx

            for box in self.bear_boxes:
                if box["active"]:
                    # Invalidate Bear OB if price closes above its top
                    if current_close > box["top"]:
                        box["active"] = False
                        box["closed_idx"] = current_bar_int_idx
                        closed_bear_count += 1
                    else:
                        box["right_idx"] = current_bar_int_idx

            if closed_bull_count > 0:
                self.log.info(
                    f"{Fore.YELLOW}Closed {closed_bull_count} Bull OBs due to price violation.{Style.RESET_ALL}"
                )
            if closed_bear_count > 0:
                self.log.info(
                    f"{Fore.YELLOW}Closed {closed_bear_count} Bear OBs due to price violation.{Style.RESET_ALL}"
                )

        # --- Prune Order Blocks ---
        # Keep only the 'max_boxes' most recent *active* boxes of each type.
        # Keep a limited number of recent inactive ones for context/debugging.
        active_bull = sorted(
            [b for b in self.bull_boxes if b["active"]],
            key=lambda x: x["id"],
            reverse=True,
        )
        inactive_bull = sorted(
            [b for b in self.bull_boxes if not b["active"]],
            key=lambda x: x["id"],
            reverse=True,
        )
        initial_bull_len = len(self.bull_boxes)
        # Keep max_boxes active + double that number inactive (most recent ones)
        self.bull_boxes = (
            active_bull[: self.max_boxes] + inactive_bull[: self.max_boxes * 2]
        )
        if len(self.bull_boxes) < initial_bull_len:
            self.log.debug(
                f"Pruned {initial_bull_len - len(self.bull_boxes)} older Bull OBs."
            )

        active_bear = sorted(
            [b for b in self.bear_boxes if b["active"]],
            key=lambda x: x["id"],
            reverse=True,
        )
        inactive_bear = sorted(
            [b for b in self.bear_boxes if not b["active"]],
            key=lambda x: x["id"],
            reverse=True,
        )
        initial_bear_len = len(self.bear_boxes)
        self.bear_boxes = (
            active_bear[: self.max_boxes] + inactive_bear[: self.max_boxes * 2]
        )
        if len(self.bear_boxes) < initial_bear_len:
            self.log.debug(
                f"Pruned {initial_bear_len - len(self.bear_boxes)} older Bear OBs."
            )

        # --- Signal Generation ---
        signal = "HOLD"  # Default signal
        active_bull_boxes = [
            b for b in self.bull_boxes if b["active"]
        ]  # Get currently active boxes
        active_bear_boxes = [b for b in self.bear_boxes if b["active"]]

        # Check conditions only if trend and close price are valid
        if self.current_trend is not None and pd.notna(current_close):
            # 1. Check for Trend Change Exit first
            if trend_just_changed:
                # Check internal state to see if we were in a position that needs exiting
                if (
                    not self.current_trend and self.last_signal_state == "BUY"
                ):  # Trend flipped DOWN while intended LONG
                    signal = "EXIT_LONG"
                    self.log.warning(
                        f"{Fore.YELLOW}{Style.BRIGHT}*** EXIT LONG Signal (Trend Flip to DOWN) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}"
                    )
                elif (
                    self.current_trend and self.last_signal_state == "SELL"
                ):  # Trend flipped UP while intended SHORT
                    signal = "EXIT_SHORT"
                    self.log.warning(
                        f"{Fore.YELLOW}{Style.BRIGHT}*** EXIT SHORT Signal (Trend Flip to UP) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}"
                    )

            # 2. Check for Entries only if not exiting and not already in the desired state
            if signal == "HOLD":  # Only look for entries if not exiting
                if self.current_trend:  # Trend is UP -> Look for Long Entries
                    if (
                        self.last_signal_state != "BUY"
                    ):  # Only enter if not already intending long
                        # Check if price entered any active Bull OB
                        for box in active_bull_boxes:
                            # Check if close is within the box range (inclusive)
                            if box["bottom"] <= current_close <= box["top"]:
                                signal = "BUY"
                                self.log.info(
                                    f"{Fore.GREEN}{Style.BRIGHT}*** BUY Signal (Trend UP + Price in Bull OB {box['id']}) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}"
                                )
                                break  # Take the first valid signal

                elif not self.current_trend:  # Trend is DOWN -> Look for Short Entries
                    if (
                        self.last_signal_state != "SELL"
                    ):  # Only enter if not already intending short
                        # Check if price entered any active Bear OB
                        for box in active_bear_boxes:
                            if box["bottom"] <= current_close <= box["top"]:
                                signal = "SELL"
                                self.log.info(
                                    f"{Fore.RED}{Style.BRIGHT}*** SELL Signal (Trend DOWN + Price in Bear OB {box['id']}) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}"
                                )
                                break  # Take the first valid signal

        # Update internal state *only* if signal implies a state change
        if signal == "BUY":
            self.last_signal_state = "BUY"
        elif signal == "SELL":
            self.last_signal_state = "SELL"
        elif signal in ["EXIT_LONG", "EXIT_SHORT"]:
            self.last_signal_state = (
                "HOLD"  # After exit, we are neutral until a new entry signal
            )

        self.log.debug(
            f"Signal Generated: {signal} (Internal State: {self.last_signal_state})"
        )

        # Drop the temporary integer index column before returning
        df.drop(columns=["int_index"], inplace=True, errors="ignore")

        # Return results
        return AnalysisResults(
            dataframe=df,  # Return the DataFrame with calculated indicators
            last_signal=signal,
            active_bull_boxes=active_bull_boxes,  # Return only active boxes
            active_bear_boxes=active_bear_boxes,
            last_close=current_close if pd.notna(current_close) else np.nan,
            current_trend=self.current_trend,
            trend_changed=trend_just_changed,
            last_atr=last_atr_value if pd.notna(last_atr_value) else None,
        )
