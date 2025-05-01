# -*- coding: utf-8 -*-
"""
Strategy Definition: Enhanced Volumatic Trend + OB
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style
from typing import List, Dict, Optional, Any, TypedDict, Tuple
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP

log = logging.getLogger(__name__) # Gets logger configured in main.py

# --- Type Definitions (consistent with main.py) ---
class OrderBlock(TypedDict):
    id: int # Index of the candle where the OB formed
    type: str # 'bull' or 'bear'
    left_idx: int # Index of bar where OB formed
    right_idx: int # Index of last bar OB is valid for (updated if active)
    top: float
    bottom: float
    active: bool # Still considered valid?
    closed_idx: Optional[int] # Index where it was invalidated

class AnalysisResults(TypedDict):
    dataframe: pd.DataFrame
    last_signal: str # 'BUY', 'SELL', 'HOLD', 'EXIT_LONG', 'EXIT_SHORT'
    active_bull_boxes: List[OrderBlock]
    active_bear_boxes: List[OrderBlock]
    last_close: float
    current_trend: Optional[bool] # True for UP, False for DOWN
    trend_changed: bool
    # Add calculated indicator values needed for SL/TP etc.
    last_atr: Optional[float]

# --- Strategy Class ---
class VolumaticOBStrategy:
    """
    Encapsulates the logic from the 'Enhanced Volumatic Trend + OB' Pine Script.
    Calculates indicators and generates trading signals.
    Accepts market_info for precision handling.
    """
    def __init__(self, market_info: Dict[str, Any], **params):
        """
        Initializes the strategy.
        Args:
            market_info: Dictionary containing instrument details from Bybit.
            **params: Strategy parameters loaded from config.json.
        """
        self.market_info = market_info
        self._parse_params(params)
        self._validate_params()

        # State Variables
        self.upper: Optional[float] = None
        self.lower: Optional[float] = None
        self.lower_vol: Optional[float] = None
        self.upper_vol: Optional[float] = None
        self.step_up: Optional[float] = None
        self.step_dn: Optional[float] = None
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []
        # last_signal_state tracks the *intended* state ('BUY' means trying to be long, 'SELL' means trying to be short)
        self.last_signal_state = "HOLD" # Used internally for signal generation logic
        self.current_trend: Optional[bool] = None # True=UP, False=DOWN

        # Calculate minimum required data length
        self.min_data_len = max(
            self.length + 4, # For _ema_swma shift
            self.vol_atr_period,
            self.vol_percentile_len,
            self.pivot_left_h + self.pivot_right_h + 1,
            self.pivot_left_l + self.pivot_right_l + 1
        )

        # Get precision from market_info
        self.tick_size = Decimal(self.market_info['priceFilter']['tickSize'])
        self.qty_step = Decimal(self.market_info['lotSizeFilter']['qtyStep'])
        self.price_precision = self._get_decimal_places(self.tick_size)
        self.qty_precision = self._get_decimal_places(self.qty_step)

        log.info(f"{Fore.MAGENTA}Initializing VolumaticOB Strategy Engine...{Style.RESET_ALL}")
        log.info(f"Params: TrendLen={self.length}, Pivots={self.pivot_left_h}/{self.pivot_right_h}, "
                 f"MaxBoxes={self.max_boxes}, OB Source={self.ob_source}")
        log.info(f"Minimum data points required: {self.min_data_len}")
        log.debug(f"Tick Size: {self.tick_size}, Qty Step: {self.qty_step}")
        log.debug(f"Price Precision: {self.price_precision}, Qty Precision: {self.qty_precision}")

    def _parse_params(self, params: Dict):
        """Load parameters from the config dict."""
        self.length = int(params.get('length', 40))
        self.vol_atr_period = int(params.get('vol_atr_period', 200))
        self.vol_percentile_len = int(params.get('vol_percentile_len', 1000))
        self.vol_percentile = int(params.get('vol_percentile', 100))
        self.ob_source = str(params.get('ob_source', "Wicks"))
        self.pivot_left_h = int(params.get('pivot_left_h', 25))
        self.pivot_right_h = int(params.get('pivot_right_h', 25))
        self.pivot_left_l = int(params.get('pivot_left_l', 25))
        self.pivot_right_l = int(params.get('pivot_right_l', 25))
        self.max_boxes = int(params.get('max_boxes', 10))

    def _validate_params(self):
        """Basic validation of strategy parameters."""
        if self.ob_source not in ["Wicks", "Bodys"]:
            raise ValueError("ob_source must be 'Wicks' or 'Bodys'")
        lengths = [self.length, self.vol_atr_period, self.vol_percentile_len,
                   self.pivot_left_h, self.pivot_right_h, self.pivot_left_l, self.pivot_right_l]
        if not all(isinstance(x, int) and x > 0 for x in lengths):
             raise ValueError("All length/period parameters must be positive integers.")
        if not isinstance(self.max_boxes, int) or self.max_boxes <= 0:
             raise ValueError("max_boxes must be a positive integer.")
        if not 0 < self.vol_percentile <= 100:
             raise ValueError("vol_percentile must be between 1 and 100.")

    def _get_decimal_places(self, decimal_val: Decimal) -> int:
        """Calculates decimal places from a Decimal object."""
        return abs(decimal_val.as_tuple().exponent) if decimal_val.as_tuple().exponent < 0 else 0

    def round_price(self, price: float) -> float:
        """Rounds price according to tickSize (down for sell SL/TP, up for buy SL/TP if needed, default round half up)."""
        # Simple rounding based on tick size decimal places
        # For SL/TP, might need ROUND_DOWN or ROUND_UP depending on side
        return float(Decimal(str(price)).quantize(self.tick_size)) # Default rounding mode

    def round_qty(self, qty: float) -> float:
        """Rounds quantity DOWN according to qtyStep."""
        # Use Decimal for precision and round down
        qty_decimal = Decimal(str(qty))
        rounded_qty = (qty_decimal // self.qty_step) * self.qty_step
        return float(rounded_qty)

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Helper for smoothed EMA (weighted avg + EMA)."""
        if len(series) < 4:
            return pd.Series(np.nan, index=series.index)

        weights = np.array([1, 2, 2, 1]) / 6.0
        swma = series.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)
        ema_of_swma = ta.ema(swma, length=length, fillna=np.nan)
        return ema_of_swma

    def _find_pivots(self, df: pd.DataFrame, left: int, right: int, is_high: bool) -> pd.Series:
        """Finds pivot points similar to Pine Script's ta.pivothigh/low."""
        if self.ob_source == "Wicks":
             source_col = 'high' if is_high else 'low'
        else: # Bodys
             source_col = 'close' if is_high else 'open'

        if source_col not in df.columns:
            log.error(f"Source column '{source_col}' not found for pivot calc.")
            return pd.Series(np.nan, index=df.index)

        source_series = df[source_col]
        pivots = pd.Series(np.nan, index=df.index)

        # Iterate through possible pivot points (excluding boundaries where lookback/forward is not possible)
        for i in range(left, len(df) - right):
            pivot_val = source_series.iloc[i]
            if pd.isna(pivot_val): continue

            is_pivot = True
            # Check left side (indices i-left to i-1)
            for j in range(1, left + 1):
                left_val = source_series.iloc[i - j]
                if pd.isna(left_val): continue # Skip comparison if neighbor is NaN (can happen with sparse data)
                # Strict inequality on the left side
                if (is_high and left_val > pivot_val) or \
                   (not is_high and left_val < pivot_val):
                    is_pivot = False; break
            if not is_pivot: continue

            # Check right side (indices i+1 to i+right)
            for j in range(1, right + 1):
                right_val = source_series.iloc[i + j]
                if pd.isna(right_val): continue
                # Pine Script logic: >= for high pivots, <= for low pivots on right side
                if (is_high and right_val >= pivot_val) or \
                   (not is_high and right_val <= pivot_val):
                    is_pivot = False; break

            if is_pivot:
                pivots.iloc[i] = pivot_val # Store the pivot value at the pivot index `i`
        return pivots

    def update(self, df_input: pd.DataFrame) -> AnalysisResults:
        """
        Calculates indicators and generates signals based on the input DataFrame.
        Args:
            df_input: pandas DataFrame with ['open', 'high', 'low', 'close', 'volume']
                      index must be DatetimeIndex, sorted chronologically.
        Returns:
            AnalysisResults dictionary.
        """
        required_len = self.min_data_len
        if df_input.empty or len(df_input) < required_len:
            log.warning(f"Not enough data ({len(df_input)}/{required_len}) for analysis.")
            # Return default/previous state carefully
            return AnalysisResults(
                dataframe=df_input, last_signal="HOLD", active_bull_boxes=self.bull_boxes, # Return existing boxes
                active_bear_boxes=self.bear_boxes, last_close=np.nan, current_trend=self.current_trend,
                trend_changed=False, last_atr=None # Need last ATR from previous run if possible? Complex. Setting None is safer.
            )

        df = df_input.copy()
        log.debug(f"Analyzing {len(df)} candles ending {df.index[-1]}...")

        # --- Volumatic Trend Calculations ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.vol_atr_period, fillna=np.nan)
        df['ema1'] = self._ema_swma(df['close'], length=self.length)
        df['ema2'] = ta.ema(df['close'], length=self.length, fillna=np.nan)

        # Determine trend, ffill to handle initial NaNs
        df['trend_up'] = np.where(df['ema1'] < df['ema2'], True,
                         np.where(df['ema1'] >= df['ema2'], False, np.nan))
        df['trend_up'] = df['trend_up'].ffill()

        # Detect trend change, ignoring NaNs and initial state
        df['trend_changed'] = (df['trend_up'] != df['trend_up'].shift(1)) & df['trend_up'].notna() & df['trend_up'].shift(1).notna()

        # --- Update Levels on Trend Change ---
        last_row = df.iloc[-1]
        current_trend_up = last_row['trend_up']
        trend_just_changed = last_row['trend_changed']
        last_atr_value = last_row['atr']

        # Update persistent trend state if it's valid
        if pd.notna(current_trend_up):
            if self.current_trend is None: # First valid trend detection
                self.current_trend = current_trend_up
                log.info(f"Initial Trend detected: {'UP' if self.current_trend else 'DOWN'}")
                # Trigger level calculation on first detection too
                trend_just_changed = True # Force level update

            elif trend_just_changed and current_trend_up != self.current_trend:
                self.current_trend = current_trend_up # Update trend state
                log.info(f"{Fore.MAGENTA}Trend Changed! New Trend: {'UP' if current_trend_up else 'DOWN'}. Updating levels...{Style.RESET_ALL}")

                # Find the row where the trend actually changed to get levels
                # Need to handle case where trend_changed is True but current_trend_up hasn't updated yet? Check df.iloc[-2] trend?
                # Simpler: use the latest values if trend just changed
                current_ema1 = last_row['ema1']
                current_atr = last_row['atr']

                if pd.notna(current_ema1) and pd.notna(current_atr) and current_atr > 1e-9: # Ensure ATR is positive
                    self.upper = current_ema1 + current_atr * 3
                    self.lower = current_ema1 - current_atr * 3
                    self.lower_vol = self.lower + current_atr * 4
                    self.upper_vol = self.upper - current_atr * 4
                    # Prevent levels from crossing due to large ATR
                    if self.lower_vol < self.lower: self.lower_vol = self.lower
                    if self.upper_vol > self.upper: self.upper_vol = self.upper

                    self.step_up = (self.lower_vol - self.lower) / 100 if self.lower_vol > self.lower else 0
                    self.step_dn = (self.upper - self.upper_vol) / 100 if self.upper > self.upper_vol else 0

                    log.info(f"Levels Updated @ {df.index[-1]}: U={self.upper:.{self.price_precision}f}, L={self.lower:.{self.price_precision}f}")

                else:
                     log.warning(f"Could not update levels at {df.index[-1]} due to NaN/zero values (EMA1={current_ema1}, ATR={current_atr}). Levels reset.")
                     self.upper, self.lower, self.lower_vol, self.upper_vol, self.step_up, self.step_dn = [None] * 6
            # else: Trend did not change, levels remain the same

        # --- Volume Normalization ---
        roll_window = min(self.vol_percentile_len, len(df)) # Ensure window doesn't exceed data length
        min_p = max(1, min(roll_window // 2, 50)) # Require reasonable number of periods
        # Use np.nanpercentile for robustness against NaNs within the window
        df['vol_percentile_val'] = df['volume'].rolling(window=roll_window,min_periods=min_p).apply(
            lambda x: np.nanpercentile(x, self.vol_percentile) if np.any(~np.isnan(x) & (x > 0)) else np.nan,
            raw=True
        )

        df['vol_norm'] = np.where(
            (df['vol_percentile_val'].notna()) & (df['vol_percentile_val'] > 1e-9), # Avoid division by near-zero
            (df['volume'] / df['vol_percentile_val'] * 100),
            0 # Assign 0 if percentile is NaN or (near) zero
        ).fillna(0).astype(float)

        # --- Pivot Order Block Calculations ---
        df['ph'] = self._find_pivots(df, self.pivot_left_h, self.pivot_right_h, is_high=True)
        df['pl'] = self._find_pivots(df, self.pivot_left_l, self.pivot_right_l, is_high=False)

        # --- Create and Manage Order Blocks ---
        # Iterate only through bars where new pivots *could* have been confirmed
        # A pivot at index `i` is confirmed `right` bars later.
        # We only need to check bars from `max(right_h, right_l)` ago up to now.
        check_start_idx = max(0, len(df) - max(self.pivot_right_h, self.pivot_right_l) - 5) # Add buffer
        new_boxes_created_count = 0

        # Create integer index mapping for box IDs
        df['int_index'] = range(len(df))

        for i in range(check_start_idx, len(df)):
            # Use integer index for list management and box ID
            current_int_index = i

            # Bearish Box from Pivot High confirmed at bar `i`
            # The actual pivot occurred at `i - self.pivot_right_h`
            pivot_occur_int_idx = current_int_index - self.pivot_right_h
            if pivot_occur_int_idx < 0: continue # Ensure pivot index is valid

            # Check if a PH value exists at the pivot *occurrence* index
            if pd.notna(df['ph'].iloc[pivot_occur_int_idx]):
                # Check if a box for this pivot *occurrence* index already exists
                if not any(b['id'] == pivot_occur_int_idx for b in self.bear_boxes):
                    ob_candle = df.iloc[pivot_occur_int_idx] # Candle where OB is defined
                    top_price, bottom_price = np.nan, np.nan

                    if self.ob_source == "Wicks":
                        top_price = ob_candle['high']
                        bottom_price = ob_candle['close'] # Common definition uses close
                    else: # Bodys
                        top_price = ob_candle['close']
                        bottom_price = ob_candle['open']

                    # Ensure valid prices and top > bottom
                    if pd.notna(top_price) and pd.notna(bottom_price):
                        if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price # Swap if needed

                        # Check for zero-range OB (can happen with dojis etc.)
                        if abs(top_price - bottom_price) > 1e-9 * self.tick_size.to_eng_string(): # Compare against tick size order of magnitude
                            new_box = OrderBlock(
                                id=pivot_occur_int_idx, # Use integer index as ID
                                type='bear', left_idx=pivot_occur_int_idx,
                                right_idx=len(df)-1, # Extend to current bar initially
                                top=top_price, bottom=bottom_price, active=True, closed_idx=None
                            )
                            self.bear_boxes.append(new_box)
                            new_boxes_created_count += 1
                            log.info(f"{Fore.RED}New Bear OB {pivot_occur_int_idx} @ {df.index[pivot_occur_int_idx]}: T={top_price:.{self.price_precision}f}, B={bottom_price:.{self.price_precision}f}{Style.RESET_ALL}")

            # Bullish Box from Pivot Low confirmed at bar `i`
            # The actual pivot occurred at `i - self.pivot_right_l`
            pivot_occur_int_idx = current_int_index - self.pivot_right_l
            if pivot_occur_int_idx < 0: continue

            if pd.notna(df['pl'].iloc[pivot_occur_int_idx]):
                if not any(b['id'] == pivot_occur_int_idx for b in self.bull_boxes):
                    ob_candle = df.iloc[pivot_occur_int_idx]
                    top_price, bottom_price = np.nan, np.nan

                    if self.ob_source == "Wicks":
                         top_price = ob_candle['open'] # Common definition uses open
                         bottom_price = ob_candle['low']
                    else: # Bodys
                         top_price = ob_candle['open']
                         bottom_price = ob_candle['close']

                    if pd.notna(top_price) and pd.notna(bottom_price):
                        if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price

                        if abs(top_price - bottom_price) > 1e-9 * self.tick_size.to_eng_string():
                            new_box = OrderBlock(
                                id=pivot_occur_int_idx, type='bull', left_idx=pivot_occur_int_idx,
                                right_idx=len(df)-1,
                                top=top_price, bottom=bottom_price, active=True, closed_idx=None
                            )
                            self.bull_boxes.append(new_box)
                            new_boxes_created_count += 1
                            log.info(f"{Fore.GREEN}New Bull OB {pivot_occur_int_idx} @ {df.index[pivot_occur_int_idx]}: T={top_price:.{self.price_precision}f}, B={bottom_price:.{self.price_precision}f}{Style.RESET_ALL}")

        if new_boxes_created_count > 0:
             log.debug(f"Created {new_boxes_created_count} new order blocks.")

        # --- Manage existing boxes (close or extend) ---
        current_close = last_row['close']
        current_bar_int_idx = len(df) - 1 # Use simple integer index for management

        if pd.notna(current_close): # Only manage boxes if close price is valid
            closed_bull_count = 0
            closed_bear_count = 0
            for box in self.bull_boxes:
                if box['active']:
                    if current_close < box['bottom']: # Price closed below bull box
                        box['active'] = False
                        box['closed_idx'] = current_bar_int_idx
                        closed_bull_count += 1
                    else:
                        box['right_idx'] = current_bar_int_idx # Extend active box

            for box in self.bear_boxes:
                if box['active']:
                    if current_close > box['top']: # Price closed above bear box
                        box['active'] = False
                        box['closed_idx'] = current_bar_int_idx
                        closed_bear_count +=1
                    else:
                        box['right_idx'] = current_bar_int_idx

            if closed_bull_count > 0: log.info(f"{Fore.YELLOW}Closed {closed_bull_count} Bull OBs due to price violation.{Style.RESET_ALL}")
            if closed_bear_count > 0: log.info(f"{Fore.YELLOW}Closed {closed_bear_count} Bear OBs due to price violation.{Style.RESET_ALL}")

        # --- Prune Order Blocks ---
        # Keep only the 'max_boxes' most recent *active* boxes of each type
        # Also keep some recent inactive ones for potential debugging/display
        active_bull = sorted([b for b in self.bull_boxes if b['active']], key=lambda x: x['id'], reverse=True)
        inactive_bull = sorted([b for b in self.bull_boxes if not b['active']], key=lambda x: x['id'], reverse=True)
        initial_bull_len = len(self.bull_boxes)
        self.bull_boxes = inactive_bull[:self.max_boxes * 2] + active_bull[:self.max_boxes]
        if len(self.bull_boxes) < initial_bull_len: log.debug(f"Pruned {initial_bull_len - len(self.bull_boxes)} Bull OBs.")


        active_bear = sorted([b for b in self.bear_boxes if b['active']], key=lambda x: x['id'], reverse=True)
        inactive_bear = sorted([b for b in self.bear_boxes if not b['active']], key=lambda x: x['id'], reverse=True)
        initial_bear_len = len(self.bear_boxes)
        self.bear_boxes = inactive_bear[:self.max_boxes * 2] + active_bear[:self.max_boxes]
        if len(self.bear_boxes) < initial_bear_len: log.debug(f"Pruned {initial_bear_len - len(self.bear_boxes)} Bear OBs.")


        # --- Signal Generation ---
        signal = "HOLD"
        active_bull_boxes = [b for b in self.bull_boxes if b['active']]
        active_bear_boxes = [b for b in self.bear_boxes if b['active']]

        # Check conditions only if trend and close price are valid
        if self.current_trend is not None and pd.notna(current_close):
            # Check for Trend Change Exit first
            if trend_just_changed:
                # Check internal state to see if we were in a position that needs exiting
                if not self.current_trend and self.last_signal_state == "BUY": # Trend flipped down while intended long
                    signal = "EXIT_LONG"
                    log.warning(f"{Fore.YELLOW}{Style.BRIGHT}*** EXIT LONG Signal (Trend Flip to DOWN) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}")
                elif self.current_trend and self.last_signal_state == "SELL": # Trend flipped up while intended short
                    signal = "EXIT_SHORT"
                    log.warning(f"{Fore.YELLOW}{Style.BRIGHT}*** EXIT SHORT Signal (Trend Flip to UP) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}")

            # Check for Entries only if not exiting and not already in desired state
            if signal == "HOLD":
                if self.current_trend: # Trend is UP, look for Long Entries
                    if self.last_signal_state != "BUY": # Only enter if not already intending long
                        for box in active_bull_boxes:
                            # Check if close touches or is within the box range
                            if box['bottom'] <= current_close <= box['top']:
                                signal = "BUY"
                                log.info(f"{Fore.GREEN}{Style.BRIGHT}*** BUY Signal (Trend UP + Price in Bull OB {box['id']}) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}")
                                break # Take first signal

                elif not self.current_trend: # Trend is DOWN, look for Short Entries
                    if self.last_signal_state != "SELL": # Only enter if not already intending short
                        for box in active_bear_boxes:
                            if box['bottom'] <= current_close <= box['top']:
                                signal = "SELL"
                                log.info(f"{Fore.RED}{Style.BRIGHT}*** SELL Signal (Trend DOWN + Price in Bear OB {box['id']}) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}")
                                break

        # Update internal state *only* if signal implies a state change
        if signal in ["BUY", "SELL"]:
            self.last_signal_state = signal
        elif signal in ["EXIT_LONG", "EXIT_SHORT"]:
            self.last_signal_state = "HOLD" # After exit, we are neutral until new entry

        log.debug(f"Signal Generated: {signal} (Internal State: {self.last_signal_state})")

        # Drop the temporary integer index column before returning
        df.drop(columns=['int_index'], inplace=True, errors='ignore')

        return AnalysisResults(
            dataframe=df, # Return the DataFrame with calculated indicators
            last_signal=signal, # The action signal for this candle
            active_bull_boxes=active_bull_boxes,
            active_bear_boxes=active_bear_boxes,
            last_close=current_close,
            current_trend=self.current_trend,
            trend_changed=trend_just_changed,
            last_atr=last_atr_value if pd.notna(last_atr_value) else None
        )
