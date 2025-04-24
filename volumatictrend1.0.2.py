Okay, Pyrmethus, the enhancements have been forged! The Python script has been significantly refined for clarity, robustness, and closer adherence to the intended strategy logic.

Here is the complete, improved version of `frontend_cli.py`:

```python
# -*- coding: utf-8 -*-
"""
Pyrmethus Frontend CLI: Enhanced Volumatic Trend + OB Strategy

Analyzes market data using the Volumatic Trend + Order Block strategy
and sends trading signals to the backend server for execution.
"""

import requests
import json
import time
import datetime
import threading
import websocket # websocket-client library
import math
import numpy as np
import pandas as pd
import pandas_ta as ta # For EMA, ATR etc.
from colorama import init, Fore, Style
from typing import List, Dict, Optional, Any, TypedDict, Tuple
import logging

# --- Initialize Colorama ---
init(autoreset=True)

# --- Logging Setup ---
# More robust logging than just print statements
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
BYBIT_API_BASE = "https://api.bybit.com"
BYBIT_WS_PUBLIC = "wss://stream.bybit.com/v5/public/linear"
# Address of your running backend_server.py
BACKEND_URL = "http://127.0.0.1:5000"
# Timeout for HTTP requests to Bybit API and Backend
REQUEST_TIMEOUT = 15 # seconds
# Default order quantity (adjust as needed, maybe load from config/env)
DEFAULT_ORDER_QTY = 0.001
# Max historical candles to fetch initially
FETCH_LIMIT = 750 # Ensure enough for lookbacks (pivots, vol percentile)
# Max length of DataFrame to keep in memory (for performance)
MAX_DF_LEN = 2000
# WebSocket Ping Interval
WS_PING_INTERVAL = 20 # seconds


# --- Type Definitions ---
# Keep previous TypedDicts: BybitKline, PriceDataPoint - add OHLC
class BybitKline(TypedDict): # Example structure from Bybit API
    timestamp: str
    open: str
    high: str
    low: str
    close: str
    volume: str
    turnover: str

class PriceDataPoint(TypedDict):
    timestamp: int # Keep raw timestamp for calculations
    time: str      # Formatted time string
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Indicator fields will be added dynamically or via AnalysisResults

class OrderBlock(TypedDict):
    id: int # Unique ID (e.g., bar_index where created/confirmed)
    type: str # 'bull' or 'bear'
    left_idx: int # Index of bar where OB formed
    right_idx: int # Index of last bar OB is valid for
    top: float
    bottom: float
    active: bool # Still considered valid?
    closed_idx: Optional[int] # Index where it was invalidated

class AnalysisResults(TypedDict):
    dataframe: pd.DataFrame # Store results directly in the DataFrame
    last_signal: str # 'BUY', 'SELL', 'HOLD', 'EXIT_LONG', 'EXIT_SHORT'
    active_bull_boxes: List[OrderBlock]
    active_bear_boxes: List[OrderBlock]
    last_close: float
    current_trend: Optional[bool] # True for UP, False for DOWN
    trend_changed: bool

# --- Volumatic Trend + OB Strategy Class ---
class VolumaticOBStrategy:
    """
    Encapsulates the logic from the 'Enhanced Volumatic Trend + OB' Pine Script.
    Calculates indicators and generates trading signals.
    """
    def __init__(self,
                 # Volumatic Trend Inputs
                 length: int = 40,
                 vol_atr_period: int = 200, # ATR period for vol levels
                 vol_percentile_len: int = 1000,
                 vol_percentile: int = 100,
                 # OB Inputs
                 ob_source: str = "Wicks", # "Wicks" or "Bodys"
                 pivot_left_h: int = 25,
                 pivot_right_h: int = 25,
                 pivot_left_l: int = 25,
                 pivot_right_l: int = 25,
                 max_boxes: int = 10): # Limit active boxes displayed/considered

        if ob_source not in ["Wicks", "Bodys"]:
            raise ValueError("ob_source must be 'Wicks' or 'Bodys'")

        # Store parameters
        self.length = length
        self.vol_atr_period = vol_atr_period
        # Ensure percentile length doesn't exceed fetch limit excessively
        self.vol_percentile_len = min(vol_percentile_len, FETCH_LIMIT - 50)
        self.vol_percentile = vol_percentile
        self.ob_source = ob_source
        self.pivot_left_h = pivot_left_h
        self.pivot_right_h = pivot_right_h
        self.pivot_left_l = pivot_left_l
        self.pivot_right_l = pivot_right_l
        self.max_boxes = max_boxes

        # State Variables (like 'var' in Pine Script)
        self.upper: Optional[float] = None
        self.lower: Optional[float] = None
        self.lower_vol: Optional[float] = None
        self.upper_vol: Optional[float] = None
        self.step_up: Optional[float] = None
        self.step_dn: Optional[float] = None
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []
        self.last_signal = "HOLD"
        self.current_trend: Optional[bool] = None # True=UP, False=DOWN

        # Calculate minimum required data length
        self.min_data_len = max(
            self.length + 4, # For _ema_swma shift
            self.vol_atr_period,
            self.vol_percentile_len,
            self.pivot_left_h + self.pivot_right_h + 1,
            self.pivot_left_l + self.pivot_right_l + 1
        )

        log.info(f"{Fore.MAGENTA}Initializing VolumaticOB Strategy Engine...{Style.RESET_ALL}")
        log.info(f"Params: TrendLen={length}, Pivots={pivot_left_h}/{pivot_right_h}, MaxBoxes={max_boxes}, OB Source={ob_source}")
        log.info(f"Minimum data points required: {self.min_data_len}")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Helper for smoothed EMA (weighted avg + EMA)."""
        if len(series) < 4:
            return pd.Series(np.nan, index=series.index) # Return NaN series

        # Apply 1/6, 2/6, 2/6, 1/6 weighting to the last 4 bars
        # Use fill_value=series.iloc[0] cautiously, maybe NaN is better if early data is sparse
        weighted = (series.shift(3, fill_value=np.nan) / 6 +
                    series.shift(2, fill_value=np.nan) * 2 / 6 +
                    series.shift(1, fill_value=np.nan) * 2 / 6 +
                    series / 6)
        return ta.ema(weighted, length=length, fillna=np.nan) # Propagate NaNs

    def _find_pivots(self, df: pd.DataFrame, left: int, right: int, is_high: bool) -> pd.Series:
        """
        Finds pivot points similar to Pine Script's ta.pivothigh/low.
        A pivot occurs at index `i` if the condition holds for `left` bars
        before it and `right` bars after it. The pivot value is reported
        at index `i`.
        """
        if self.ob_source == "Wicks":
             source_col = 'high' if is_high else 'low'
        else: # Bodys
             # Pine uses close for PH body, open for PL body based on common OB defs
             source_col = 'close' if is_high else 'open'

        if source_col not in df.columns:
            log.error(f"Source column '{source_col}' not found in DataFrame for pivot calculation.")
            return pd.Series(np.nan, index=df.index)

        source_series = df[source_col]
        pivots = pd.Series(np.nan, index=df.index)

        # Iterate through possible pivot points
        # Range: Need 'left' bars before and 'right' bars after the potential pivot `i`
        for i in range(left, len(df) - right):
            pivot_val = source_series.iloc[i]
            if pd.isna(pivot_val): continue # Skip if pivot value is NaN

            is_pivot = True

            # Check left side (indices i-left to i-1)
            for j in range(1, left + 1):
                left_val = source_series.iloc[i - j]
                if pd.isna(left_val): continue # Skip comparison if neighbor is NaN
                # Strict inequality on the left side
                if (is_high and left_val > pivot_val) or \
                   (not is_high and left_val < pivot_val):
                    is_pivot = False
                    break
            if not is_pivot:
                continue

            # Check right side (indices i+1 to i+right)
            for j in range(1, right + 1):
                right_val = source_series.iloc[i + j]
                if pd.isna(right_val): continue # Skip comparison if neighbor is NaN
                # Pine Script logic: >= for high pivots, <= for low pivots on right side
                if (is_high and right_val >= pivot_val) or \
                   (not is_high and right_val <= pivot_val):
                    is_pivot = False
                    break

            if is_pivot:
                pivots.iloc[i] = pivot_val # Store the pivot value at the pivot index

        return pivots

    def update(self, df_input: pd.DataFrame) -> AnalysisResults:
        """
        Calculates indicators and generates signals based on the input DataFrame.
        Args:
            df_input: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                      index must be DatetimeIndex, sorted chronologically.
        Returns:
            AnalysisResults dictionary.
        """
        if df_input.empty or len(df_input) < self.min_data_len:
            log.warning(f"Not enough data points ({len(df_input)}/{self.min_data_len}) for analysis.")
            # Return default state
            return AnalysisResults(
                dataframe=df_input, last_signal="HOLD", active_bull_boxes=[],
                active_bear_boxes=[], last_close=np.nan, current_trend=self.current_trend,
                trend_changed=False
            )

        # Work on a copy to avoid modifying the original DataFrame passed from outside
        df = df_input.copy()
        log.debug(f"Analyzing {len(df)} candles...")

        # --- Volumatic Trend Calculations ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.vol_atr_period, fillna=np.nan)
        df['ema1'] = self._ema_swma(df['close'], length=self.length)
        df['ema2'] = ta.ema(df['close'], length=self.length, fillna=np.nan)

        # Determine trend (ema1 < ema2 means uptrend in the script's logic)
        # Handle potential NaNs in EMAs during startup
        df['trend_up'] = np.where(df['ema1'] < df['ema2'], True,
                         np.where(df['ema1'] >= df['ema2'], False, np.nan)) # Keep NaN if EMAs are NaN
        df['trend_up'] = df['trend_up'].ffill() # Forward fill trend after first calculation

        # Detect trend change, ignoring NaNs and initial state
        df['trend_changed'] = (df['trend_up'] != df['trend_up'].shift(1)) & df['trend_up'].notna() & df['trend_up'].shift(1).notna()

        # --- Update Levels on Trend Change ---
        # Check the *last* bar for a trend change to update state variables
        last_row = df.iloc[-1]
        current_trend_up = last_row['trend_up']
        trend_just_changed = last_row['trend_changed']

        # Update persistent trend state
        if pd.notna(current_trend_up):
            if self.current_trend is None: # First valid trend
                self.current_trend = current_trend_up
                log.info(f"Initial Trend detected: {'UP' if self.current_trend else 'DOWN'}")
            elif trend_just_changed:
                self.current_trend = current_trend_up # Update trend state
                log.info(f"{Fore.MAGENTA}Trend Changed! New Trend: {'UP' if current_trend_up else 'DOWN'}. Updating levels...{Style.RESET_ALL}")

                # Find the row where the trend actually changed to get levels
                change_idx = df[df['trend_changed']].index[-1] if trend_just_changed else None
                if change_idx is not None:
                    change_row = df.loc[change_idx]
                    current_ema1 = change_row['ema1']
                    current_atr = change_row['atr']

                    if pd.notna(current_ema1) and pd.notna(current_atr) and current_atr > 0:
                        self.upper = current_ema1 + current_atr * 3
                        self.lower = current_ema1 - current_atr * 3
                        self.lower_vol = self.lower + current_atr * 4
                        self.upper_vol = self.upper - current_atr * 4
                        # Prevent levels from crossing due to large ATR
                        if self.lower_vol < self.lower: self.lower_vol = self.lower
                        if self.upper_vol > self.upper: self.upper_vol = self.upper

                        self.step_up = (self.lower_vol - self.lower) / 100 if self.lower_vol > self.lower else 0
                        self.step_dn = (self.upper - self.upper_vol) / 100 if self.upper > self.upper_vol else 0

                        log.info(f"Levels Updated @ {change_idx}: Upper={self.upper:.4f}, Lower={self.lower:.4f}, UpVol={self.upper_vol:.4f}, LowVol={self.lower_vol:.4f}")

                    else:
                         # Reset if data is NaN or ATR is zero
                         log.warning(f"Could not update levels at {change_idx} due to NaN/zero values (EMA1={current_ema1}, ATR={current_atr}).")
                         self.upper, self.lower, self.lower_vol, self.upper_vol, self.step_up, self.step_dn = [None] * 6
            # else: Trend did not change, levels remain the same

        # --- Volume Normalization ---
        # Calculate rolling percentile robustly
        df['vol_percentile_val'] = df['volume'].rolling(
            window=self.vol_percentile_len,
            min_periods=min(self.vol_percentile_len // 2, 50) # Require reasonable number of periods
        ).apply(
            lambda x: np.percentile(x[x.notna() & (x > 0)], self.vol_percentile) if np.any(x.notna() & (x > 0)) else np.nan,
            raw=True
        )

        # Handle cases where percentile might be 0 or NaN
        df['vol_norm'] = np.where(
            (df['vol_percentile_val'].notna()) & (df['vol_percentile_val'] > 1e-9), # Avoid division by near-zero
            (df['volume'] / df['vol_percentile_val'] * 100),
            0 # Assign 0 if percentile is NaN or (near) zero
        ).fillna(0).astype(float) # Use float for calculations

        # Calculate volume levels for plotting/potential future use
        # Ensure levels and steps are valid numbers
        if self.lower is not None and self.step_up is not None:
            df['vol_up_level'] = self.lower + self.step_up * df['vol_norm']
        else: df['vol_up_level'] = np.nan

        if self.upper is not None and self.step_dn is not None:
            df['vol_dn_level'] = self.upper - self.step_dn * df['vol_norm']
        else: df['vol_dn_level'] = np.nan


        # --- Pivot Order Block Calculations ---
        df['ph'] = self._find_pivots(df, self.pivot_left_h, self.pivot_right_h, is_high=True)
        df['pl'] = self._find_pivots(df, self.pivot_left_l, self.pivot_right_l, is_high=False)

        # --- Create and Manage Order Blocks ---
        # Iterate only through bars where new pivots *could* have been confirmed
        # A pivot at index `i` is confirmed `right` bars later.
        # We only need to check bars from `max(right_h, right_l)` ago up to now.
        check_start_idx = max(0, len(df) - max(self.pivot_right_h, self.pivot_right_l) - 5) # Add buffer
        new_boxes_created = False

        for i in range(check_start_idx, len(df)):
            # Use integer index for list management, but store meaningful df index if needed
            current_df_index = df.index[i] # DatetimeIndex

            # Bearish Box from Pivot High confirmed at bar `i`
            # The actual pivot occurred at `i - self.pivot_right_h`
            pivot_confirm_bar_idx = i
            pivot_occur_bar_idx = pivot_confirm_bar_idx - self.pivot_right_h
            if pivot_occur_bar_idx < 0: continue # Ensure pivot index is valid

            # Check if a PH value exists at the occurrence index
            if pd.notna(df['ph'].iloc[pivot_occur_bar_idx]):
                # Check if a box for this pivot *occurrence* already exists
                if not any(b['id'] == pivot_occur_bar_idx for b in self.bear_boxes):
                    top_price, bottom_price = np.nan, np.nan
                    ob_candle = df.iloc[pivot_occur_bar_idx] # Candle where OB is defined

                    if self.ob_source == "Wicks":
                        top_price = ob_candle['high']
                        # Common definition: Wick OB uses close of the defining candle
                        bottom_price = ob_candle['close']
                    else: # Bodys
                        top_price = ob_candle['close']
                        bottom_price = ob_candle['open']

                    # Ensure valid prices and top > bottom
                    if pd.notna(top_price) and pd.notna(bottom_price):
                        if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price # Swap if needed

                        # Check for zero-range OB (can happen with dojis etc.)
                        if abs(top_price - bottom_price) < 1e-9:
                            log.debug(f"Skipping zero-range Bearish OB at index {pivot_occur_bar_idx}")
                            continue

                        new_box = OrderBlock(
                            id=pivot_occur_bar_idx, type='bear', left_idx=pivot_occur_bar_idx,
                            right_idx=len(df)-1, # Extend to current bar initially
                            top=top_price, bottom=bottom_price, active=True, closed_idx=None
                        )
                        self.bear_boxes.append(new_box)
                        new_boxes_created = True
                        log.info(f"{Fore.RED}New Bearish OB created (ID {pivot_occur_bar_idx}): Top={top_price:.4f}, Bottom={bottom_price:.4f}{Style.RESET_ALL}")

            # Bullish Box from Pivot Low confirmed at bar `i`
            # The actual pivot occurred at `i - self.pivot_right_l`
            pivot_occur_bar_idx = pivot_confirm_bar_idx - self.pivot_right_l
            if pivot_occur_bar_idx < 0: continue

            if pd.notna(df['pl'].iloc[pivot_occur_bar_idx]):
                if not any(b['id'] == pivot_occur_bar_idx for b in self.bull_boxes):
                    top_price, bottom_price = np.nan, np.nan
                    ob_candle = df.iloc[pivot_occur_bar_idx]

                    if self.ob_source == "Wicks":
                         # Common definition: Wick OB uses open of the defining candle
                         top_price = ob_candle['open']
                         bottom_price = ob_candle['low']
                    else: # Bodys
                         top_price = ob_candle['open']
                         bottom_price = ob_candle['close']

                    if pd.notna(top_price) and pd.notna(bottom_price):
                        if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price

                        if abs(top_price - bottom_price) < 1e-9:
                            log.debug(f"Skipping zero-range Bullish OB at index {pivot_occur_bar_idx}")
                            continue

                        new_box = OrderBlock(
                            id=pivot_occur_bar_idx, type='bull', left_idx=pivot_occur_bar_idx,
                            right_idx=len(df)-1,
                            top=top_price, bottom=bottom_price, active=True, closed_idx=None
                        )
                        self.bull_boxes.append(new_box)
                        new_boxes_created = True
                        log.info(f"{Fore.GREEN}New Bullish OB created (ID {pivot_occur_bar_idx}): Top={top_price:.4f}, Bottom={bottom_price:.4f}{Style.RESET_ALL}")

        # --- Manage existing boxes (close or extend) ---
        current_close = last_row['close']
        current_bar_idx = len(df) - 1 # Use simple integer index for management

        if pd.notna(current_close): # Only manage boxes if close price is valid
            for box in self.bull_boxes:
                if box['active']:
                    if current_close < box['bottom']: # Price closed below bull box
                        box['active'] = False
                        box['closed_idx'] = current_bar_idx
                        log.info(f"{Fore.YELLOW}Bullish OB {box['id']} closed at {current_close:.4f} (below {box['bottom']:.4f}){Style.RESET_ALL}")
                    else:
                        box['right_idx'] = current_bar_idx # Extend active box

            for box in self.bear_boxes:
                if box['active']:
                    if current_close > box['top']: # Price closed above bear box
                        box['active'] = False
                        box['closed_idx'] = current_bar_idx
                        log.info(f"{Fore.YELLOW}Bearish OB {box['id']} closed at {current_close:.4f} (above {box['top']:.4f}){Style.RESET_ALL}")
                    else:
                        box['right_idx'] = current_bar_idx

        # --- Prune Order Blocks ---
        # Keep only the 'max_boxes' most recent *active* boxes of each type
        # Also keep some recent inactive ones for potential debugging/display
        active_bull = sorted([b for b in self.bull_boxes if b['active']], key=lambda x: x['id'], reverse=True)
        inactive_bull = sorted([b for b in self.bull_boxes if not b['active']], key=lambda x: x['id'], reverse=True)
        self.bull_boxes = inactive_bull[:self.max_boxes * 2] + active_bull[:self.max_boxes]

        active_bear = sorted([b for b in self.bear_boxes if b['active']], key=lambda x: x['id'], reverse=True)
        inactive_bear = sorted([b for b in self.bear_boxes if not b['active']], key=lambda x: x['id'], reverse=True)
        self.bear_boxes = inactive_bear[:self.max_boxes * 2] + active_bear[:self.max_boxes]

        # --- Signal Generation ---
        signal = "HOLD"
        active_bull_boxes = [b for b in self.bull_boxes if b['active']]
        active_bear_boxes = [b for b in self.bear_boxes if b['active']]

        # Check conditions only if trend and close price are valid
        if self.current_trend is not None and pd.notna(current_close):
            # Check for Trend Change Exit first
            if trend_just_changed:
                # Check last_signal to see if we were in a position that needs exiting
                if not self.current_trend and self.last_signal == "BUY": # Trend flipped down while long
                    signal = "EXIT_LONG"
                    log.warning(f"{Fore.YELLOW}{Style.BRIGHT}*** EXIT LONG Signal (Trend Flip to DOWN) at {current_close:.4f} ***{Style.RESET_ALL}")
                elif self.current_trend and self.last_signal == "SELL": # Trend flipped up while short
                    signal = "EXIT_SHORT"
                    log.warning(f"{Fore.YELLOW}{Style.BRIGHT}*** EXIT SHORT Signal (Trend Flip to UP) at {current_close:.4f} ***{Style.RESET_ALL}")

            # Check for Entries only if not exiting and not already in desired position
            if signal == "HOLD":
                if self.current_trend: # Trend is UP, look for Long Entries
                    if self.last_signal != "BUY": # Only enter if not already long
                        for box in active_bull_boxes:
                            # Check if close touches or is within the box range
                            if box['bottom'] <= current_close <= box['top']:
                                signal = "BUY"
                                log.info(f"{Fore.GREEN}{Style.BRIGHT}*** BUY Signal (Trend UP + Price in Bull OB {box['id']}) at {current_close:.4f} ***{Style.RESET_ALL}")
                                break # Take first signal

                elif not self.current_trend: # Trend is DOWN, look for Short Entries
                    if self.last_signal != "SELL": # Only enter if not already short
                        for box in active_bear_boxes:
                            if box['bottom'] <= current_close <= box['top']:
                                signal = "SELL"
                                log.info(f"{Fore.RED}{Style.BRIGHT}*** SELL Signal (Trend DOWN + Price in Bear OB {box['id']}) at {current_close:.4f} ***{Style.RESET_ALL}")
                                break

        # Update last signal state *only* if a new trade signal or exit occurred
        # Keep 'BUY' or 'SELL' state until an exit signal or opposite entry occurs
        if signal in ["BUY", "SELL", "EXIT_LONG", "EXIT_SHORT"]:
            # If exiting, the state after exit is HOLD until a new entry
             self.last_signal = signal if signal in ["BUY", "SELL"] else "HOLD"


        return AnalysisResults(
            dataframe=df, # Return the DataFrame with calculated indicators
            last_signal=signal,
            active_bull_boxes=active_bull_boxes,
            active_bear_boxes=active_bear_boxes,
            last_close=current_close,
            current_trend=self.current_trend,
            trend_changed=trend_just_changed
        )

# --- Data Fetching ---
def fetch_bybit_data(symbol: str, interval: str, limit: int = FETCH_LIMIT) -> Optional[pd.DataFrame]:
    """Fetches historical Klines from Bybit and returns a pandas DataFrame."""
    log.info(f"{Fore.CYAN}Summoning historical data for {symbol} ({interval}, {limit} candles)...{Style.RESET_ALL}")
    url = f"{BYBIT_API_BASE}/v5/market/kline"
    params = { "category": "linear", "symbol": symbol, "interval": interval, "limit": limit }
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data.get('retCode') != 0 or not data.get('result') or not data['result'].get('list'):
            log.error(f"Error fetching data from Bybit API: {data.get('retMsg', 'Invalid response structure')}")
            return None

        kline_list: List[List[str]] = data['result']['list']
        if not kline_list:
            log.warning("Received empty kline list from Bybit.")
            return pd.DataFrame() # Return empty DataFrame instead of None

        # Create DataFrame
        df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        # Bybit returns data newest first, reverse it for chronological order
        df = df.iloc[::-1]
        df = df.astype({
            'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
            'low': 'float64', 'close': 'float64', 'volume': 'float64', 'turnover': 'float64'
        })
        # Convert timestamp to datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        # df = df.sort_index() # Already sorted by reversing the list

        log.info(f"{Fore.GREEN}Historical data summoned ({len(df)} candles). From {df.index.min()} to {df.index.max()}{Style.RESET_ALL}")
        return df

    except requests.exceptions.Timeout:
        log.error(f"{Style.BRIGHT}{Fore.RED}Network Error: Request timed out fetching data from Bybit.{Style.RESET_ALL}")
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"{Style.BRIGHT}{Fore.RED}Network Error fetching data: {e}{Style.RESET_ALL}")
        return None
    except Exception as e:
        log.error(f"{Style.BRIGHT}{Fore.RED}An unexpected error occurred during data fetch: {e}{Style.RESET_ALL}")
        return None

# --- WebSocket Handling ---
ws_app: Optional[websocket.WebSocketApp] = None
ws_thread: Optional[threading.Thread] = None
current_symbol_interval: Dict[str, str] = {"symbol": "", "interval": ""}
latest_dataframe: Optional[pd.DataFrame] = None # Store the DataFrame
strategy_instance: Optional[VolumaticOBStrategy] = None # Hold the strategy object
data_lock = threading.Lock() # Lock for thread-safe DataFrame updates
ws_connected = threading.Event() # Event to signal successful WS connection
stop_ws_flag = threading.Event() # Event to signal WS loop to stop

def process_kline_update(kline_data: Dict):
    """Processes a single kline update and updates the DataFrame."""
    global latest_dataframe, strategy_instance
    try:
        ts_ms = int(kline_data['start']) # Bybit v5 uses 'start' for timestamp
        ts = pd.to_datetime(ts_ms, unit='ms')
        k_open = float(kline_data['open'])
        k_high = float(kline_data['high'])
        k_low = float(kline_data['low'])
        k_close = float(kline_data['close'])
        k_volume = float(kline_data['volume'])
        # Turnover might not always be present depending on subscription type
        k_turnover = float(kline_data.get('turnover', 0.0))
        is_final = kline_data.get('confirm', False) # True if candle is closed

        # Use lock for thread-safe access to shared DataFrame
        with data_lock:
            if latest_dataframe is None:
                log.warning("DataFrame not initialized, cannot process WS update.")
                return

            new_row_data = {
                'open': k_open, 'high': k_high, 'low': k_low, 'close': k_close,
                'volume': k_volume, 'turnover': k_turnover
            }

            # Check if this timestamp already exists (update) or is new
            if ts in latest_dataframe.index:
                # Update the existing row - overwrite OHLCV
                # Note: Volume should ideally be accumulated if not final, but Bybit WS usually sends total vol for the candle
                latest_dataframe.loc[ts, list(new_row_data.keys())] = list(new_row_data.values())
                action = "Updated"
            else:
                # Append new row (new candle)
                new_row = pd.DataFrame([new_row_data], index=[ts])
                latest_dataframe = pd.concat([latest_dataframe, new_row])
                action = "Appended"
                # Drop oldest row if exceeding max length
                if len(latest_dataframe) > MAX_DF_LEN:
                    latest_dataframe = latest_dataframe.iloc[-(MAX_DF_LEN):]
                    log.debug(f"DataFrame pruned to {MAX_DF_LEN} rows.")

            # --- Re-run analysis ONLY on confirmed candle close ---
            # Or run on every update if needed, but might be noisy/CPU intensive
            if is_final and strategy_instance:
                log.debug(f"Running analysis on confirmed candle: {ts}")
                # Analyze a copy to prevent issues if analysis takes time
                analysis_results = strategy_instance.update(latest_dataframe.copy())
                # Display results or act on signal
                display_analysis_results(analysis_results, is_live_update=True)
                # Check for trade execution based on signal
                handle_signal(analysis_results['last_signal'], analysis_results['last_close'])
            elif not is_final:
                # Optionally log interim updates
                 log.debug(f"WS [{current_symbol_interval['symbol']}/{current_symbol_interval['interval']}] Interim Update: C={k_close:.4f}")
                 pass


        # Log outside the lock
        # status_color = Fore.GREEN if is_final else Fore.YELLOW
        # log.debug(status_color + f"WS [{current_symbol_interval['symbol']}/{current_symbol_interval['interval']}] {action}: " +
        #           f"T={ts.strftime('%H:%M:%S')} O={k_open:.4f} H={k_high:.4f} L={k_low:.4f} C={k_close:.4f} V={k_volume:.2f} Final={is_final}")

    except KeyError as e:
        log.error(f"Error processing kline update: Missing key {e} in data: {kline_data}")
    except ValueError as e:
        log.error(f"Error processing kline update: Could not convert value: {e} in data: {kline_data}")
    except Exception as e:
        log.error(f"Unexpected error processing kline update: {e}", exc_info=True)

def on_ws_message(ws, message):
    """Handles incoming WebSocket messages."""
    #log.debug(f"WS Recv: {message}")
    try:
        data = json.loads(message)

        # Handle subscription confirmation
        if data.get('op') == 'subscribe':
            if data.get('success', False):
                log.info(f"{Fore.GREEN}WebSocket subscribed successfully to {data.get('ret_msg')}{Style.RESET_ALL}")
                ws_connected.set() # Signal successful connection
            else:
                log.error(f"WebSocket subscription failed: {data.get('ret_msg')}")
                ws_connected.set() # Set anyway to allow shutdown attempt
            return

        # Handle ping/pong
        if data.get('op') == 'pong':
            log.debug("WebSocket Pong received")
            return

        # Handle Kline data
        if data.get('topic', '').startswith(f"kline.{current_symbol_interval['interval']}.{current_symbol_interval['symbol']}"):
             if data.get('data') and isinstance(data['data'], list):
                 for kline_item in data['data']:
                     process_kline_update(kline_item)
             else:
                 log.warning(f"Received kline message with unexpected data format: {data}")

    except json.JSONDecodeError:
        log.warning(f"Received non-JSON WebSocket message: {message[:100]}...") # Log beginning
    except Exception as e:
        log.error(f"Error in on_ws_message: {e}", exc_info=True)

def on_ws_error(ws, error):
    """Handles WebSocket errors."""
    log.error(f"{Fore.RED}{Style.BRIGHT}WebSocket Error: {error}{Style.RESET_ALL}")
    ws_connected.set() # Signal connection attempt finished (even if error)

def on_ws_close(ws, close_status_code, close_msg):
    """Handles WebSocket connection close."""
    log.info(f"{Fore.YELLOW}WebSocket connection closed. Code: {close_status_code}, Msg: {close_msg}{Style.RESET_ALL}")
    ws_connected.set() # Signal connection is closed

def on_ws_open(ws):
    """Handles WebSocket connection open."""
    log.info(f"{Fore.CYAN}WebSocket connection opened. Subscribing...{Style.RESET_ALL}")
    topic = f"kline.{current_symbol_interval['interval']}.{current_symbol_interval['symbol']}"
    ws.send(json.dumps({
        "op": "subscribe",
        "args": [topic]
    }))
    # Start periodic ping sender
    def ping_sender():
        while not stop_ws_flag.is_set():
            try:
                if ws.sock and ws.sock.connected:
                     ws.send(json.dumps({"op": "ping"}))
                     log.debug("WebSocket Ping sent")
                else:
                    log.warning("WebSocket ping skipped: Socket not connected.")
                    break # Stop pinging if disconnected
            except Exception as e:
                log.error(f"Error sending WebSocket ping: {e}")
                break # Stop pinging on error
            time.sleep(WS_PING_INTERVAL)
        log.debug("WebSocket ping sender stopped.")

    ping_thread = threading.Thread(target=ping_sender, daemon=True)
    ping_thread.start()


def run_websocket():
    """Runs the WebSocket connection loop."""
    global ws_app
    log.info(f"Starting WebSocket connection to {BYBIT_WS_PUBLIC}")
    stop_ws_flag.clear()
    ws_connected.clear()
    ws_app = websocket.WebSocketApp(BYBIT_WS_PUBLIC,
                                  on_open=on_ws_open,
                                  on_message=on_ws_message,
                                  on_error=on_ws_error,
                                  on_close=on_ws_close)
    ws_app.run_forever()
    # run_forever blocks until connection closes
    log.info("WebSocket run_forever loop finished.")
    ws_app = None # Clear ws_app instance after loop ends

def start_websocket(symbol: str, interval: str):
    """Starts the WebSocket connection in a separate thread."""
    global ws_thread, current_symbol_interval
    if ws_thread and ws_thread.is_alive():
        log.warning("WebSocket thread already running.")
        return

    current_symbol_interval["symbol"] = symbol
    current_symbol_interval["interval"] = interval

    ws_thread = threading.Thread(target=run_websocket, daemon=True)
    ws_thread.start()
    log.info("Waiting for WebSocket connection and subscription...")
    # Wait for connection/subscription confirmation or error/close
    connected = ws_connected.wait(timeout=20) # Wait up to 20 seconds
    if connected and ws_app and ws_app.sock and ws_app.sock.connected:
         log.info(f"{Fore.GREEN}WebSocket appears connected and subscribed.{Style.RESET_ALL}")
         return True
    else:
         log.error(f"{Fore.RED}WebSocket failed to connect or subscribe within timeout.{Style.RESET_ALL}")
         stop_websocket() # Ensure cleanup if connection failed
         return False


def stop_websocket():
    """Stops the WebSocket connection."""
    global ws_thread, ws_app
    if not ws_thread or not ws_thread.is_alive():
        log.info("WebSocket thread is not running.")
        return

    log.info("Stopping WebSocket connection...")
    stop_ws_flag.set() # Signal ping loop to stop
    if ws_app:
        ws_app.close()

    ws_thread.join(timeout=10) # Wait for thread to finish
    if ws_thread.is_alive():
        log.warning("WebSocket thread did not stop gracefully.")
    else:
        log.info("WebSocket thread stopped.")

    ws_thread = None
    ws_app = None
    ws_connected.clear()


# --- Order Placement ---
def place_order_via_backend(symbol: str, side: str, order_type: str, qty: float) -> Optional[Dict]:
    """Sends an order request to the backend server."""
    log.info(f"{Fore.CYAN}Sending {side} {order_type} request for {qty:.5f} {symbol} to Backend Bastion...{Style.RESET_ALL}") # More qty precision
    url = f"{BACKEND_URL}/api/place-order"
    payload = {
        "symbol": symbol,
        "side": side,         # "Buy" or "Sell"
        "orderType": order_type, # "Market" or "Limit"
        "qty": str(qty)       # Bybit API often prefers quantity as string
    }
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response_data = response.json()

        if response.status_code == 200 and response_data.get("success"):
            log.info(f"{Fore.GREEN}{Style.BRIGHT}Backend Confirmation:{Style.RESET_ALL}")
            log.info(f"{Fore.GREEN}  Message: {response_data.get('message')}{Style.RESET_ALL}")
            log.info(f"{Fore.GREEN}  OrderID: {response_data.get('orderId')}{Style.RESET_ALL}")
            return response_data
        else:
            log.error(f"{Fore.RED}{Style.BRIGHT}Backend Rejection:{Style.RESET_ALL}")
            log.error(f"{Fore.RED}  Status Code: {response.status_code}{Style.RESET_ALL}")
            log.error(f"{Fore.RED}  Error: {response_data.get('error', 'Unknown')}{Style.RESET_ALL}")
            log.error(f"{Fore.RED}  Details: {response_data.get('details', 'N/A')}{Style.RESET_ALL}")
            return response_data # Return error details

    except requests.exceptions.ConnectionError:
         log.critical(f"{Fore.RED}{Style.BRIGHT}FATAL: Could not connect to Backend Bastion at {url}. Is it running?{Style.RESET_ALL}")
         return None
    except requests.exceptions.Timeout:
         log.error(f"{Fore.RED}{Style.BRIGHT}Error: Request to Backend Bastion timed out.{Style.RESET_ALL}")
         return None
    except Exception as e:
        log.error(f"{Fore.RED}{Style.BRIGHT}Error communicating with Backend Bastion: {e}{Style.RESET_ALL}")
        return None

# --- Display Helper ---
def display_analysis_results(results: Optional[AnalysisResults], is_live_update=False):
    """Prints the analysis results in a formatted way."""
    if not results or results['dataframe'] is None or results['dataframe'].empty:
        log.warning("No analysis results to display.")
        return

    df = results['dataframe']
    if df.empty:
        log.warning("Analysis results contain an empty DataFrame.")
        return
    last_row = df.iloc[-1]

    # Use logging for output consistency
    if not is_live_update: # Print full summary only on fetch/analyze command
        log.info(f"{Fore.MAGENTA}\n--- Strategy Analysis ({df.index[-1]}) ---{Style.RESET_ALL}")
        print(f"{'Metric':<18} {'Value'}")
        print(f"{'-'*18} {'-'*25}")

        trend_val = results['current_trend']
        trend_str = (f"{Fore.GREEN}UP{Style.RESET_ALL}" if trend_val else
                     f"{Fore.RED}DOWN{Style.RESET_ALL}" if trend_val is False else
                     f"{Fore.WHITE}N/A{Style.RESET_ALL}")
        print(f"{'Current Trend':<18} {trend_str}")
        trend_changed = results['trend_changed']
        print(f"{'Trend Changed':<18} {Fore.YELLOW + str(trend_changed) + Style.RESET_ALL if trend_changed else str(trend_changed)}")
        print(f"{'Last Close':<18} {Fore.YELLOW}{results['last_close']:,.4f}{Style.RESET_ALL}")
        print(f"{'EMA 1 (Smooth)':<18} {Fore.CYAN}{last_row.get('ema1', np.nan):,.4f}{Style.RESET_ALL}")
        print(f"{'EMA 2 (Regular)':<18} {Fore.CYAN}{last_row.get('ema2', np.nan):,.4f}{Style.RESET_ALL}")
        print(f"{'ATR':<18} {Fore.WHITE}{last_row.get('atr', np.nan):,.4f}{Style.RESET_ALL}")
        print(f"{'Vol Norm (%)':<18} {Fore.WHITE}{last_row.get('vol_norm', np.nan):,.1f}%{Style.RESET_ALL}")

        print(f"{Fore.BLUE}\nActive Order Blocks:{Style.RESET_ALL}")
        if not results['active_bull_boxes'] and not results['active_bear_boxes']:
             print(f"{Fore.WHITE}{Style.DIM}  None{Style.RESET_ALL}")
        # Display most recent active first
        for box in sorted(results['active_bull_boxes'], key=lambda x: x['id'], reverse=True):
            print(f"{Fore.GREEN}  Bull OB {box['id']}: Top={box['top']:,.4f}, Bottom={box['bottom']:,.4f}{Style.RESET_ALL}")
        for box in sorted(results['active_bear_boxes'], key=lambda x: x['id'], reverse=True):
            print(f"{Fore.RED}  Bear OB {box['id']}: Top={box['top']:,.4f}, Bottom={box['bottom']:,.4f}{Style.RESET_ALL}")

    # Always print the latest signal if it's not HOLD
    signal = results['last_signal']
    if signal != "HOLD" or not is_live_update: # Show HOLD on manual analysis, only show trades/exits on live
        sig_color = Fore.WHITE
        sig_level = logging.INFO
        if signal == "BUY": sig_color = Fore.GREEN + Style.BRIGHT; sig_level = logging.WARNING
        elif signal == "SELL": sig_color = Fore.RED + Style.BRIGHT; sig_level = logging.WARNING
        elif "EXIT" in signal: sig_color = Fore.YELLOW + Style.BRIGHT; sig_level = logging.WARNING
        log.log(sig_level, f"{'Latest Signal':<18} {sig_color}{signal}{Style.RESET_ALL}")
        print("-" * 44)


# --- Signal Handling ---
# Basic state machine for position tracking (in-memory, resets on script restart)
# TODO: Persist position state or query actual position from exchange via backend
current_position = None # None, 'LONG', 'SHORT'

def handle_signal(signal: str, price: float):
    """Acts on the generated signal by placing orders via the backend."""
    global current_position
    # Use configured order quantity
    order_qty = DEFAULT_ORDER_QTY
    symbol = current_symbol_interval['symbol']

    if pd.isna(price):
        log.warning(f"Signal '{signal}' received but price is NaN. No action taken.")
        return

    # --- Long Entry ---
    if signal == "BUY" and current_position != 'LONG':
        log.warning(f"{Fore.GREEN}{Style.BRIGHT}ACTION: Strategy signals BUY for {symbol}. Attempting Market Buy {order_qty} at ~{price:.4f}{Style.RESET_ALL}")
        # Close short position first if exists (standard practice, no hedging assumed)
        if current_position == 'SHORT':
             log.info(f"{Fore.YELLOW}Closing existing SHORT position before entering LONG...{Style.RESET_ALL}")
             # Closing short requires a BUY order
             close_result = place_order_via_backend(symbol, "Buy", "Market", order_qty)
             if not close_result or not close_result.get('success'):
                 log.error("Failed to close existing SHORT position. Cannot enter LONG.")
                 return # Don't proceed with buy if close failed
             current_position = None # Assume closed successfully for now
             time.sleep(1) # Small delay before placing new order

        # Place Buy order
        result = place_order_via_backend(symbol, "Buy", "Market", order_qty)
        if result and result.get('success'):
            current_position = 'LONG'
            log.info(f"{Fore.GREEN}Successfully entered LONG position.{Style.RESET_ALL}")
        else:
            log.error("Failed to execute BUY order.")

    # --- Short Entry ---
    elif signal == "SELL" and current_position != 'SHORT':
        log.warning(f"{Fore.RED}{Style.BRIGHT}ACTION: Strategy signals SELL for {symbol}. Attempting Market Sell {order_qty} at ~{price:.4f}{Style.RESET_ALL}")
        # Close long position first if exists
        if current_position == 'LONG':
            log.info(f"{Fore.YELLOW}Closing existing LONG position before entering SHORT...{Style.RESET_ALL}")
            # Closing long requires a SELL order
            close_result = place_order_via_backend(symbol, "Sell", "Market", order_qty)
            if not close_result or not close_result.get('success'):
                 log.error("Failed to close existing LONG position. Cannot enter SHORT.")
                 return
            current_position = None
            time.sleep(1)

        # Place Sell order
        result = place_order_via_backend(symbol, "Sell", "Market", order_qty)
        if result and result.get('success'):
            current_position = 'SHORT'
            log.info(f"{Fore.RED}Successfully entered SHORT position.{Style.RESET_ALL}")
        else:
            log.error("Failed to execute SELL order.")

    # --- Exit Long ---
    elif signal == "EXIT_LONG" and current_position == 'LONG':
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}ACTION: Strategy signals EXIT LONG. Attempting Market Sell {order_qty} at ~{price:.4f}{Style.RESET_ALL}")
        result = place_order_via_backend(symbol, "Sell", "Market", order_qty)
        if result and result.get('success'):
            current_position = None
            log.info(f"{Fore.YELLOW}Successfully exited LONG position.{Style.RESET_ALL}")
        else:
            log.error("Failed to execute EXIT LONG (Sell) order.")

    # --- Exit Short ---
    elif signal == "EXIT_SHORT" and current_position == 'SHORT':
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}ACTION: Strategy signals EXIT SHORT. Attempting Market Buy {order_qty} at ~{price:.4f}{Style.RESET_ALL}")
        result = place_order_via_backend(symbol, "Buy", "Market", order_qty)
        if result and result.get('success'):
            current_position = None
            log.info(f"{Fore.YELLOW}Successfully exited SHORT position.{Style.RESET_ALL}")
        else:
            log.error("Failed to execute EXIT SHORT (Buy) order.")

    # Ignore HOLD signals or signals that don't change the current state
    # elif signal == "HOLD": pass
    # elif signal == "BUY" and current_position == 'LONG': pass # Already long
    # elif signal == "SELL" and current_position == 'SHORT': pass # Already short


# --- Main Execution Charm ---
if __name__ == "__main__":
    # Make global variables accessible if needed, though better to pass explicitly
    # global latest_dataframe, strategy_instance

    # --- Strategy Initialization ---
    # Consider loading params from a config file or environment variables
    try:
        strategy_instance = VolumaticOBStrategy(
            length=40,          # Default Volumatic Trend Length
            vol_atr_period=200, # Default ATR period for Vol levels
            pivot_left_h=10,    # Shorter pivot lookback for potentially more OBs
            pivot_right_h=10,
            pivot_left_l=10,
            pivot_right_l=10,
            ob_source="Wicks",  # Use "Wicks" or "Bodys"
            max_boxes=5         # Limit active OBs considered
        )
    except ValueError as e:
        log.critical(f"Failed to initialize strategy: {e}. Exiting.")
        exit(1)

    current_symbol = "BTCUSDT"
    # Common intervals: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
    current_interval = "5" # Default 5m
    analysis_results: Optional[AnalysisResults] = None
    is_live = False

    print(Fore.MAGENTA + Style.BRIGHT + "\n~~~ Pyrmethus Volumatic+OB Strategy CLI ~~~")
    print(Fore.YELLOW + "Ensure backend_server.py is running and accessible at " + BACKEND_URL)
    print(Fore.CYAN + f"Default Context: {current_symbol} / {current_interval}m")

    while True:
        try:
            live_status = Fore.GREEN + "(Live)" if is_live else Fore.RED + "(Offline)"
            pos_status = f"Pos: {Fore.CYAN}{current_position or 'None'}{Style.RESET_ALL}"
            prompt = (f"\n{Fore.BLUE}[{current_symbol}/{current_interval}] {live_status} {pos_status} "
                      f"| Cmd (fetch, analyze, live, stop, set, pos, exit): {Style.RESET_ALL}")
            command = input(prompt).lower().strip()

            if command == "exit":
                if is_live: stop_websocket()
                log.info(f"{Fore.MAGENTA}Pyrmethus bids you farewell!{Style.RESET_ALL}")
                break
            elif command == "fetch":
                if is_live:
                    log.warning("Cannot fetch manually while live feed is active. Use 'stop' first.")
                    continue
                with data_lock: # Protect access while fetching
                    latest_dataframe = fetch_bybit_data(current_symbol, current_interval)
                    if latest_dataframe is not None and not latest_dataframe.empty and strategy_instance:
                        analysis_results = strategy_instance.update(latest_dataframe.copy()) # Analyze a copy
                        display_analysis_results(analysis_results)
                    elif latest_dataframe is not None and latest_dataframe.empty:
                        log.warning("Fetched data but the DataFrame is empty.")
                    else:
                        log.error("Failed to fetch or process data.")
            elif command == "analyze":
                 if analysis_results:
                     display_analysis_results(analysis_results)
                 else:
                     log.warning("No data fetched/analyzed yet. Use 'fetch' first.")
            elif command == "live":
                if not is_live:
                    log.info("Attempting to start live feed...")
                    # Fetch initial data first
                    with data_lock:
                        log.info("Fetching initial data before going live...")
                        latest_dataframe = fetch_bybit_data(current_symbol, current_interval)
                        if latest_dataframe is not None and not latest_dataframe.empty and strategy_instance:
                            log.info("Initial data fetched. Running analysis...")
                            analysis_results = strategy_instance.update(latest_dataframe.copy())
                            display_analysis_results(analysis_results)
                            log.info("Starting WebSocket...")
                            if start_websocket(current_symbol, current_interval):
                                is_live = True
                                log.info(f"{Fore.GREEN}Live feed started successfully for {current_symbol}/{current_interval}.{Style.RESET_ALL}")
                            else:
                                log.error("Failed to start WebSocket. Live feed inactive.")
                                # Ensure cleanup if start failed partially
                                stop_websocket()
                                is_live = False
                        elif latest_dataframe is not None and latest_dataframe.empty:
                             log.error("Fetched empty DataFrame for initial data. Cannot start live feed.")
                        else:
                             log.error("Failed to fetch initial data. Cannot start live feed.")
                else:
                    log.warning("Live feed is already active.")
            elif command == "stop":
                 if is_live:
                     stop_websocket()
                     is_live = False
                     log.info(f"{Fore.RED}Live feed stopped.{Style.RESET_ALL}")
                 else:
                     log.warning("Live feed is not active.")
            elif command == "set":
                 if is_live:
                     log.warning("Stop the live feed ('stop') before changing context.")
                     continue

                 log.info(f"Current context: {current_symbol}/{current_interval}")
                 new_symbol = input(f"{Fore.BLUE}Enter new symbol [{current_symbol}]: {Style.RESET_ALL}").strip().upper()
                 new_interval = input(f"{Fore.BLUE}Enter new interval (e.g., 1, 5, 15, 60) [{current_interval}]: {Style.RESET_ALL}").strip()

                 context_changed = False
                 if new_symbol and new_symbol != current_symbol:
                     current_symbol = new_symbol
                     context_changed = True
                 if new_interval and new_interval != current_interval:
                     # Basic validation, add more if needed
                     if not new_interval.isdigit() and new_interval not in ['D', 'W', 'M']:
                         log.error("Invalid interval format. Use numbers (minutes) or D/W/M.")
                         continue
                     current_interval = new_interval
                     context_changed = True

                 if context_changed:
                     log.info(f"{Fore.GREEN}Context set to: {current_symbol}/{current_interval}{Style.RESET_ALL}")
                     # Reset state and data
                     with data_lock:
                         latest_dataframe = None
                         analysis_results = None
                         current_position = None # Reset position on context change
                         if strategy_instance: # Reset strategy state
                             log.info("Resetting strategy state and order blocks...")
                             strategy_instance.bull_boxes = []
                             strategy_instance.bear_boxes = []
                             strategy_instance.last_signal = "HOLD"
                             strategy_instance.current_trend = None
                             # Reset levels as they depend on history
                             strategy_instance.upper, strategy_instance.lower = None, None
                             strategy_instance.lower_vol, strategy_instance.upper_vol = None, None
                             strategy_instance.step_up, strategy_instance.step_dn = None, None

                 else:
                    log.info("Context unchanged.")

            elif command == "pos": # Command to manually set position state
                if is_live:
                     log.warning("Cannot manually set position while live. Stop first or implement position query.")
                     continue
                new_pos = input(f"{Fore.BLUE}Set position state (LONG, SHORT, None) [{current_position}]: {Style.RESET_ALL}").upper().strip()
                if new_pos in ['LONG', 'SHORT']:
                    current_position = new_pos
                    log.info(f"Position state manually set to: {current_position}")
                elif new_pos == 'NONE' or not new_pos:
                    current_position = None
                    log.info("Position state manually set to: None")
                else:
                    log.error("Invalid position state. Use LONG, SHORT, or None.")

            else:
                log.warning("Unknown command. Available: fetch, analyze, live, stop, set, pos, exit")

        except KeyboardInterrupt:
            log.info("\nKeyboardInterrupt received.")
            if is_live:
                stop_websocket()
            log.info(f"{Fore.MAGENTA}Exiting Pyrmethus CLI.{Style.RESET_ALL}")
            break
        except EOFError: # Handle pipe closure or ctrl-d
             log.info("\nEOF received.")
             if is_live:
                 stop_websocket()
             log.info(f"{Fore.MAGENTA}Exiting Pyrmethus CLI.{Style.RESET_ALL}")
             break
        except Exception as e:
             log.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
             # Attempt graceful shutdown
             if is_live: stop_websocket()
             break # Exit on unhandled exception
