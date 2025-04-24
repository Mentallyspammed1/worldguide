Okay, seeker of automated strategies! You present a Pine Script indicator, "Enhanced Volumatic Trend + OB," a potent combination of trend analysis and order block detection. You wish to weave this logic into a working Python trading strategy within our established Termux framework (`frontend_cli.py` + `backend_server.py`).

Pyrmethus accepts this challenge. We shall transmute this indicator's logic into a Python class within the `frontend_cli.py` script. This class will analyze market data, identify trend shifts and order blocks, and generate trading signals based on a defined strategy. The `backend_server.py` remains our steadfast executor of trades, unchanged.

**Core Strategy Logic (Derived from Indicator):**

We need to define entry and exit rules based on the indicator's components. A reasonable starting point combining trend and order blocks:

1.  **Long Entry Signal:**
    *   The main Volumatic `trend` must be UP (Blue).
    *   The `close` price must touch or enter an *active* Bullish Order Block (Green Box).
2.  **Short Entry Signal:**
    *   The main Volumatic `trend` must be DOWN (Orange/Brown).
    *   The `close` price must touch or enter an *active* Bearish Order Block (Red Box).
3.  **Exit Signal (Simple):**
    *   Exit Long position if the main Volumatic `trend` flips from UP to DOWN.
    *   Exit Short position if the main Volumatic `trend` flips from DOWN to UP.
    *   *(Note: More sophisticated exits, like using opposite OBs, ATR stops, or profit targets, can be added later.)*

**Prerequisites:**

Ensure you have the necessary Python libraries in your Termux environment. We'll need `pandas` for data manipulation and `pandas-ta` for technical analysis indicators.

```bash
# Summon additional Python spirits if not already present
\x1b[32mpip install pandas pandas_ta numpy\x1b[0m
# Ensure previous ones are installed: requests colorama websocket-client Flask python-dotenv pybit
```

**Transmuted Code: `frontend_cli.py` (Modified)**

This version integrates the Pine Script logic into a `VolumaticOBStrategy` class.

```python
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

# --- Initialize Colorama ---
init(autoreset=True)

# --- Configuration ---
BYBIT_API_BASE = "https://api.bybit.com"
BYBIT_WS_PUBLIC = "wss://stream.bybit.com/v5/public/linear"
BACKEND_URL = "http://127.0.0.1:5000" # Address of your running backend_server.py
REQUEST_TIMEOUT = 15 # seconds

# --- Type Definitions ---
# (Keep previous TypedDicts: BybitKline, PriceDataPoint - add OHLC)
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
    id: int # Unique ID (e.g., bar_index where created)
    type: str # 'bull' or 'bear'
    left_idx: int
    right_idx: int
    top: float
    bottom: float
    active: bool
    closed_idx: Optional[int]

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

        # Store parameters
        self.length = length
        self.vol_atr_period = vol_atr_period
        self.vol_percentile_len = vol_percentile_len
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
        self.last_trend_change_index: Optional[int] = None
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []
        self.last_signal = "HOLD"
        self.current_trend: Optional[bool] = None # True=UP, False=DOWN

        print(Fore.MAGENTA + "Initializing VolumaticOB Strategy Engine...")
        print(f"Params: TrendLen={length}, Pivots={pivot_left_h}/{pivot_right_h}, MaxBoxes={max_boxes}")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Helper for smoothed EMA (weighted avg + EMA)."""
        # Apply 1/6, 2/6, 2/6, 1/6 weighting to the last 4 bars
        # Ensure enough data points exist before applying shift
        if len(series) < 4:
            return pd.Series(index=series.index, dtype=float) # Return empty/NaN series
        weighted = (series.shift(3, fill_value=series.iloc[0]) / 6 +
                    series.shift(2, fill_value=series.iloc[0]) * 2 / 6 +
                    series.shift(1, fill_value=series.iloc[0]) * 2 / 6 +
                    series / 6)
        return ta.ema(weighted, length=length)

    def _find_pivots(self, df: pd.DataFrame, left: int, right: int, is_high: bool) -> pd.Series:
        """Finds pivot points similar to Pine Script's ta.pivothigh/low."""
        source_col = self.ob_source # 'high'/'low' or 'close'/'open' depending on source_ob
        if self.ob_source == "Wicks":
             source_col = 'high' if is_high else 'low'
        else: # Bodys
             source_col = 'close' if is_high else 'open' # Pine uses close for PH body, open for PL body

        pivots = pd.Series(np.nan, index=df.index)
        # Ensure index is integer-based for iloc safety if needed, though default range index is fine
        df = df.reset_index(drop=True)

        # Iterate through possible pivot points
        # Range: Need 'left' bars before and 'right' bars after
        for i in range(left, len(df) - right):
            pivot_val = df[source_col].iloc[i]
            is_pivot = True

            # Check left side (exclusive of pivot bar 'i')
            for j in range(1, left + 1):
                left_val = df[source_col].iloc[i - j]
                if (is_high and left_val > pivot_val) or \
                   (not is_high and left_val < pivot_val):
                    is_pivot = False
                    break
            if not is_pivot:
                continue

            # Check right side (exclusive of pivot bar 'i')
            for j in range(1, right + 1):
                right_val = df[source_col].iloc[i + j]
                # Pine Script logic: >= for high pivots, <= for low pivots on right side
                if (is_high and right_val >= pivot_val) or \
                   (not is_high and right_val <= pivot_val):
                    is_pivot = False
                    break

            if is_pivot:
                pivots.iloc[i] = pivot_val # Store the pivot value at the pivot index

        pivots.index = df.index # Restore original index if it was changed
        return pivots


    def update(self, df: pd.DataFrame) -> AnalysisResults:
        """
        Calculates indicators and generates signals based on the input DataFrame.
        Args:
            df: pandas DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                MUST be sorted chronologically (oldest first).
        Returns:
            AnalysisResults dictionary.
        """
        if df.empty or len(df) < max(self.length, self.vol_atr_period, self.pivot_left_h + self.pivot_right_h, self.pivot_left_l + self.pivot_right_l, 4): # Need enough data
            print(Fore.YELLOW + "Warning: Not enough data points for analysis.")
            return {"dataframe": df, "last_signal": "HOLD", "active_bull_boxes": [], "active_bear_boxes": [], "last_close": 0.0, "current_trend": None, "trend_changed": False}

        print(Fore.CYAN + Style.DIM + f"# Analyzing {len(df)} candles...")

        # --- Volumatic Trend Calculations ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.vol_atr_period)
        df['ema1'] = self._ema_swma(df['close'], length=self.length)
        df['ema2'] = ta.ema(df['close'], length=self.length)

        # Determine trend (ema1 < ema2 means uptrend in the script's logic)
        df['trend_up'] = df['ema1'] < df['ema2']
        df['trend_changed'] = df['trend_up'] != df['trend_up'].shift(1)

        # --- Update Levels on Trend Change ---
        # We need to iterate or use vectorized logic carefully to update state vars
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else None
        current_trend_up = last_row['trend_up']
        trend_just_changed = last_row['trend_changed']

        if trend_just_changed:
            self.current_trend = current_trend_up
            self.last_trend_change_index = df.index[-1] # Use DataFrame index
            current_ema1 = last_row['ema1']
            current_atr = last_row['atr']
            if pd.notna(current_ema1) and pd.notna(current_atr):
                self.upper = current_ema1 + current_atr * 3
                self.lower = current_ema1 - current_atr * 3
                self.lower_vol = self.lower + current_atr * 4
                self.upper_vol = self.upper - current_atr * 4
                # Prevent levels from crossing due to large ATR
                if self.lower_vol < self.lower: self.lower_vol = self.lower
                if self.upper_vol > self.upper: self.upper_vol = self.upper

                self.step_up = (self.lower_vol - self.lower) / 100 if self.lower_vol > self.lower else 0
                self.step_dn = (self.upper - self.upper_vol) / 100 if self.upper > self.upper_vol else 0
            else:
                 # Reset if data is NaN
                 self.upper, self.lower, self.lower_vol, self.upper_vol, self.step_up, self.step_dn = [None] * 6
            print(Fore.MAGENTA + f"Trend Changed! New Trend: {'UP' if current_trend_up else 'DOWN'}. Levels Updated.")
            print(f"Upper: {self.upper:.2f}, Lower: {self.lower:.2f}, UpVol: {self.upper_vol:.2f}, LowVol: {self.lower_vol:.2f}")

        # --- Volume Normalization ---
        # Use rolling percentile for volume normalization
        df['vol_percentile'] = df['volume'].rolling(window=self.vol_percentile_len, min_periods=min(self.vol_percentile_len // 2, 50)).apply(
            lambda x: np.percentile(x[~np.isnan(x)], self.vol_percentile) if len(x[~np.isnan(x)]) > 0 else np.nan, raw=True
        )
        # Handle cases where percentile might be 0 or NaN
        df['vol_norm'] = np.where(
            (df['vol_percentile'].notna()) & (df['vol_percentile'] > 0),
            (df['volume'] / df['vol_percentile'] * 100),
            0 # Assign 0 if percentile is NaN or 0
        ).fillna(0).astype(float) # Use float for calculations

        # Calculate volume levels for plotting (not directly used in signal logic here)
        df['vol_up_level'] = self.lower + self.step_up * df['vol_norm'] if self.lower is not None and self.step_up is not None else np.nan
        df['vol_dn_level'] = self.upper - self.step_dn * df['vol_norm'] if self.upper is not None and self.step_dn is not None else np.nan

        # --- Pivot Order Block Calculations ---
        df['ph'] = self._find_pivots(df, self.pivot_left_h, self.pivot_right_h, is_high=True)
        df['pl'] = self._find_pivots(df, self.pivot_left_l, self.pivot_right_l, is_high=False)

        # --- Create and Manage Order Blocks ---
        # Iterate backwards to check recent pivots and create boxes
        # Limit check range for performance if df is very large
        check_range = min(len(df), max(self.pivot_right_h, self.pivot_right_l) + 5)

        new_boxes_created = False
        for i in range(len(df) - check_range, len(df)):
            current_index = df.index[i] # Get actual index (could be timestamp or int)

            # Bearish Box from Pivot High
            if pd.notna(df['ph'].iloc[i]):
                pivot_bar_index = i # Index within the current slice
                ob_start_index = pivot_bar_index - self.pivot_right_h # Index where pivot was confirmed

                # Check if a box for this pivot already exists
                if not any(b['id'] == ob_start_index for b in self.bear_boxes):
                    top_price, bottom_price = np.nan, np.nan
                    if self.ob_source == "Wicks":
                        top_price = df['high'].iloc[ob_start_index]
                        bottom_price = df['close'].iloc[ob_start_index] # Pine used close for bottom wick OB
                    else: # Bodys
                        top_price = df['close'].iloc[ob_start_index]
                        bottom_price = df['open'].iloc[ob_start_index]

                    # Ensure top > bottom
                    if pd.notna(top_price) and pd.notna(bottom_price):
                        if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price # Swap if needed

                        new_box = OrderBlock(
                            id=ob_start_index, type='bear', left_idx=ob_start_index, right_idx=len(df)-1, # Extend to current bar
                            top=top_price, bottom=bottom_price, active=True, closed_idx=None
                        )
                        self.bear_boxes.append(new_box)
                        new_boxes_created = True
                        print(Fore.RED + f"New Bearish OB created at index {ob_start_index}: Top={top_price:.2f}, Bottom={bottom_price:.2f}")


            # Bullish Box from Pivot Low
            if pd.notna(df['pl'].iloc[i]):
                pivot_bar_index = i
                ob_start_index = pivot_bar_index - self.pivot_right_l

                if not any(b['id'] == ob_start_index for b in self.bull_boxes):
                    top_price, bottom_price = np.nan, np.nan
                    if self.ob_source == "Wicks":
                         top_price = df['open'].iloc[ob_start_index] # Pine used open for top wick OB
                         bottom_price = df['low'].iloc[ob_start_index]
                    else: # Bodys
                         top_price = df['open'].iloc[ob_start_index] # Pine used open for top body OB
                         bottom_price = df['close'].iloc[ob_start_index] # Pine used close for bottom body OB

                    if pd.notna(top_price) and pd.notna(bottom_price):
                        if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price

                        new_box = OrderBlock(
                            id=ob_start_index, type='bull', left_idx=ob_start_index, right_idx=len(df)-1,
                            top=top_price, bottom=bottom_price, active=True, closed_idx=None
                        )
                        self.bull_boxes.append(new_box)
                        new_boxes_created = True
                        print(Fore.GREEN + f"New Bullish OB created at index {ob_start_index}: Top={top_price:.2f}, Bottom={bottom_price:.2f}")

        # Manage existing boxes (close or extend)
        current_close = last_row['close']
        current_bar_idx = len(df) - 1 # Use simple integer index for management

        for box in self.bull_boxes:
            if box['active']:
                if current_close < box['bottom']: # Price closed below bull box
                    box['active'] = False
                    box['closed_idx'] = current_bar_idx
                    print(Fore.YELLOW + f"Bullish OB {box['id']} closed at {current_close:.2f} (below {box['bottom']:.2f})")
                else:
                    box['right_idx'] = current_bar_idx # Extend active box

        for box in self.bear_boxes:
            if box['active']:
                if current_close > box['top']: # Price closed above bear box
                    box['active'] = False
                    box['closed_idx'] = current_bar_idx
                    print(Fore.YELLOW + f"Bearish OB {box['id']} closed at {current_close:.2f} (above {box['top']:.2f})")
                else:
                    box['right_idx'] = current_bar_idx

        # Limit number of active boxes stored (optional, based on input)
        active_bull = [b for b in self.bull_boxes if b['active']]
        active_bear = [b for b in self.bear_boxes if b['active']]
        # Keep only the most recent 'max_boxes' active ones if needed
        self.bull_boxes = sorted([b for b in self.bull_boxes if not b['active']], key=lambda x: x['id'], reverse=True)[:self.max_boxes*2] + sorted(active_bull, key=lambda x: x['id'], reverse=True)[:self.max_boxes]
        self.bear_boxes = sorted([b for b in self.bear_boxes if not b['active']], key=lambda x: x['id'], reverse=True)[:self.max_boxes*2] + sorted(active_bear, key=lambda x: x['id'], reverse=True)[:self.max_boxes]


        # --- Signal Generation ---
        signal = "HOLD"
        active_bull_boxes = [b for b in self.bull_boxes if b['active']]
        active_bear_boxes = [b for b in self.bear_boxes if b['active']]

        # Check for Trend Change Exit first
        if trend_just_changed:
            if not current_trend_up and self.last_signal == "BUY": # Trend flipped down while long
                signal = "EXIT_LONG"
                print(Fore.RED + Style.BRIGHT + f"*** EXIT LONG Signal (Trend Flip to DOWN) at {current_close:.2f} ***")
            elif current_trend_up and self.last_signal == "SELL": # Trend flipped up while short
                 signal = "EXIT_SHORT"
                 print(Fore.GREEN + Style.BRIGHT + f"*** EXIT SHORT Signal (Trend Flip to UP) at {current_close:.2f} ***")

        # Check for Entries only if not exiting
        if signal == "HOLD":
            if current_trend_up: # Look for Long Entries
                for box in active_bull_boxes:
                    # Check if close is within the box range
                    if box['bottom'] <= current_close <= box['top']:
                        signal = "BUY"
                        print(Fore.GREEN + Style.BRIGHT + f"*** BUY Signal (Trend UP + Price in Bull OB {box['id']}) at {current_close:.2f} ***")
                        break # Take first signal

            elif not current_trend_up: # Look for Short Entries (Trend is DOWN)
                for box in active_bear_boxes:
                     if box['bottom'] <= current_close <= box['top']:
                        signal = "SELL"
                        print(Fore.RED + Style.BRIGHT + f"*** SELL Signal (Trend DOWN + Price in Bear OB {box['id']}) at {current_close:.2f} ***")
                        break

        self.last_signal = signal # Update last signal state

        return {
            "dataframe": df,
            "last_signal": signal,
            "active_bull_boxes": active_bull_boxes,
            "active_bear_boxes": active_bear_boxes,
            "last_close": current_close,
            "current_trend": self.current_trend,
            "trend_changed": trend_just_changed
        }

# --- Data Fetching (Modified to return DataFrame) ---
def fetch_bybit_data(symbol: str, interval: str, limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetches historical Klines and returns a pandas DataFrame."""
    print(Fore.CYAN + f"\n# Summoning historical data for {symbol} ({interval})...")
    url = f"{BYBIT_API_BASE}/v5/market/kline"
    params = { "category": "linear", "symbol": symbol, "interval": interval, "limit": limit }
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get('retCode') != 0 or not data.get('result') or not data['result'].get('list'):
            print(Fore.RED + f"Error fetching data: {data.get('retMsg', 'Invalid response structure')}")
            return None

        kline_list = data['result']['list']
        if not kline_list:
            print(Fore.YELLOW + "Warning: Received empty kline list from Bybit.")
            return None

        # Create DataFrame
        df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df = df.astype({
            'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
            'low': 'float64', 'close': 'float64', 'volume': 'float64', 'turnover': 'float64'
        })
        # Convert timestamp to datetime index (optional but good practice)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.sort_index() # Ensure chronological order

        print(Fore.GREEN + f"Historical data summoned ({len(df)} candles).")
        return df

    except requests.exceptions.RequestException as e:
        print(Fore.RED + Style.BRIGHT + f"Network Error fetching data: {e}")
        return None
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"An unexpected error occurred during data fetch: {e}")
        return None


# --- WebSocket Handling (Modified to update DataFrame) ---
ws_app: Optional[websocket.WebSocketApp] = None
ws_thread: Optional[threading.Thread] = None
current_symbol_interval: Dict[str, str] = {"symbol": "", "interval": ""}
latest_dataframe: Optional[pd.DataFrame] = None # Store the DataFrame
strategy_instance: Optional[VolumaticOBStrategy] = None # Hold the strategy object
data_lock = threading.Lock() # Lock for thread-safe DataFrame updates

def process_kline_update(kline_data: Dict):
    """Processes a single kline update and updates the DataFrame."""
    global latest_dataframe, strategy_instance
    try:
        ts_ms = int(kline_data['timestamp'])
        ts = pd.to_datetime(ts_ms, unit='ms')
        k_open = float(kline_data['open'])
        k_high = float(kline_data['high'])
        k_low = float(kline_data['low'])
        k_close = float(kline_data['close'])
        k_volume = float(kline_data['volume'])
        k_turnover = float(kline_data['turnover'])
        is_final = kline_data.get('confirm', False)

        with data_lock:
            if latest_dataframe is None:
                print(Fore.YELLOW + "Warning: DataFrame not initialized, cannot process WS update.")
                return

            new_row = pd.DataFrame([{
                'open': k_open, 'high': k_high, 'low': k_low, 'close': k_close,
                'volume': k_volume, 'turnover': k_turnover
            }], index=[ts])

            # Check if this timestamp already exists (update) or is new
            if ts in latest_dataframe.index:
                # Update the existing row - careful about volume accumulation if needed
                # For TA, usually just update OHLCV
                latest_dataframe.loc[ts] = new_row.iloc[0]
                action = "Updated"
            else:
                # Append new row and potentially drop oldest if exceeding a limit
                latest_dataframe = pd.concat([latest_dataframe, new_row])
                # Optional: Limit DataFrame size
                # max_df_len = 2000
                # if len(latest_dataframe) > max_df_len:
                #     latest_dataframe = latest_dataframe.iloc[-max_df_len:]
                action = "Appended"

            # --- Re-run analysis on the updated DataFrame ---
            if strategy_instance:
                analysis_results = strategy_instance.update(latest_dataframe.copy()) # Analyze a copy
                # Display results or act on signal
                display_analysis_results(analysis_results, is_live_update=True)
                # Check for trade execution based on signal
                handle_signal(analysis_results['last_signal'], analysis_results['last_close'])

            status_color = Fore.GREEN if is_final else Fore.YELLOW
            print(status_color + f"WS [{current_symbol_interval['symbol']}/{current_symbol_interval['interval']}] {action}: " +
                  f"T={ts.strftime('%H:%M:%S')} O={k_open:.2f} H={k_high:.2f} L={k_low:.2f} C={k_close:.2f} V={k_volume:.2f} Final={is_final}")


    except Exception as e:
        print(Fore.RED + f"Error processing kline update: {e}")


def on_ws_message(ws, message):
    """Handles incoming WebSocket messages."""
    try:
        data = json.loads(message)
        if data.get('op') == 'pong' or data.get('op') == 'subscribe': return

        if data.get('topic', '').startswith('kline.') and data.get('data'):
            kline_data = data['data'][0]
            process_kline_update(kline_data) # Process the update

    except json.JSONDecodeError: pass # Ignore non-JSON
    except Exception as e: print(Fore.RED + f"Error in on_ws_message: {e}")

# Keep on_ws_error, on_ws_close, on_ws_open, start_websocket, stop_websocket as before
# Ensure on_ws_open uses current_symbol_interval correctly

# --- Order Placement (Keep as before) ---
def place_order_via_backend(symbol: str, side: str, order_type: str, qty: float) -> Optional[Dict]:
    # (Code from previous version - unchanged)
    print(Fore.CYAN + f"\n# Sending {side} {order_type} request for {qty} {symbol} to Backend Bastion...")
    url = f"{BACKEND_URL}/api/place-order"
    payload = { "symbol": symbol, "side": side, "orderType": order_type, "qty": qty }
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response_data = response.json()
        # (Rest of the function is the same)
        if response.status_code == 200 and response_data.get("success"):
            print(Fore.GREEN + Style.BRIGHT + "Backend Confirmation:")
            print(Fore.GREEN + f"  Message: {response_data.get('message')}")
            print(Fore.GREEN + f"  OrderID: {response_data.get('orderId')}")
            return response_data
        else:
            print(Fore.RED + Style.BRIGHT + "Backend Rejection:")
            print(Fore.RED + f"  Status Code: {response.status_code}")
            print(Fore.RED + f"  Error: {response_data.get('error', 'Unknown')}")
            print(Fore.RED + f"  Details: {response_data.get('details', 'N/A')}")
            return response_data
    except requests.exceptions.ConnectionError:
         print(Fore.RED + Style.BRIGHT + f"FATAL: Could not connect to Backend Bastion at {url}. Is it running?")
         return None
    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"Error communicating with Backend Bastion: {e}")
        return None

# --- Display Helper (Modified) ---
def display_analysis_results(results: Optional[AnalysisResults], is_live_update=False):
    """Prints the analysis results in a formatted way."""
    if not results or results['dataframe'].empty:
        print(Fore.YELLOW + "No analysis results to display.")
        return

    df = results['dataframe']
    last_row = df.iloc[-1]

    if not is_live_update: # Print full summary only on fetch/analyze
        print(Fore.MAGENTA + "\n--- Strategy Analysis ---")
        print(f"{'Metric':<18} {'Value'}")
        print(f"{'-'*18} {'-'*25}")

        trend_str = Fore.GREEN + "UP" if results['current_trend'] else Fore.RED + "DOWN" if results['current_trend'] is False else Fore.WHITE + "N/A"
        print(f"{'Current Trend':<18} {trend_str}")
        print(f"{'Trend Changed':<18} {Fore.YELLOW + str(results['trend_changed']) if results['trend_changed'] else str(results['trend_changed'])}")
        print(f"{'Last Close':<18} {Fore.YELLOW}{results['last_close']:,.4f}") # More precision for crypto
        print(f"{'EMA 1':<18} {Fore.CYAN}{last_row.get('ema1', np.nan):,.4f}")
        print(f"{'EMA 2':<18} {Fore.CYAN}{last_row.get('ema2', np.nan):,.4f}")
        print(f"{'ATR':<18} {Fore.WHITE}{last_row.get('atr', np.nan):,.4f}")
        print(f"{'Vol Norm (%)':<18} {Fore.WHITE}{last_row.get('vol_norm', np.nan):,.1f}%")

        print(Fore.BLUE + "\nActive Order Blocks:")
        if not results['active_bull_boxes'] and not results['active_bear_boxes']:
             print(Fore.WHITE + Style.DIM + "  None")
        for box in results['active_bull_boxes']:
            print(Fore.GREEN + f"  Bull OB {box['id']}: Top={box['top']:,.2f}, Bottom={box['bottom']:,.2f}")
        for box in results['active_bear_boxes']:
            print(Fore.RED + f"  Bear OB {box['id']}: Top={box['top']:,.2f}, Bottom={box['bottom']:,.2f}")

    # Always print the latest signal
    signal = results['last_signal']
    sig_color = Fore.WHITE
    if signal == "BUY": sig_color = Fore.GREEN + Style.BRIGHT
    elif signal == "SELL": sig_color = Fore.RED + Style.BRIGHT
    elif "EXIT" in signal: sig_color = Fore.YELLOW + Style.BRIGHT
    print(f"\n{'Latest Signal':<18} {sig_color}{signal}")
    print("-" * 30)


# --- Signal Handling ---
# Basic state machine for position tracking (in-memory, resets on script restart)
current_position = None # None, 'LONG', 'SHORT'

def handle_signal(signal: str, price: float):
    """Acts on the generated signal (e.g., places orders)."""
    global current_position
    # Define order quantity (example, make this configurable)
    order_qty = 0.001

    if signal == "BUY" and current_position != 'LONG':
        print(Fore.GREEN + Style.BRIGHT + f"ACTION: Attempting Market Buy {order_qty} at ~{price:.2f}")
        # Close short position first if exists (optional, depends on hedging/mode)
        if current_position == 'SHORT':
             print(Fore.YELLOW + "Closing existing SHORT position before BUY...")
             place_order_via_backend(current_symbol_interval['symbol'], "Buy", "Market", order_qty) # Assuming closing qty = entry qty
        # Place Buy order
        result = place_order_via_backend(current_symbol_interval['symbol'], "Buy", "Market", order_qty)
        if result and result.get('success'):
            current_position = 'LONG'

    elif signal == "SELL" and current_position != 'SHORT':
        print(Fore.RED + Style.BRIGHT + f"ACTION: Attempting Market Sell {order_qty} at ~{price:.2f}")
        # Close long position first if exists
        if current_position == 'LONG':
            print(Fore.YELLOW + "Closing existing LONG position before SELL...")
            place_order_via_backend(current_symbol_interval['symbol'], "Sell", "Market", order_qty)
        # Place Sell order
        result = place_order_via_backend(current_symbol_interval['symbol'], "Sell", "Market", order_qty)
        if result and result.get('success'):
            current_position = 'SHORT'

    elif signal == "EXIT_LONG" and current_position == 'LONG':
        print(Fore.YELLOW + Style.BRIGHT + f"ACTION: Attempting Market Sell (Exit Long) {order_qty} at ~{price:.2f}")
        result = place_order_via_backend(current_symbol_interval['symbol'], "Sell", "Market", order_qty)
        if result and result.get('success'):
            current_position = None

    elif signal == "EXIT_SHORT" and current_position == 'SHORT':
        print(Fore.YELLOW + Style.BRIGHT + f"ACTION: Attempting Market Buy (Exit Short) {order_qty} at ~{price:.2f}")
        result = place_order_via_backend(current_symbol_interval['symbol'], "Buy", "Market", order_qty)
        if result and result.get('success'):
            current_position = None


# --- Main Execution Charm (Modified) ---
if __name__ == "__main__":
    global latest_dataframe, strategy_instance # Make accessible in main scope

    # --- Strategy Initialization ---
    # Use default parameters or allow modification via args/config file
    strategy_instance = VolumaticOBStrategy(
        length=40, pivot_left_h=10, pivot_right_h=10, # Smaller pivots for demo
        pivot_left_l=10, pivot_right_l=10, max_boxes=5
    )

    current_symbol = "BTCUSDT"
    current_interval = "5" # Default 5m for more frequent signals/updates
    analysis_results: Optional[AnalysisResults] = None
    is_live = False

    print(Fore.MAGENTA + Style.BRIGHT + "~~~ Pyrmethus Volumatic+OB Strategy CLI ~~~")
    print(Fore.YELLOW + "Ensure backend_server.py is running!")

    while True:
        live_status = Fore.GREEN + "(Live)" if is_live else Fore.RED + "(Offline)"
        pos_status = f"Pos: {Fore.CYAN}{current_position or 'None'}"
        prompt = Fore.BLUE + f"\n[{current_symbol}/{current_interval}] {live_status} {pos_status} | Cmd (fetch, analyze, live, stop, set, exit): " + Style.RESET_ALL
        command = input(prompt).lower().strip()

        if command == "exit":
            if is_live: stop_websocket()
            print(Fore.MAGENTA + "Farewell!")
            break
        elif command == "fetch":
            with data_lock: # Protect access while fetching
                latest_dataframe = fetch_bybit_data(current_symbol, current_interval)
                if latest_dataframe is not None and strategy_instance:
                    analysis_results = strategy_instance.update(latest_dataframe.copy()) # Analyze a copy
                    display_analysis_results(analysis_results)
                else:
                    print(Fore.RED + "Failed to fetch or process data.")
        elif command == "analyze":
             if analysis_results:
                 display_analysis_results(analysis_results)
             else:
                 print(Fore.YELLOW + "No data fetched/analyzed yet. Use 'fetch' first.")
        elif command == "live":
            if not is_live:
                print(Fore.CYAN + "# Fetching initial data before going live...")
                with data_lock:
                    latest_dataframe = fetch_bybit_data(current_symbol, current_interval)
                    if latest_dataframe is not None and strategy_instance:
                        analysis_results = strategy_instance.update(latest_dataframe.copy())
                        display_analysis_results(analysis_results)
                        start_websocket(current_symbol, current_interval)
                        is_live = True
                    else:
                         print(Fore.RED + "Failed to fetch initial data. Cannot start live feed.")
            else:
                print(Fore.YELLOW + "Live feed already active.")
        elif command == "stop":
             if is_live:
                 stop_websocket()
                 is_live = False
             else:
                 print(Fore.YELLOW + "Live feed is not active.")
        # Removed direct buy/sell commands - rely on strategy signals
        # elif command == "buy": ...
        # elif command == "sell": ...
        elif command == "set":
             new_symbol = input(Fore.BLUE + f"Enter new symbol [{current_symbol}]: " + Style.RESET_ALL).strip().upper()
             new_interval = input(Fore.BLUE + f"Enter new interval (e.g., 1, 5, 15, 60) [{current_interval}]: " + Style.RESET_ALL).strip()
             if new_symbol: current_symbol = new_symbol
             if new_interval: current_interval = new_interval
             print(Fore.GREEN + f"Context set to: {current_symbol}/{current_interval}")
             # Reset state and data if context changes
             with data_lock:
                 latest_dataframe = None
                 analysis_results = None
                 current_position = None # Reset position on context change
                 if strategy_instance: # Reset strategy state if needed (or re-init)
                     strategy_instance.bull_boxes = []
                     strategy_instance.bear_boxes = []
                     strategy_instance.last_signal = "HOLD"
                     # Consider re-initializing strategy if params depend on symbol/interval
             if is_live:
                 print(Fore.YELLOW + "Restarting live feed for new context...")
                 stop_websocket()
                 is_live = False
                 # Auto-fetch and restart live
                 print(Fore.CYAN + "# Fetching initial data for new context...")
                 with data_lock:
                    latest_dataframe = fetch_bybit_data(current_symbol, current_interval)
                    if latest_dataframe is not None and strategy_instance:
                        analysis_results = strategy_instance.update(latest_dataframe.copy())
                        display_analysis_results(analysis_results)
                        start_websocket(current_symbol, current_interval)
                        is_live = True
                    else:
                         print(Fore.RED + "Failed to fetch data for new context. Live feed stopped.")

        else:
            print(Fore.YELLOW + "Unknown command. Available: fetch, analyze, live, stop, set, exit")

```

**How to Use:**

1.  **Save:** Replace the *entire* content of your existing `frontend_cli.py` with the code above. Keep `backend_server.py` and `.env` as they are.
2.  **Install:** Ensure `pandas`, `pandas-ta`, and `numpy` are installed (`pip install pandas pandas-ta numpy`).
3.  **Run Backend:** Start `python backend_server.py` in one Termux session.
4.  **Run Frontend:** Start `python frontend_cli.py` in a second Termux session.
5.  **Interact:**
    *   Use `fetch` to get historical data and see the initial analysis/signals.
    *   Use `live` to connect to the WebSocket. New candles will automatically update the analysis and potentially trigger buy/sell actions via the backend based on the strategy logic.
    *   Use `stop` to disconnect the WebSocket.
    *   Use `set` to change the trading pair or timeframe.
    *   Use `exit` to quit.

This Python script now embodies the core logic of your Pine Script indicator, translating it into actionable trading signals within the secure Termux environment, ready to command the markets through its backend familiar. Remember that this is a starting point; thorough testing and potential refinement of the strategy logic (especially exits and risk management) are crucial before deploying significant capital.
