#!/bin/bash

# Script to set up the Pyrmethus Volumatic Trend + OB Bot directory and files (CCXT Async Version)

PROJECT_DIR="pyrmethus_volumatic_bot_ccxt"
ENV_FILE=".env"
CONFIG_FILE="config.json"
STRATEGY_FILE="strategy.py"
MAIN_FILE="main.py"
REQ_FILE="requirements.txt"
SETUP_SCRIPT_NAME="setup_bot.sh" # Name of this script

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up Pyrmethus Volumatic Bot (CCXT Async Version) in: ${YELLOW}${PROJECT_DIR}${NC}"

# Create project directory
mkdir -p "$PROJECT_DIR"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Could not create directory ${PROJECT_DIR}${NC}"
    exit 1
fi
cd "$PROJECT_DIR" || exit 1

echo -e "${GREEN}Directory created successfully.${NC}"

# --- Create requirements.txt ---
echo -e "${BLUE}Creating ${YELLOW}${REQ_FILE}${NC}..."
cat << EOF > "$REQ_FILE"
ccxt>=4.1.60 # Use a recent version for stable Bybit V5 & asyncio support
python-dotenv
pandas
pandas-ta
numpy
colorama
# asyncio is built-in
requests # Still useful for checks or other APIs
EOF

# --- Create .env file ---
echo -e "${BLUE}Creating ${YELLOW}${ENV_FILE}${NC} (Remember to fill in your API keys!)"
cat << EOF > "$ENV_FILE"
# Bybit API Credentials (use Testnet keys for development!)
# Register API keys here: https://www.bybit.com/app/user/api-management
BYBIT_API_KEY="YOUR_API_KEY_HERE"
BYBIT_API_SECRET="YOUR_API_SECRET_HERE"

# Optional: Set to True for Testnet, False or leave blank for Mainnet
BYBIT_TESTNET="True"
EOF

# --- Create config.json ---
echo -e "${BLUE}Creating ${YELLOW}${CONFIG_FILE}${NC}..."
cat << EOF > "$CONFIG_FILE"
{
  "exchange": "bybit", // Specify exchange for CCXT
  "symbol": "BTC/USDT", // CCXT standard symbol format
  "timeframe": "5m", // CCXT standard timeframe format (e.g., 1m, 5m, 1h, 1d)
  "account_type": "contract", // 'unified', 'contract', or 'spot' (adjust based on your Bybit setup & CCXT needs)
  "mode": "Live", // "Live" or "Paper"
  "log_level": "INFO", // DEBUG, INFO, WARNING, ERROR

  "order": {
    "type": "Market", // "Market" or "Limit" (Limit orders require careful price handling)
    "risk_per_trade_percent": 1.0, // e.g., 1.0 means risk 1% of equity per trade
    "leverage": 5, // Desired leverage (Ensure it's set on Bybit manually or via API if supported)
    "tp_ratio": 2.0, // Take Profit Risk:Reward ratio (e.g., 2.0 means TP is 2x SL distance)
    "sl_trigger_type": "LastPrice", // Bybit trigger: LastPrice, IndexPrice, MarkPrice
    "tp_trigger_type": "LastPrice"  // Bybit trigger: LastPrice, IndexPrice, MarkPrice
  },

  "strategy": {
    "class": "VolumaticOBStrategy", // Matches class name in strategy.py
    "params": {
      "length": 40,
      "vol_atr_period": 200,
      "vol_percentile_len": 1000,
      "vol_percentile": 95, // Example: Use 95th percentile
      "ob_source": "Wicks",
      "pivot_left_h": 10,
      "pivot_right_h": 10,
      "pivot_left_l": 10,
      "pivot_right_l": 10,
      "max_boxes": 5
    },
    "stop_loss": {
      "method": "ATR", // "ATR" or "OB" (Use Order Block boundary)
      "atr_multiplier": 1.5 // ATR multiplier for stop loss
    }
  },

  "data": {
      "fetch_limit": 750, // Candles for initial fetch
      "max_df_len": 2000 // Max candles to keep in memory
  },

  "checks": {
      "position_check_interval": 30, // How often to fetch position via REST (seconds)
      "health_check_interval": 60 // How often to run general health checks (seconds)
  }
}
EOF

# --- Create strategy.py ---
echo -e "${BLUE}Creating ${YELLOW}${STRATEGY_FILE}${NC}..."
cat << 'EOF' > "$STRATEGY_FILE"
# -*- coding: utf-8 -*-
"""
Strategy Definition: Enhanced Volumatic Trend + OB (for CCXT Async Bot)
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style
from typing import List, Dict, Optional, Any, TypedDict, Tuple
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext

log = logging.getLogger(__name__) # Gets logger configured in main.py
getcontext().prec = 18 # Set decimal precision

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
    Accepts CCXT market structure for precision handling.
    """
    def __init__(self, market: Dict[str, Any], **params):
        """
        Initializes the strategy.
        Args:
            market: CCXT market structure dictionary.
            **params: Strategy parameters loaded from config.json.
        """
        self.market = market
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

        # Get precision from CCXT market structure
        self.price_precision = int(market.get('precision', {}).get('price', 8)) # Default 8 if not found
        self.amount_precision = int(market.get('precision', {}).get('amount', 8)) # Default 8
        self.price_tick = Decimal(str(market.get('precision', {}).get('price', '0.00000001')))
        self.amount_tick = Decimal(str(market.get('precision', {}).get('amount', '0.00000001')))


        log.info(f"{Fore.MAGENTA}Initializing VolumaticOB Strategy Engine...{Style.RESET_ALL}")
        log.info(f"Symbol: {market['symbol']}")
        log.info(f"Params: TrendLen={self.length}, Pivots={self.pivot_left_h}/{self.pivot_right_h}, "
                 f"MaxBoxes={self.max_boxes}, OB Source={self.ob_source}")
        log.info(f"Minimum data points required: {self.min_data_len}")
        log.debug(f"Price Precision: {self.price_precision}, Amount Precision: {self.amount_precision}")
        log.debug(f"Price Tick: {self.price_tick}, Amount Tick: {self.amount_tick}")


    def _parse_params(self, params: Dict):
        """Load parameters from the config dict."""
        self.length = int(params.get('length', 40))
        self.vol_atr_period = int(params.get('vol_atr_period', 200))
        self.vol_percentile_len = int(params.get('vol_percentile_len', 1000))
        self.vol_percentile = int(params.get('vol_percentile', 95))
        self.ob_source = str(params.get('ob_source', "Wicks"))
        self.pivot_left_h = int(params.get('pivot_left_h', 10))
        self.pivot_right_h = int(params.get('pivot_right_h', 10))
        self.pivot_left_l = int(params.get('pivot_left_l', 10))
        self.pivot_right_l = int(params.get('pivot_right_l', 10))
        self.max_boxes = int(params.get('max_boxes', 5))

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

    # --- Precision Handling using CCXT market data ---
    def format_price(self, price: float) -> str:
        """Formats price according to market precision."""
        return format(Decimal(str(price)), f'.{self.price_precision}f')

    def format_amount(self, amount: float) -> str:
        """Formats amount according to market precision."""
        return format(Decimal(str(amount)), f'.{self.amount_precision}f')

    def round_price(self, price: float) -> float:
        """Rounds price based on the tick size."""
        price_decimal = Decimal(str(price))
        rounded = (price_decimal / self.price_tick).quantize(Decimal('1'), rounding=ROUND_UP if price_decimal > 0 else ROUND_DOWN) * self.price_tick
        return float(rounded)

    def round_amount(self, amount: float) -> float:
        """Rounds amount DOWN based on the amount tick size (qtyStep)."""
        amount_decimal = Decimal(str(amount))
        rounded = (amount_decimal / self.amount_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * self.amount_tick
        return float(rounded)
    # --- End Precision Handling ---


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

        for i in range(left, len(df) - right):
            pivot_val = source_series.iloc[i]
            if pd.isna(pivot_val): continue

            is_pivot = True
            for j in range(1, left + 1):
                left_val = source_series.iloc[i - j]
                if pd.isna(left_val): continue
                if (is_high and left_val > pivot_val) or \
                   (not is_high and left_val < pivot_val):
                    is_pivot = False; break
            if not is_pivot: continue

            for j in range(1, right + 1):
                right_val = source_series.iloc[i + j]
                if pd.isna(right_val): continue
                if (is_high and right_val >= pivot_val) or \
                   (not is_high and right_val <= pivot_val):
                    is_pivot = False; break

            if is_pivot:
                pivots.iloc[i] = pivot_val
        return pivots

    def update(self, df_input: pd.DataFrame) -> AnalysisResults:
        """
        Calculates indicators and generates signals based on the input DataFrame.
        Args:
            df_input: pandas DataFrame with OHLCV columns and DatetimeIndex.
        Returns:
            AnalysisResults dictionary.
        """
        required_len = self.min_data_len
        if df_input.empty or len(df_input) < required_len:
            log.warning(f"Not enough data ({len(df_input)}/{required_len}) for analysis.")
            return AnalysisResults(
                dataframe=df_input, last_signal="HOLD", active_bull_boxes=self.bull_boxes,
                active_bear_boxes=self.bear_boxes, last_close=np.nan, current_trend=self.current_trend,
                trend_changed=False, last_atr=None
            )

        df = df_input.copy()
        log.debug(f"Analyzing {len(df)} candles ending {df.index[-1]}...")

        # --- Volumatic Trend Calculations ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.vol_atr_period, fillna=np.nan)
        df['ema1'] = self._ema_swma(df['close'], length=self.length)
        df['ema2'] = ta.ema(df['close'], length=self.length, fillna=np.nan)

        df['trend_up'] = np.where(df['ema1'] < df['ema2'], True,
                         np.where(df['ema1'] >= df['ema2'], False, np.nan))
        df['trend_up'] = df['trend_up'].ffill()

        df['trend_changed'] = (df['trend_up'] != df['trend_up'].shift(1)) & df['trend_up'].notna() & df['trend_up'].shift(1).notna()

        last_row = df.iloc[-1]
        current_trend_up = last_row['trend_up']
        trend_just_changed = last_row['trend_changed']
        last_atr_value = last_row['atr']

        if pd.notna(current_trend_up):
            if self.current_trend is None or (trend_just_changed and current_trend_up != self.current_trend):
                is_initial_trend = self.current_trend is None
                self.current_trend = current_trend_up
                log.info(f"{Fore.MAGENTA}{'Initial Trend Detected' if is_initial_trend else 'Trend Changed'}! New Trend: {'UP' if current_trend_up else 'DOWN'}. Updating levels...{Style.RESET_ALL}")

                current_ema1 = last_row['ema1']
                current_atr = last_row['atr']
                if pd.notna(current_ema1) and pd.notna(current_atr) and current_atr > 1e-9:
                    self.upper = current_ema1 + current_atr * 3
                    self.lower = current_ema1 - current_atr * 3
                    self.lower_vol = self.lower + current_atr * 4
                    self.upper_vol = self.upper - current_atr * 4
                    if self.lower_vol < self.lower: self.lower_vol = self.lower
                    if self.upper_vol > self.upper: self.upper_vol = self.upper
                    self.step_up = (self.lower_vol - self.lower) / 100 if self.lower_vol > self.lower else 0
                    self.step_dn = (self.upper - self.upper_vol) / 100 if self.upper > self.upper_vol else 0
                    log.info(f"Levels Updated @ {df.index[-1]}: U={self.upper:.{self.price_precision}f}, L={self.lower:.{self.price_precision}f}")
                else:
                     log.warning(f"Could not update levels @ {df.index[-1]} due to NaN/zero values. Levels reset.")
                     self.upper, self.lower, self.lower_vol, self.upper_vol, self.step_up, self.step_dn = [None] * 6

        # --- Volume Normalization ---
        roll_window = min(self.vol_percentile_len, len(df))
        min_p = max(1, min(roll_window // 2, 50))
        df['vol_percentile_val'] = df['volume'].rolling(window=roll_window, min_periods=min_p).apply(
            lambda x: np.nanpercentile(x, self.vol_percentile) if np.any(~np.isnan(x) & (x > 0)) else np.nan, raw=True)

        df['vol_norm'] = np.where(
            (df['vol_percentile_val'].notna()) & (df['vol_percentile_val'] > 1e-9),
            (df['volume'] / df['vol_percentile_val'] * 100), 0).fillna(0).astype(float)

        # --- Pivot Order Block Calculations ---
        df['ph'] = self._find_pivots(df, self.pivot_left_h, self.pivot_right_h, is_high=True)
        df['pl'] = self._find_pivots(df, self.pivot_left_l, self.pivot_right_l, is_high=False)

        # --- Create and Manage Order Blocks ---
        check_start_idx = max(0, len(df) - max(self.pivot_right_h, self.pivot_right_l) - 5)
        new_boxes_created_count = 0
        df['int_index'] = range(len(df))

        for i in range(check_start_idx, len(df)):
            current_int_index = i
            # Bearish Box from Pivot High
            pivot_occur_int_idx = current_int_index - self.pivot_right_h
            if pivot_occur_int_idx >= 0 and pd.notna(df['ph'].iloc[pivot_occur_int_idx]):
                if not any(b['id'] == pivot_occur_int_idx for b in self.bear_boxes):
                    ob_candle = df.iloc[pivot_occur_int_idx]
                    top_price = ob_candle['high'] if self.ob_source == "Wicks" else ob_candle['close']
                    bottom_price = ob_candle['close'] if self.ob_source == "Wicks" else ob_candle['open']
                    if pd.notna(top_price) and pd.notna(bottom_price):
                        if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price
                        if Decimal(str(abs(top_price - bottom_price))) > self.price_tick / 10: # Avoid zero-range relative to tick size
                            new_box = OrderBlock(id=pivot_occur_int_idx, type='bear', left_idx=pivot_occur_int_idx,
                                                 right_idx=len(df)-1, top=top_price, bottom=bottom_price, active=True, closed_idx=None)
                            self.bear_boxes.append(new_box)
                            new_boxes_created_count += 1
                            log.info(f"{Fore.RED}New Bear OB {pivot_occur_int_idx} @ {df.index[pivot_occur_int_idx]}: T={top_price:.{self.price_precision}f}, B={bottom_price:.{self.price_precision}f}{Style.RESET_ALL}")

            # Bullish Box from Pivot Low
            pivot_occur_int_idx = current_int_index - self.pivot_right_l
            if pivot_occur_int_idx >= 0 and pd.notna(df['pl'].iloc[pivot_occur_int_idx]):
                 if not any(b['id'] == pivot_occur_int_idx for b in self.bull_boxes):
                    ob_candle = df.iloc[pivot_occur_int_idx]
                    top_price = ob_candle['open'] if self.ob_source == "Wicks" else ob_candle['open']
                    bottom_price = ob_candle['low'] if self.ob_source == "Wicks" else ob_candle['close']
                    if pd.notna(top_price) and pd.notna(bottom_price):
                        if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price
                        if Decimal(str(abs(top_price - bottom_price))) > self.price_tick / 10:
                            new_box = OrderBlock(id=pivot_occur_int_idx, type='bull', left_idx=pivot_occur_int_idx,
                                                 right_idx=len(df)-1, top=top_price, bottom=bottom_price, active=True, closed_idx=None)
                            self.bull_boxes.append(new_box)
                            new_boxes_created_count += 1
                            log.info(f"{Fore.GREEN}New Bull OB {pivot_occur_int_idx} @ {df.index[pivot_occur_int_idx]}: T={top_price:.{self.price_precision}f}, B={bottom_price:.{self.price_precision}f}{Style.RESET_ALL}")

        # Manage existing boxes
        current_close = last_row['close']
        current_bar_int_idx = len(df) - 1
        if pd.notna(current_close):
            closed_bull_count, closed_bear_count = 0, 0
            for box in self.bull_boxes:
                if box['active']:
                    if current_close < box['bottom']:
                        box['active'] = False; box['closed_idx'] = current_bar_int_idx; closed_bull_count += 1
                    else: box['right_idx'] = current_bar_int_idx
            for box in self.bear_boxes:
                if box['active']:
                    if current_close > box['top']:
                        box['active'] = False; box['closed_idx'] = current_bar_int_idx; closed_bear_count += 1
                    else: box['right_idx'] = current_bar_int_idx
            if closed_bull_count: log.info(f"{Fore.YELLOW}Closed {closed_bull_count} Bull OBs.{Style.RESET_ALL}")
            if closed_bear_count: log.info(f"{Fore.YELLOW}Closed {closed_bear_count} Bear OBs.{Style.RESET_ALL}")

        # Prune Order Blocks
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

        if self.current_trend is not None and pd.notna(current_close):
            if trend_just_changed:
                if not self.current_trend and self.last_signal_state == "BUY": signal = "EXIT_LONG"
                elif self.current_trend and self.last_signal_state == "SELL": signal = "EXIT_SHORT"

            if signal == "HOLD":
                if self.current_trend: # Trend UP -> Look for Long
                    if self.last_signal_state != "BUY":
                        for box in active_bull_boxes:
                            if box['bottom'] <= current_close <= box['top']: signal = "BUY"; break
                elif not self.current_trend: # Trend DOWN -> Look for Short
                    if self.last_signal_state != "SELL":
                        for box in active_bear_boxes:
                            if box['bottom'] <= current_close <= box['top']: signal = "SELL"; break

        # Update internal state
        if signal in ["BUY", "SELL"]: self.last_signal_state = signal
        elif signal in ["EXIT_LONG", "EXIT_SHORT"]: self.last_signal_state = "HOLD"

        # Log signal if it's not just HOLD
        if signal != "HOLD":
             color = Fore.YELLOW if "EXIT" in signal else Fore.GREEN if signal == "BUY" else Fore.RED
             log.warning(f"{color}{Style.BRIGHT}*** {signal} Signal generated at {current_close:.{self.price_precision}f} (Trend: {'UP' if self.current_trend else 'DOWN'}) ***{Style.RESET_ALL}")
        else:
            log.debug(f"Signal: HOLD (Internal State: {self.last_signal_state})")


        df.drop(columns=['int_index'], inplace=True, errors='ignore') # Cleanup index column

        return AnalysisResults(
            dataframe=df, last_signal=signal, active_bull_boxes=active_bull_boxes,
            active_bear_boxes=active_bear_boxes, last_close=current_close,
            current_trend=self.current_trend, trend_changed=trend_just_changed,
            last_atr=last_atr_value if pd.notna(last_atr_value) else None)
EOF

# --- Create main.py ---
echo -e "${BLUE}Creating ${YELLOW}${MAIN_FILE}${NC}..."
cat << 'EOF' > "$MAIN_FILE"
# -*- coding: utf-8 -*-
"""
Pyrmethus Volumatic Trend + OB Trading Bot (CCXT Async Version - Enhanced)
"""
import os
import sys
import json
import time
import datetime
import logging
import signal
import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation, getcontext

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from colorama import init, Fore, Style
import ccxt.async_support as ccxt

# Import strategy class and type hints
from strategy import VolumaticOBStrategy, AnalysisResults

# --- Initialize Colorama ---
init(autoreset=True)
getcontext().prec = 18 # Set decimal precision early

# --- Load Environment Variables ---
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = os.getenv("BYBIT_TESTNET", "False").lower() == "true"

# --- Global Variables ---
config = {}
exchange: Optional[ccxt.bybit] = None # Explicitly type hint exchange if possible
strategy_instance: Optional[VolumaticOBStrategy] = None
market: Optional[Dict[str, Any]] = None
latest_dataframe: Optional[pd.DataFrame] = None
current_position: Dict[str, Any] = {"size": Decimal(0), "side": "None", "entry_price": Decimal(0)} # Store position info
last_position_check_time: float = 0.0
last_health_check_time: float = 0.0
last_ws_update_time: float = 0.0 # Track WebSocket health
running_tasks = set() # Store running asyncio tasks for cancellation

# --- Locks for Shared Resources ---
# Use asyncio locks for async code
data_lock = asyncio.Lock()
position_lock = asyncio.Lock()
order_lock = asyncio.Lock()

# --- Logging Setup ---
log = logging.getLogger("PyrmethusVolumaticBotCCXT")
log_level = logging.INFO # Default, overridden by config

def setup_logging(level_str="INFO"):
    """Configures logging for the application."""
    global log_level
    log_level = getattr(logging, level_str.upper(), logging.INFO)
    log.setLevel(log_level)
    if not log.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        log.addHandler(ch)
        # Optional File Handler
        # log_filename = f"bot_ccxt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        # fh = logging.FileHandler(log_filename) ... add handler
    log.propagate = False

# --- Configuration Loading ---
def load_config(path="config.json") -> Dict:
    """Loads configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            conf = json.load(f)
            # Basic validation
            required_keys = ["exchange", "symbol", "timeframe", "order", "strategy", "data"]
            if not all(key in conf for key in required_keys):
                 raise ValueError("Config file missing required top-level keys.")
            return conf
    except FileNotFoundError:
        log.critical(f"CRITICAL: Configuration file '{path}' not found.")
        sys.exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        log.critical(f"CRITICAL: Error loading or validating configuration '{path}': {e}")
        sys.exit(1)
    except Exception as e:
        log.critical(f"CRITICAL: Unexpected error loading configuration: {e}", exc_info=True)
        sys.exit(1)

# --- CCXT Exchange Interaction (Async) ---
async def connect_ccxt() -> Optional[ccxt.Exchange]:
    """Initializes and connects to the CCXT exchange."""
    global exchange
    exchange_id = config.get('exchange', 'bybit').lower()
    if not hasattr(ccxt, exchange_id):
        log.critical(f"CRITICAL: Exchange '{exchange_id}' is not supported by CCXT.")
        return None

    try:
        log.info(f"Connecting to CCXT exchange '{exchange_id}' ({'Testnet' if TESTNET else 'Mainnet'})...")
        exchange = getattr(ccxt, exchange_id)({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': config.get('account_type', 'contract'), # contract (futures/swap) or spot
                'adjustForTimeDifference': True,
                 # Add Bybit specific options if needed, e.g., for Unified account V5
                'accounts': {'unified': 'UNIFIED', 'contract': 'CONTRACT'}.get(config.get('account_type', 'contract')),
                'recvWindow': 10000, # Increase recvWindow if timestamp errors occur
            }
        })
        if TESTNET:
            exchange.set_sandbox_mode(True)

        # Test connection - fetch time or markets
        await exchange.load_markets() # Also fetches time implicitly
        log.info(f"Successfully connected to {exchange.name}. Loaded {len(exchange.markets)} markets.")
        return exchange
    except ccxt.AuthenticationError as e:
        log.critical(f"CRITICAL: CCXT Authentication Error: {e}. Check API keys and permissions.")
        return None
    except ccxt.NetworkError as e:
         log.critical(f"CRITICAL: CCXT Network Error connecting to exchange: {e}")
         return None
    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize CCXT exchange: {e}", exc_info=True)
        return None

async def load_exchange_market(symbol: str) -> Optional[Dict[str, Any]]:
    """Loads or re-loads market data for a specific symbol."""
    global market
    if not exchange: return None
    try:
        await exchange.load_markets(True) # Force reload
        if symbol in exchange.markets:
            market = exchange.markets[symbol]
            # Validate market data needed by strategy/bot
            if not market or not market.get('precision') or not market.get('limits'):
                 log.error(f"Market data for {symbol} is incomplete.")
                 return None
            log.info(f"Market data loaded/updated for {symbol}.")
            log.debug(f"Market Details ({symbol}): {json.dumps(market, indent=2)}")
            return market
        else:
            log.error(f"Symbol {symbol} not found in loaded markets for {exchange.name}.")
            return None
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        log.error(f"Failed to load market data for {symbol}: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error loading market data: {e}", exc_info=True)
        return None

async def fetch_initial_data(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches historical OHLCV data using CCXT."""
    if not exchange: return None
    log.info(f"Fetching initial {limit} candles for {symbol} ({timeframe})...")
    try:
        # CCXT fetch_ohlcv returns [[timestamp, open, high, low, close, volume]]
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            log.warning(f"Received empty list from fetch_ohlcv for {symbol}, {timeframe}.")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.astype(float) # Ensure all are floats
        # Check for NaNs introduced? Should not happen with fetch_ohlcv usually
        if df.isnull().values.any():
            log.warning("NaN values found in fetched OHLCV data. Filling or dropping might be needed.")
            # df = df.fillna(method='ffill') # Example: forward fill
        log.info(f"Fetched {len(df)} initial candles. From {df.index.min()} to {df.index.max()}")
        return df
    except ccxt.NetworkError as e:
        log.error(f"Network error fetching initial klines: {e}")
        return None
    except ccxt.ExchangeError as e:
         log.error(f"Exchange error fetching initial klines: {e}")
         return None
    except Exception as e:
        log.error(f"Unexpected error fetching initial klines: {e}", exc_info=True)
        return None

async def get_current_position(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches and updates the current position state asynchronously."""
    global current_position, last_position_check_time
    if not exchange or not market: return None # Need market for parsing

    # Rate limit REST checks
    now = time.monotonic()
    check_interval = config.get('checks', {}).get('position_check_interval', 30)
    if now - last_position_check_time < check_interval:
        #log.debug("Position check skipped (rate limit).")
        return current_position # Return last known state

    log.debug(f"Fetching position for {symbol} via REST...")
    async with position_lock: # Lock to prevent concurrent updates
        try:
            # fetch_position might return a list, need to find the right one
            positions = await exchange.fetch_position(symbol) # Use fetch_position for single symbol

            # CCXT position structure varies slightly by exchange. Need robust parsing.
            # Common fields: 'info', 'symbol', 'contracts' (or 'contractSize'), 'side', 'entryPrice', 'leverage'
            # Bybit V5 specific parsing might be needed if unified API has quirks
            if positions:
                 # Assuming fetch_position returns the direct dict for Bybit V5 linear when symbol is specified
                pos = positions # Use the direct result if not a list
                size = Decimal(pos.get('contracts', pos.get('contractSize', '0')) or '0')
                side = pos.get('side', 'none').lower() # 'long', 'short', or 'none'
                entry_price = Decimal(pos.get('entryPrice', '0') or '0')

                # Update global state
                current_position = {
                    "size": size,
                    "side": "Buy" if side == 'long' else "Sell" if side == 'short' else "None",
                    "entry_price": entry_price,
                    "timestamp": time.time() # Add timestamp of check
                }
                log.debug(f"Fetched Position: Size={size}, Side={side}, Entry={entry_price}")

            else: # No position returned
                 current_position = {"size": Decimal(0), "side": "None", "entry_price": Decimal(0), "timestamp": time.time()}
                 log.debug(f"No position found for {symbol}.")

            last_position_check_time = now # Update time on successful check
            return current_position

        except ccxt.NetworkError as e:
            log.warning(f"Network error fetching position: {e}. Returning last known state.")
            return current_position # Return last known state on temporary network error
        except ccxt.ExchangeError as e:
            log.error(f"Exchange error fetching position: {e}. Assuming flat.")
            # Dangerous to assume flat on exchange error, could lead to wrong trades.
            # Maybe stop or retry heavily? For now, assume flat and log error.
            current_position = {"size": Decimal(0), "side": "None", "entry_price": Decimal(0), "timestamp": time.time()}
            return current_position
        except Exception as e:
            log.error(f"Unexpected error fetching position: {e}", exc_info=True)
            # Return last known state on unexpected errors
            return current_position

async def get_wallet_balance(quote_currency: str = "USDT") -> Optional[Decimal]:
    """Fetches available equity/balance."""
    if not exchange: return None
    try:
        balance_data = await exchange.fetch_balance()
        # Parsing balance data depends heavily on exchange and account type (spot, margin, futures)
        # For Bybit Unified V5 ('contract' or 'unified' type): Look in total/free/used for the quote currency
        quote_balance = balance_data.get(quote_currency, {})
        total_equity = Decimal(balance_data.get('total', {}).get(quote_currency, '0')) # Try total first
        free_balance = Decimal(quote_balance.get('free', '0'))

        # Use total equity as the basis for risk calculation if available
        if total_equity > 0:
            log.debug(f"Using total equity for balance: {total_equity} {quote_currency}")
            return total_equity
        elif free_balance > 0:
             log.warning(f"Total equity not found or zero, using free balance: {free_balance} {quote_currency}")
             return free_balance
        else:
             # Look in info structure for Bybit specific fields if needed
             bybit_info = balance_data.get('info', {}).get('result', {}).get('list', [{}])[0]
             equity_info = Decimal(bybit_info.get('equity', '0'))
             if equity_info > 0:
                  log.warning(f"Using 'equity' from Bybit info structure: {equity_info}")
                  return equity_info

             log.error(f"Could not determine suitable balance/equity for {quote_currency}.")
             return Decimal(0) # Return 0 if no balance found

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        log.error(f"Could not fetch wallet balance: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error fetching balance: {e}", exc_info=True)
        return None

async def calculate_order_qty(entry_price: float, sl_price: float, risk_percent: float) -> Optional[float]:
    """Calculates order quantity based on risk, SL distance, and equity."""
    if not market or not strategy_instance: return None
    sl_decimal = Decimal(str(sl_price))
    entry_decimal = Decimal(str(entry_price))

    if abs(sl_decimal - entry_decimal) < strategy_instance.price_tick:
        log.error(f"SL price {sl_price} too close to entry price {entry_price}.")
        return None

    balance = await get_wallet_balance(market['quote'])
    if balance is None or balance <= 0:
        log.error(f"Cannot calculate order quantity: Invalid balance ({balance}).")
        return None

    try:
        risk_amount = balance * (Decimal(str(risk_percent)) / 100)
        sl_distance = abs(entry_decimal - sl_decimal)

        if sl_distance == 0: raise ZeroDivisionError("SL distance is zero")

        # Qty (in Base Asset, e.g., BTC for BTC/USDT) = Risk Amount (in Quote) / SL_Distance (in Quote)
        qty_base = risk_amount / sl_distance

    except (InvalidOperation, ZeroDivisionError, TypeError) as e:
        log.error(f"Error during quantity calculation math: {e}")
        return None

    # Round down to the minimum quantity step/tick size
    qty_rounded = strategy_instance.round_amount(float(qty_base))

    # Check against min/max order quantity from market limits
    min_qty = float(market.get('limits', {}).get('amount', {}).get('min', 0))
    max_qty = float(market.get('limits', {}).get('amount', {}).get('max', float('inf')))

    if qty_rounded < min_qty:
        log.warning(f"Calculated qty {qty_rounded} below minimum ({min_qty}). Risking more.")
        qty_final = min_qty
        actual_risk_amount = Decimal(str(min_qty)) * sl_distance
        actual_risk_percent = (actual_risk_amount / balance) * 100 if balance > 0 else 0
        log.warning(f"Using min qty {min_qty}. Actual Risk: {actual_risk_amount:.2f} {market['quote']} ({actual_risk_percent:.2f}%)")
    elif qty_rounded > max_qty:
        log.warning(f"Calculated qty {qty_rounded} exceeds maximum ({max_qty}). Using max.")
        qty_final = max_qty
    else:
        qty_final = qty_rounded

    if qty_final <= 0:
        log.error(f"Final calculated quantity is zero or negative ({qty_final}).")
        return None

    log.info(f"Calculated Order Qty: {qty_final:.{strategy_instance.amount_precision}f} "
             f"(Balance={balance:.2f}, Risk={risk_percent}%, RiskAmt={risk_amount:.2f}, SLDist={sl_distance:.{strategy_instance.price_precision}f})")
    return qty_final

async def place_order(symbol: str, side: str, qty: float, price: Optional[float]=None, sl_price: Optional[float]=None, tp_price: Optional[float]=None) -> Optional[Dict]:
    """Places an order using CCXT create_order with SL/TP params."""
    if not exchange or not strategy_instance or not market: return None
    if config.get("mode", "Live").lower() == "paper":
        log.warning(f"[PAPER MODE] Would place {side} {config['order']['type']} order: {qty} {symbol} @{price} SL={sl_price} TP={tp_price}")
        return {"id": f"paper_{int(time.time())}", "info": {"paperTrade": True}} # Simulate success

    async with order_lock: # Prevent concurrent order placements
        order_type = config['order']['type'].lower()
        amount = strategy_instance.round_amount(qty)
        limit_price = strategy_instance.round_price(price) if price and order_type == 'limit' else None

        if amount <= 0:
             log.error(f"Attempted to place order with zero/negative amount: {amount}")
             return None
        min_qty = float(market.get('limits', {}).get('amount', {}).get('min', 0))
        if amount < min_qty:
             log.warning(f"Order amount {amount} below min {min_qty}. Using min.")
             amount = min_qty


        params = {
             # Bybit V5 specific parameters for SL/TP attached to Market/Limit orders
             'positionIdx': 0, # Required for One-Way mode hedge=False
        }
        if sl_price:
            sl_price_rounded = strategy_instance.round_price(sl_price)
            trigger_direction = 2 if side.lower() == 'buy' else 1 # 1: Sell trigger, 2: Buy trigger
            params.update({
                'stopLossPrice': sl_price_rounded,
                'slTriggerDirection': trigger_direction,
                'slOrderType': 'Market', # Or 'Limit'
                'slTriggerBy': config['order'].get('sl_trigger_type', 'LastPrice')
            })
            log.info(f"Prepared SL: Price={sl_price_rounded}, Trigger={params['slTriggerBy']}")

        if tp_price:
             tp_price_rounded = strategy_instance.round_price(tp_price)
             trigger_direction = 1 if side.lower() == 'buy' else 2 # Opposite of SL
             params.update({
                 'takeProfitPrice': tp_price_rounded,
                 'tpTriggerDirection': trigger_direction,
                 'tpOrderType': 'Market', # Or 'Limit'
                 'tpTriggerBy': config['order'].get('tp_trigger_type', 'LastPrice')
             })
             log.info(f"Prepared TP: Price={tp_price_rounded}, Trigger={params['tpTriggerBy']}")

        log.warning(f"Placing {side.upper()} {order_type.upper()} order: {amount} {symbol} "
                    f"{'@'+str(limit_price) if limit_price else ''} "
                    f"SL={params.get('stopLossPrice', 'N/A')} TP={params.get('takeProfitPrice', 'N/A')}")

        try:
            order = await exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=limit_price, # None for market orders
                params=params
            )
            log.info(f"{Fore.GREEN}Order placed successfully! CCXT ID: {order.get('id')}{Style.RESET_ALL}")
            log.debug(f"Order details: {order}")
            # Update position state quickly after placing order (optional, REST check will confirm later)
            # await get_current_position(symbol) # Can potentially call this to refresh state sooner
            return order
        except ccxt.InsufficientFunds as e:
            log.error(f"{Fore.RED}Order failed: Insufficient funds. {e}{Style.RESET_ALL}")
            # Check leverage, balance, maybe reduce risk?
            return None
        except ccxt.InvalidOrder as e:
             log.error(f"{Fore.RED}Order failed: Invalid order parameters. Check config/calcs. {e}{Style.RESET_ALL}")
             log.error(f"Params sent: symbol={symbol}, type={order_type}, side={side}, amount={amount}, price={limit_price}, params={params}")
             return None
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            log.error(f"{Fore.RED}Order failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
            # Implement retry logic here if appropriate
            return None
        except Exception as e:
            log.error(f"Unexpected error placing order: {e}", exc_info=True)
            return None

async def close_position(symbol: str, position_data: Dict) -> Optional[Dict]:
    """Closes the position using a reduce-only market order."""
    if not exchange or not strategy_instance or not market: return None
    if config.get("mode", "Live").lower() == "paper":
        log.warning(f"[PAPER MODE] Would close position for {symbol} (Size: {position_data['size']}, Side: {position_data['side']})")
        return {"id": f"paper_close_{int(time.time())}", "info": {"paperTrade": True}} # Simulate success

    async with order_lock: # Ensure only one closing order attempt
        current_size = position_data.get('size', Decimal(0))
        current_side = position_data.get('side', 'None') # Expect 'Buy' or 'Sell'

        if current_size <= 0 or current_side == 'None':
            log.info(f"Attempted to close position for {symbol}, but it appears flat.")
            return {"info": {"alreadyFlat": True}}

        side_to_close = 'sell' if current_side == 'Buy' else 'buy'
        amount_to_close = float(current_size) # Use the fetched size

        log.warning(f"Closing {current_side} position for {symbol} (Size: {amount_to_close}). Placing {side_to_close.upper()} Market order...")

        params = {
            'reduceOnly': True,
             'positionIdx': 0 # Required for One-Way mode hedge=False
        }
        try:
            # Optional: Cancel existing SL/TP orders first if using separate orders
            # log.debug("Attempting to cancel associated SL/TP orders...")
            # try:
            #    await exchange.cancel_all_orders(symbol, params={'orderFilter': 'StopOrder'}) # Bybit specific? Check CCXT docs
            # except Exception as cancel_e:
            #    log.warning(f"Could not cancel SL/TP orders before closing: {cancel_e}")
            # await asyncio.sleep(0.5) # Short delay

            order = await exchange.create_order(
                symbol=symbol,
                type='market',
                side=side_to_close,
                amount=amount_to_close,
                params=params
            )
            log.info(f"{Fore.YELLOW}Position close order placed successfully! CCXT ID: {order.get('id')}{Style.RESET_ALL}")
            log.debug(f"Close Order details: {order}")
            # Force position check soon after closing attempt
            global last_position_check_time
            last_position_check_time = 0
            return order

        except ccxt.InvalidOrder as e:
             # Often happens if position already closed or size changed (reduceOnly error)
             log.warning(f"Close order failed (likely position closed/changed): {e}")
             last_position_check_time = 0 # Force check
             return {"info": {"error": str(e), "alreadyFlatOrChanged": True}}
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            log.error(f"{Fore.RED}Close order failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
            return None
        except Exception as e:
            log.error(f"Unexpected error closing position: {e}", exc_info=True)
            return None

async def set_leverage(symbol: str, leverage: int):
    """Sets leverage using CCXT (if supported by exchange)."""
    if not exchange or not market: return
    if not exchange.has.get('setLeverage'):
         log.warning(f"Exchange {exchange.name} does not support setting leverage via CCXT.")
         return

    # Validate leverage against market limits if available
    max_leverage = market.get('limits', {}).get('leverage', {}).get('max', 100)
    if not 1 <= leverage <= max_leverage:
         log.error(f"Invalid leverage {leverage}. Must be between 1 and {max_leverage} for {symbol}.")
         return

    log.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
    try:
        # Note: Some exchanges require setting leverage for buy/sell separately or per position mode.
        # CCXT's set_leverage tries to abstract this. Check Bybit specifics if needed.
        await exchange.set_leverage(leverage, symbol)
        log.info(f"Leverage for {symbol} set to {leverage}x request sent (confirmation may vary).")
        # Verify leverage (requires fetching position data)
        # pos = await get_current_position(symbol)
        # if pos and int(pos.get('leverage', 0)) == leverage: log.info("Leverage confirmed.")

    except ccxt.ExchangeError as e:
         # Handle common errors like "leverage not modified"
         if "not modified" in str(e).lower() or "110044" in str(e): # Bybit code
              log.warning(f"Leverage for {symbol} already set to {leverage}x (Not modified).")
         else:
              log.error(f"Failed to set leverage for {symbol}: {e}")
    except Exception as e:
        log.error(f"Unexpected error setting leverage: {e}", exc_info=True)


# --- WebSocket Watcher Loops ---
async def watch_kline_loop(symbol: str, timeframe: str):
    """Watches for new OHLCV candles via WebSocket."""
    global latest_dataframe, last_ws_update_time
    if not exchange: return
    log.info(f"Starting Kline watcher for {symbol} ({timeframe})...")
    while not stop_event.is_set():
        try:
            candles = await exchange.watch_ohlcv(symbol, timeframe)
            # watch_ohlcv usually returns list of lists [[ts, o, h, l, c, v]]
            if not candles: continue

            last_ws_update_time = time.monotonic() # Update health check timestamp

            # Process the *last* candle received from the batch (most recent closed one)
            # Note: Some exchanges might push updates for the *current* candle.
            # CCXT's watch_ohlcv aims to return *closed* candles, but behavior can vary.
            # We only want confirmed/closed candles for strategy analysis.
            # Assumption: The last candle in the list is the most recently closed one.
            # Need verification based on exchange behavior.
            last_candle_data = candles[-1]
            ts_ms, o, h, l, c, v = last_candle_data
            ts = pd.to_datetime(ts_ms, unit='ms')

            new_data = {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
            log.debug(f"WS Candle Received: T={ts}, O={o}, H={h}, L={l}, C={c}, V={v}")

            # --- Process the Confirmed Candle ---
            await process_candle(ts, new_data) # Await the processing

        except ccxt.NetworkError as e:
            log.warning(f"Kline Watcher Network Error: {e}. Reconnecting...")
            await asyncio.sleep(5) # Wait before retrying
        except ccxt.ExchangeError as e:
            log.warning(f"Kline Watcher Exchange Error: {e}. Retrying...")
            await asyncio.sleep(5)
        except asyncio.CancelledError:
             log.info("Kline watcher task cancelled.")
             break # Exit loop cleanly on cancellation
        except Exception as e:
            log.error(f"Unexpected error in Kline watcher: {e}", exc_info=True)
            await asyncio.sleep(10) # Longer sleep on unexpected errors
    log.info("Kline watcher loop finished.")

async def process_candle(timestamp: pd.Timestamp, data: Dict):
    """Adds candle data to DataFrame and triggers analysis."""
    global latest_dataframe
    async with data_lock: # Ensure exclusive access to the dataframe
        if latest_dataframe is None:
            log.warning("DataFrame not ready, skipping candle processing.")
            return

        # Check if timestamp already exists (e.g., from initial fetch overlap or duplicate WS msg)
        if timestamp in latest_dataframe.index:
             log.debug(f"Candle {timestamp} already exists in DataFrame. Updating...")
             latest_dataframe.loc[timestamp, list(data.keys())] = list(data.values())
        else:
             log.debug(f"Adding new candle {timestamp} to DataFrame.")
             new_row = pd.DataFrame([data], index=[timestamp])
             latest_dataframe = pd.concat([latest_dataframe, new_row])
             # Prune old data
             max_len = config['data']['max_df_len']
             if len(latest_dataframe) > max_len:
                 latest_dataframe = latest_dataframe.iloc[-max_len:]
                 # log.debug(f"DataFrame pruned to {len(latest_dataframe)} rows.")

        # --- Trigger Strategy Analysis ---
        if strategy_instance:
            log.info(f"Running analysis on confirmed candle: {timestamp}")
            df_copy = latest_dataframe.copy() # Analyze a copy
            try:
                analysis_results = strategy_instance.update(df_copy)
                # Process signals asynchronously without blocking the candle processing
                asyncio.create_task(process_signals(analysis_results))
            except Exception as e:
                log.error(f"Error during strategy analysis: {e}", exc_info=True)
        else:
             log.warning("Strategy instance not available for analysis.")


async def watch_positions_loop(symbol: str):
    """(Optional) Watches for position updates via WebSocket."""
    global current_position, last_ws_update_time
    if not exchange or not exchange.has.get('watchPositions'):
        log.info("Position watcher skipped: Exchange does not support watchPositions.")
        return

    log.info(f"Starting Position watcher for {symbol}...")
    while not stop_event.is_set():
        try:
            positions = await exchange.watch_positions([symbol]) # Watch specific symbol
            last_ws_update_time = time.monotonic()
            if not positions: continue

            # watch_positions often returns a list, find our symbol
            pos_update = None
            for p in positions:
                if p.get('symbol') == symbol:
                     pos_update = p
                     break

            if pos_update:
                log.info(f"{Fore.CYAN}Position update via WS: {pos_update}{Style.RESET_ALL}")
                # Update internal state cautiously - REST check is still source of truth before ordering
                async with position_lock:
                    size = Decimal(pos_update.get('contracts', pos_update.get('contractSize', '0')) or '0')
                    side = pos_update.get('side', 'none').lower()
                    entry_price = Decimal(pos_update.get('entryPrice', '0') or '0')
                    # Only update if significantly different? Or just store? Let REST check confirm.
                    # current_position = { "size": size, ... } # Potentially update here
                    log.debug(f"WS Position Update: Size={size}, Side={side}, Entry={entry_price}")
                    # Force REST check on next periodic run if significant change detected?
                    # if abs(size - current_position['size']) > 0: last_position_check_time = 0
            else:
                 log.debug(f"Received position update via WS, but not for {symbol}.")


        except ccxt.NetworkError as e: log.warning(f"Position Watcher Network Error: {e}. Reconnecting..."); await asyncio.sleep(5)
        except ccxt.ExchangeError as e: log.warning(f"Position Watcher Exchange Error: {e}. Retrying..."); await asyncio.sleep(5)
        except asyncio.CancelledError: log.info("Position watcher task cancelled."); break
        except Exception as e: log.error(f"Unexpected error in Position watcher: {e}", exc_info=True); await asyncio.sleep(10)
    log.info("Position watcher loop finished.")

async def watch_orders_loop(symbol: str):
    """(Optional) Watches for order updates via WebSocket."""
    global last_ws_update_time
    if not exchange or not exchange.has.get('watchOrders'):
        log.info("Order watcher skipped: Exchange does not support watchOrders.")
        return

    log.info(f"Starting Order watcher for {symbol}...")
    while not stop_event.is_set():
        try:
            orders = await exchange.watch_orders(symbol)
            last_ws_update_time = time.monotonic()
            if not orders: continue

            for order_update in orders:
                 # Process order updates - e.g., log fills, SL/TP triggers
                 status = order_update.get('status')
                 order_id = order_update.get('id')
                 filled = order_update.get('filled')
                 price = order_update.get('average', order_update.get('price'))
                 log.info(f"{Fore.CYAN}Order Update via WS [{symbol}]: ID={order_id}, Status={status}, Filled={filled}, AvgPrice={price}{Style.RESET_ALL}")
                 if status == 'closed' and filled > 0:
                      log.warning(f"{Fore.GREEN}Order {order_id} FILLED/CLOSED via WS.{Style.RESET_ALL}")
                      # Trigger position check maybe?
                      # global last_position_check_time
                      # last_position_check_time = 0
                 elif status == 'canceled':
                      log.warning(f"{Fore.YELLOW}Order {order_id} CANCELED via WS.{Style.RESET_ALL}")


        except ccxt.NetworkError as e: log.warning(f"Order Watcher Network Error: {e}. Reconnecting..."); await asyncio.sleep(5)
        except ccxt.ExchangeError as e: log.warning(f"Order Watcher Exchange Error: {e}. Retrying..."); await asyncio.sleep(5)
        except asyncio.CancelledError: log.info("Order watcher task cancelled."); break
        except Exception as e: log.error(f"Unexpected error in Order watcher: {e}", exc_info=True); await asyncio.sleep(10)
    log.info("Order watcher loop finished.")

# --- Signal Processing & Execution (Async) ---
async def process_signals(results: AnalysisResults):
    """Processes strategy signals and executes trades asynchronously."""
    if not results or not strategy_instance or not market:
        log.warning("Signal processing skipped: Missing data/setup.")
        return
    if stop_event.is_set():
        log.warning("Signal processing skipped: Stop event set.")
        return

    signal = results['last_signal']
    last_close = results['last_close']
    last_atr = results['last_atr']
    symbol = config['symbol']

    log.debug(f"Processing Signal: {signal}, Last Close: {last_close}, Last ATR: {last_atr}")

    if pd.isna(last_close):
        log.warning("Cannot process signal: Last close price is NaN.")
        return

    # --- Get Current Position State ---
    # Fetch position state *before* making decisions based on the signal
    pos_data = await get_current_position(symbol) # Use latest fetched state
    if not pos_data: # Indicates fetch error or still rate limited
        log.warning("Could not get reliable position data. Skipping signal action.")
        return

    current_pos_size = pos_data.get('size', Decimal(0))
    current_pos_side = pos_data.get('side', 'None') # 'Buy', 'Sell', 'None'
    is_long = current_pos_side == 'Buy' and current_pos_size > 0
    is_short = current_pos_side == 'Sell' and current_pos_size > 0
    is_flat = not is_long and not is_short

    log.info(f"Processing signal '{signal}' | Current Position: {'Long' if is_long else 'Short' if is_short else 'Flat'} (Size: {current_pos_size})")

    # --- Execute Actions based on Signal and State ---
    tp_ratio = Decimal(str(config['order'].get('tp_ratio', 2.0)))

    # BUY Signal
    if signal == "BUY" and is_flat:
        log.warning(f"{Fore.GREEN}{Style.BRIGHT}BUY Signal - Attempting to Enter Long.{Style.RESET_ALL}")
        sl_price_raw = None
        sl_method = config['strategy']['stop_loss']['method']
        if sl_method == "ATR" and last_atr:
            sl_multiplier = float(config['strategy']['stop_loss']['atr_multiplier'])
            sl_price_raw = last_close - (last_atr * sl_multiplier)
        elif sl_method == "OB":
             relevant_obs = [b for b in results['active_bull_boxes'] if b['bottom'] < last_close]
             if relevant_obs:
                  lowest_bottom = min(b['bottom'] for b in relevant_obs)
                  sl_buffer = (last_atr * 0.1) if last_atr else (last_close * 0.001)
                  sl_price_raw = lowest_bottom - sl_buffer
             else:
                  log.warning("OB SL method: No relevant Bull OB found. Falling back to ATR.")
                  if last_atr: sl_price_raw = last_close - (last_atr * 2.0)

        if sl_price_raw is None or sl_price_raw >= last_close:
             log.error(f"Invalid SL price for BUY ({sl_price_raw}). Order cancelled.")
             return
        sl_price = strategy_instance.round_price(sl_price_raw)

        sl_distance = Decimal(str(last_close)) - Decimal(str(sl_price))
        tp_price_raw = float(Decimal(str(last_close)) + (sl_distance * tp_ratio)) if sl_distance > 0 else None
        tp_price = strategy_instance.round_price(tp_price_raw) if tp_price_raw else None

        qty = await calculate_order_qty(last_close, sl_price, config['order']['risk_per_trade_percent'])
        if qty and qty > 0:
            await place_order(symbol, 'buy', qty, price=last_close if config['order']['type'] == 'limit' else None, sl_price=sl_price, tp_price=tp_price)
        else: log.error("BUY order cancelled: Qty calculation failed.")

    # SELL Signal
    elif signal == "SELL" and is_flat:
        log.warning(f"{Fore.RED}{Style.BRIGHT}SELL Signal - Attempting to Enter Short.{Style.RESET_ALL}")
        sl_price_raw = None
        sl_method = config['strategy']['stop_loss']['method']
        if sl_method == "ATR" and last_atr:
             sl_multiplier = float(config['strategy']['stop_loss']['atr_multiplier'])
             sl_price_raw = last_close + (last_atr * sl_multiplier)
        elif sl_method == "OB":
            relevant_obs = [b for b in results['active_bear_boxes'] if b['top'] > last_close]
            if relevant_obs:
                  highest_top = max(b['top'] for b in relevant_obs)
                  sl_buffer = (last_atr * 0.1) if last_atr else (last_close * 0.001)
                  sl_price_raw = highest_top + sl_buffer
            else:
                  log.warning("OB SL method: No relevant Bear OB found. Falling back to ATR.")
                  if last_atr: sl_price_raw = last_close + (last_atr * 2.0)

        if sl_price_raw is None or sl_price_raw <= last_close:
            log.error(f"Invalid SL price for SELL ({sl_price_raw}). Order cancelled.")
            return
        sl_price = strategy_instance.round_price(sl_price_raw)

        sl_distance = Decimal(str(sl_price)) - Decimal(str(last_close))
        tp_price_raw = float(Decimal(str(last_close)) - (sl_distance * tp_ratio)) if sl_distance > 0 else None
        tp_price = strategy_instance.round_price(tp_price_raw) if tp_price_raw else None

        qty = await calculate_order_qty(last_close, sl_price, config['order']['risk_per_trade_percent'])
        if qty and qty > 0:
            await place_order(symbol, 'sell', qty, price=last_close if config['order']['type'] == 'limit' else None, sl_price=sl_price, tp_price=tp_price)
        else: log.error("SELL order cancelled: Qty calculation failed.")

    # EXIT_LONG Signal
    elif signal == "EXIT_LONG" and is_long:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_LONG Signal - Attempting to Close Long Position.{Style.RESET_ALL}")
        await close_position(symbol, pos_data)

    # EXIT_SHORT Signal
    elif signal == "EXIT_SHORT" and is_short:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_SHORT Signal - Attempting to Close Short Position.{Style.RESET_ALL}")
        await close_position(symbol, pos_data)

    # No action needed signals
    elif signal == "HOLD": log.debug("HOLD Signal - No action.")
    elif signal == "BUY" and is_long: log.debug("BUY Signal - Already Long.")
    elif signal == "SELL" and is_short: log.debug("SELL Signal - Already Short.")
    elif signal == "EXIT_LONG" and not is_long: log.debug("EXIT_LONG Signal - Not Long.")
    elif signal == "EXIT_SHORT" and not is_short: log.debug("EXIT_SHORT Signal - Not Short.")


# --- Periodic Health Check ---
async def periodic_check_loop():
    """Runs periodic checks for WebSocket health and stale positions."""
    global last_health_check_time, last_position_check_time
    check_interval = config.get('checks', {}).get('health_check_interval', 60)
    pos_check_interval = config.get('checks', {}).get('position_check_interval', 30)
    ws_timeout = check_interval * 2 # Consider WS dead if no update for this long

    log.info(f"Starting Periodic Check loop (Interval: {check_interval}s)")
    while not stop_event.is_set():
        try:
            await asyncio.sleep(check_interval)
            now = time.monotonic()
            log.debug(f"Running periodic checks (Time: {now})...")
            last_health_check_time = now

            # 1. Check WebSocket Health (based on last message time)
            time_since_last_ws = now - last_ws_update_time if last_ws_update_time > 0 else ws_timeout
            if time_since_last_ws >= ws_timeout:
                 log.warning(f"No WebSocket update received for {time_since_last_ws:.1f}s. Health check failed!")
                 # Consider attempting to restart WS connection? More complex logic needed here.
                 # For now, just log it. A dead WS thread check exists in main loop.
            else:
                 log.debug(f"WebSocket health check OK (last update {time_since_last_ws:.1f}s ago).")

            # 2. Force Position Check if Stale (double check against WS updates)
            time_since_last_pos_check = now - last_position_check_time if last_position_check_time > 0 else pos_check_interval
            if time_since_last_pos_check >= pos_check_interval:
                 log.info("Periodic check forcing position update...")
                 await get_current_position(config['symbol'])

            # Add other checks: balance low, API errors etc.

        except asyncio.CancelledError:
            log.info("Periodic check task cancelled.")
            break
        except Exception as e:
            log.error(f"Error in periodic check loop: {e}", exc_info=True)
            await asyncio.sleep(check_interval) # Wait before retrying after error

    log.info("Periodic check loop finished.")


# --- Graceful Shutdown ---
async def shutdown(signal_type):
    """Cleans up resources and exits."""
    log.warning(f"Shutdown initiated by signal {signal_type.name}...")
    stop_event.set() # Signal all loops to stop

    # Cancel all running asyncio tasks
    tasks_to_cancel = list(running_tasks)
    if tasks_to_cancel:
        log.info(f"Cancelling {len(tasks_to_cancel)} running tasks...")
        for task in tasks_to_cancel:
            task.cancel()
        # Wait for tasks to finish cancelling
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        log.info("Tasks cancelled.")

    # Close the CCXT exchange connection
    if exchange and hasattr(exchange, 'close'):
        log.info("Closing CCXT exchange connection...")
        try:
            await exchange.close()
            log.info("Exchange connection closed.")
        except Exception as e:
            log.error(f"Error closing exchange connection: {e}", exc_info=True)

    # Optional: Close open positions on exit? (Use with extreme caution)
    # await close_open_position_on_exit() # Implement this function carefully if needed

    log.warning("Shutdown complete. Exiting.")
    # Allow logs to flush
    await asyncio.sleep(0.5)
    sys.exit(0)

async def close_open_position_on_exit():
     """Placeholder for logic to close position during shutdown."""
     log.warning("Attempting to check/close position on exit (USE WITH CAUTION)...")
     # Need to re-initialize exchange or ensure it's still valid? Risky.
     # Best practice is usually to let existing SL/TP handle it or manage manually.
     # If implementing: fetch position one last time and call close_position.
     pass


# --- Main Application ---
async def main():
    """Main async function to run the bot."""
    global config, exchange, market, strategy_instance, latest_dataframe, running_tasks

    print(Fore.MAGENTA + Style.BRIGHT + "\n~~~ Pyrmethus Volumatic+OB Trading Bot Starting (CCXT Async) ~~~")

    config = load_config()
    setup_logging(config.get("log_level", "INFO"))
    log.info("Configuration loaded.")
    log.debug(f"Config: {json.dumps(config, indent=2)}")

    # Connect to exchange
    exchange = await connect_ccxt()
    if not exchange: sys.exit(1)

    # Load market info
    market = await load_exchange_market(config['symbol'])
    if not market: sys.exit(1)

    # Set leverage (if applicable)
    await set_leverage(config['symbol'], config['order']['leverage'])

    # Initialize Strategy
    try:
        strategy_instance = VolumaticOBStrategy(market=market, **config['strategy']['params'])
    except Exception as e:
        log.critical(f"Failed to initialize strategy: {e}. Check config.json.", exc_info=True)
        await exchange.close() # Close connection before exiting
        sys.exit(1)

    # Fetch Initial Data
    async with data_lock: # Lock dataframe during initial fetch
        latest_dataframe = await fetch_initial_data(
            config['symbol'],
            config['timeframe'],
            config['data']['fetch_limit']
        )
        if latest_dataframe is None: # Indicates fetch error
            log.critical("Failed to fetch initial market data. Exiting.")
            await exchange.close()
            sys.exit(1)
        elif len(latest_dataframe) < strategy_instance.min_data_len:
            log.warning(f"Initial data ({len(latest_dataframe)}) insufficient for strategy ({strategy_instance.min_data_len}). Waiting for WS data.")
        else:
            # Run initial analysis if enough data
            log.info("Running initial analysis on historical data...")
            try:
                initial_results = strategy_instance.update(latest_dataframe.copy())
                log.info(f"Initial Analysis Results: Trend={initial_results['current_trend']}, Signal={initial_results['last_signal']}")
            except Exception as e:
                 log.error(f"Error during initial analysis: {e}", exc_info=True)


    # Fetch initial position state
    await get_current_position(config['symbol'])
    log.info(f"Initial Position: {current_position['side']} (Size: {current_position['size']})")

    log.info(f"{Fore.CYAN}Setup complete. Starting WebSocket watchers and main loops...{Style.RESET_ALL}")
    log.info(f"Trading Mode: {config.get('mode', 'Live')}")
    log.info(f"Symbol: {config['symbol']} | Timeframe: {config['timeframe']}")

    # Start background tasks
    tasks = [
        asyncio.create_task(watch_kline_loop(config['symbol'], config['timeframe'])),
        # Optional watchers (uncomment if needed):
        # asyncio.create_task(watch_positions_loop(config['symbol'])),
        # asyncio.create_task(watch_orders_loop(config['symbol'])),
        asyncio.create_task(periodic_check_loop()),
    ]
    running_tasks.update(tasks) # Add tasks to the global set for cancellation

    # Keep main running, tasks execute in background
    # await asyncio.gather(*tasks) # This would block until *all* tasks finish (or one errors)

    # Instead, keep main alive and let shutdown handler cancel tasks
    while not stop_event.is_set():
         # Check if essential tasks (like kline watcher) are still running
         kline_task = tasks[0] # Assuming kline is first
         if kline_task.done():
              log.critical("Kline watcher task has terminated unexpectedly!")
              try:
                   kline_task.result() # Raise exception if task failed
              except Exception as e:
                   log.critical(f"Kline watcher failed with error: {e}", exc_info=True)
              log.critical("Attempting to stop bot gracefully due to essential task failure...")
              # Signal shutdown without waiting for OS signal
              await shutdown(signal.SIGTERM) # Simulate signal
              break # Exit main loop

         await asyncio.sleep(10) # Heartbeat sleep for main loop


if __name__ == "__main__":
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    try:
        loop.run_until_complete(main())
    except asyncio.CancelledError:
         log.info("Main task cancelled during shutdown.")
    finally:
         # Ensure exchange connection is closed even if main loop exits unexpectedly
         if exchange and hasattr(exchange, 'close') and not exchange.is_closed:
             loop.run_until_complete(exchange.close())
         log.info("Application finished.")

EOF

echo -e "${GREEN}All files created successfully in ${YELLOW}${PROJECT_DIR}${NC}"
echo -e "\n${YELLOW}--- Next Steps ---${NC}"
echo -e "1. ${YELLOW}Edit the ${BLUE}${ENV_FILE}${YELLOW} file and add your actual Bybit API Key and Secret.${NC}"
echo -e "   ${RED}IMPORTANT: Start with Testnet keys (BYBIT_TESTNET=\"True\")!${NC}"
echo -e "2. Review ${YELLOW}${CONFIG_FILE}${NC} carefully:"
echo -e "   - Set your desired ${BLUE}symbol${NC} (e.g., BTC/USDT) and ${BLUE}timeframe${NC} (e.g., 5m)."
echo -e "   - Adjust ${BLUE}risk_per_trade_percent${NC}, ${BLUE}leverage${NC}, ${BLUE}tp_ratio${NC}."
echo -e "   - Configure strategy parameters under ${BLUE}strategy.params${NC}."
echo -e "   - Ensure ${BLUE}account_type${NC} matches your Bybit account (usually 'contract' for linear perps)."
echo -e "3. Install required Python packages:"
echo -e "   ${GREEN}pip install -r ${REQ_FILE}${NC}"
echo -e "4. Run the bot:"
echo -e "   ${GREEN}python ${MAIN_FILE}${NC}"
echo -e "\n${BLUE}Bot setup complete! This is an asynchronous bot. Remember to test thoroughly on Testnet before using real funds.${NC}"

# Make the setup script non-executable after running (optional)
# chmod -x "$SETUP_SCRIPT_NAME" # Commented out, might want to re-run

exit 0