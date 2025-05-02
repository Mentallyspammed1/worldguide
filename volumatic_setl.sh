#!/bin/bash
# Strict mode
set -euo pipefail

# Script to set up the Pyrmethus Volumatic Trend + OB Bot directory and files (CCXT Async Version)

# --- Configuration ---
PROJECT_DIR="pyrmethus_volumatic_bot_ccxt"
ENV_FILE=".env"
CONFIG_FILE="config.json"
STRATEGY_FILE="strategy.py"
MAIN_FILE="main.py"
REQ_FILE="requirements.txt"
# SETUP_SCRIPT_NAME="$(basename "$0")" # Get script name dynamically if needed

# --- Colors for output ---
# Check if stdout is a terminal and define colors, otherwise leave them empty
if [ -t 1 ]; then
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  RED='\033[0;31m'
  NC='\033[0m' # No Color
else
  GREEN=""
  YELLOW=""
  BLUE=""
  RED=""
  NC=""
fi

# --- Helper Functions ---
info() {
    echo -e "${BLUE}INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}WARN: $1${NC}"
}

error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

# --- Main Script Logic ---
info "Setting up Pyrmethus Volumatic Bot (CCXT Async Version) in: ${YELLOW}${PROJECT_DIR}${NC}"

# Check if project directory exists and contains key files
if [ -d "$PROJECT_DIR" ]; then
    warn "Directory ${YELLOW}${PROJECT_DIR}${NC} already exists."
    # Check for potential overwrites of key files
    for file in "$ENV_FILE" "$CONFIG_FILE" "$STRATEGY_FILE" "$MAIN_FILE" "$REQ_FILE"; do
        if [ -e "${PROJECT_DIR}/${file}" ]; then
            error_exit "File ${YELLOW}${PROJECT_DIR}/${file}${NC} already exists. Aborting to prevent overwrite."
        fi
    done
    info "Directory exists but key files not found. Proceeding to create files inside."
else
    # Create project directory if it doesn't exist
    info "Creating project directory: ${YELLOW}${PROJECT_DIR}${NC}"
    mkdir -p "$PROJECT_DIR"
fi

# Change into the project directory; exit if failed
cd "$PROJECT_DIR" || error_exit "Could not change directory to ${PROJECT_DIR}"

info "Successfully changed directory to ${YELLOW}$(pwd)${NC}"

# --- Create requirements.txt ---
info "Creating ${YELLOW}${REQ_FILE}${NC}..."
cat << EOF > "$REQ_FILE"
# Use specific versions known to work or use >= for latest compatible
ccxt>=4.1.60 # Check CCXT releases for latest stable Bybit V5 & asyncio support
python-dotenv>=0.20.0
pandas>=1.4.0
pandas-ta>=0.3.14b # For technical analysis indicators
numpy>=1.21.0
colorama>=0.4.4
requests>=2.27.0 # Often useful for auxiliary API calls or health checks
# asyncio is built-in Python 3.7+
EOF

# --- Create .env file ---
info "Creating ${YELLOW}${ENV_FILE}${NC} (Remember to fill in your API keys!)"
cat << EOF > "$ENV_FILE"
# Bybit API Credentials
# Create API keys here: https://www.bybit.com/app/user/api-management
# Ensure keys have permissions for: Contract Trade (or Unified Trade), Read/Write Orders & Positions
BYBIT_API_KEY="YOUR_API_KEY_HERE"
BYBIT_API_SECRET="YOUR_API_SECRET_HERE"

# Set to "True" for Testnet, "False" or leave empty for Mainnet
# Testnet URL: https://testnet.bybit.com
BYBIT_TESTNET="True"
EOF

# --- Create config.json ---
info "Creating ${YELLOW}${CONFIG_FILE}${NC}..."
cat << EOF > "$CONFIG_FILE"
{
  "exchange": "bybit",
  "symbol": "BTC/USDT:USDT", // Use CCXT unified symbol format (e.g., BASE/QUOTE:SETTLE)
  "timeframe": "5m", // CCXT standard timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)
  "account_type": "contract", // 'contract' (for USDT/USDC perps), 'unified', or 'spot'. Check CCXT/Bybit docs.
  "mode": "Paper", // Start with "Paper" or "Testnet". Change to "Live" for real trading.
  "log_level": "INFO", // Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL

  "order": {
    "type": "Market", // "Market" or "Limit". Limit orders need careful implementation.
    "risk_per_trade_percent": 1.0, // Percentage of equity to risk per trade (e.g., 1.0 = 1%)
    "leverage": 5, // Desired leverage. Ensure it's allowed and set on Bybit (script attempts to set it).
    "tp_ratio": 2.0, // Take Profit Risk:Reward ratio (e.g., 2.0 means TP is 2x SL distance from entry)
    "sl_trigger_type": "LastPrice", // Bybit trigger price type: LastPrice, IndexPrice, MarkPrice
    "tp_trigger_type": "LastPrice"  // Bybit trigger price type: LastPrice, IndexPrice, MarkPrice
  },

  "strategy": {
    "class": "VolumaticOBStrategy", // Must match the class name in strategy.py
    "params": {
      "length": 40,           // Trend calculation period
      "vol_atr_period": 200,    // ATR period for volume normalization/levels
      "vol_percentile_len": 1000, // Lookback for volume percentile calculation
      "vol_percentile": 95,   // Volume percentile threshold (1-100)
      "ob_source": "Wicks",     // "Wicks" or "Bodys" for Order Block definition
      "pivot_left_h": 10,       // Left lookback bars for pivot high
      "pivot_right_h": 10,      // Right lookback bars for pivot high
      "pivot_left_l": 10,       // Left lookback bars for pivot low
      "pivot_right_l": 10,      // Right lookback bars for pivot low
      "max_boxes": 5            // Max number of *active* OBs of each type to track
    },
    "stop_loss": {
      "method": "ATR", // "ATR" or "OB" (uses Order Block boundary)
      "atr_multiplier": 1.5 // ATR multiplier if method is "ATR"
    }
  },

  "data": {
      "fetch_limit": 750,   // Number of candles for initial historical fetch
      "max_df_len": 2000    // Maximum number of candles to keep in the DataFrame in memory
  },

  "checks": {
      "position_check_interval": 30, // How often to fetch position via REST API (seconds)
      "health_check_interval": 60    // How often to run general health checks (seconds)
  }
}
EOF

# --- Create strategy.py ---
info "Creating ${YELLOW}${STRATEGY_FILE}${NC}..."
# Use 'EOF' to prevent shell expansion within the Python code
cat << 'EOF' > "$STRATEGY_FILE"
# -*- coding: utf-8 -*-
"""
Strategy Definition: Enhanced Volumatic Trend + OB (for CCXT Async Bot)
"""
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style

log = logging.getLogger(__name__) # Gets logger configured in main.py
getcontext().prec = 18 # Set decimal precision for calculations

# --- Type Definitions (consistent with main.py) ---
class OrderBlock(TypedDict):
    id: int # Index of the candle where the OB formed (relative to current df)
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
    last_atr: Optional[float] # Last calculated ATR value

# --- Strategy Class ---
class VolumaticOBStrategy:
    """
    Encapsulates the logic from the 'Enhanced Volumatic Trend + OB' Pine Script.
    Calculates indicators and generates trading signals based on OHLCV data.
    Uses CCXT market structure for precision handling.
    """
    def __init__(self, market: Dict[str, Any], **params: Any):
        """
        Initializes the strategy.
        Args:
            market: CCXT market structure dictionary.
            **params: Strategy parameters loaded from config.json['strategy']['params'].
        """
        self.market = market
        self._parse_params(params)
        self._validate_params()

        # State Variables - initialized to None or empty
        self.upper: Optional[float] = None
        self.lower: Optional[float] = None
        self.lower_vol: Optional[float] = None
        self.upper_vol: Optional[float] = None
        self.step_up: Optional[float] = None
        self.step_dn: Optional[float] = None
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []
        # last_signal_state tracks the *intended* state ('BUY' means trying to be long, 'SELL' means trying to be short)
        # This helps prevent duplicate entry signals if the condition persists.
        self.last_signal_state = "HOLD" # Internal state: HOLD, BUY (want long), SELL (want short)
        self.current_trend: Optional[bool] = None # True=UP, False=DOWN

        # Calculate minimum required data length based on largest period needed
        self.min_data_len = max(
            self.length + 4, # For _ema_swma shift and calculation
            self.vol_atr_period,
            self.vol_percentile_len,
            self.pivot_left_h + self.pivot_right_h + 1, # Lookback + current + lookforward
            self.pivot_left_l + self.pivot_right_l + 1
        )

        # Extract precision details from CCXT market structure
        try:
            self.price_precision = int(market.get('precision', {}).get('price', 8)) # Default 8 if not found
            self.amount_precision = int(market.get('precision', {}).get('amount', 8)) # Default 8
            # Use Decimal for tick sizes to avoid float inaccuracies
            self.price_tick = Decimal(str(market.get('precision', {}).get('price', '0.00000001')))
            self.amount_tick = Decimal(str(market.get('precision', {}).get('amount', '0.00000001')))
            if self.price_tick <= 0 or self.amount_tick <= 0:
                raise ValueError("Market precision ticks must be positive.")
        except (ValueError, TypeError) as e:
            log.error(f"Could not parse market precision: {e}. Using defaults.")
            self.price_precision = 8
            self.amount_precision = 8
            self.price_tick = Decimal('0.00000001')
            self.amount_tick = Decimal('0.00000001')


        log.info(f"{Fore.MAGENTA}Initializing VolumaticOB Strategy Engine...{Style.RESET_ALL}")
        log.info(f"Symbol: {market.get('symbol', 'N/A')}")
        log.info(f"Params: TrendLen={self.length}, Pivots={self.pivot_left_h}/{self.pivot_right_h}, "
                 f"MaxBoxes={self.max_boxes}, OB Source={self.ob_source}")
        log.info(f"Minimum data points required: {self.min_data_len}")
        log.debug(f"Price Precision: {self.price_precision}, Amount Precision: {self.amount_precision}")
        log.debug(f"Price Tick: {self.price_tick}, Amount Tick: {self.amount_tick}")


    def _parse_params(self, params: Dict[str, Any]):
        """Load and type-cast parameters from the config dict."""
        try:
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
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid strategy parameter type in config: {e}") from e

    def _validate_params(self):
        """Basic validation of strategy parameters."""
        if self.ob_source not in ["Wicks", "Bodys"]:
            raise ValueError("Invalid 'ob_source' parameter. Must be 'Wicks' or 'Bodys'.")
        lengths = [self.length, self.vol_atr_period, self.vol_percentile_len,
                   self.pivot_left_h, self.pivot_right_h, self.pivot_left_l, self.pivot_right_l]
        if not all(isinstance(x, int) and x > 0 for x in lengths):
             raise ValueError("All length/period parameters must be positive integers.")
        if not isinstance(self.max_boxes, int) or self.max_boxes <= 0:
             raise ValueError("'max_boxes' must be a positive integer.")
        if not 0 < self.vol_percentile <= 100:
             raise ValueError("'vol_percentile' must be between 1 and 100 (exclusive of 0).")

    # --- Precision Handling using CCXT market data ---
    def format_price(self, price: float) -> str:
        """Formats price according to market precision (for display/logging)."""
        return f"{Decimal(str(price)):.{self.price_precision}f}"

    def format_amount(self, amount: float) -> str:
        """Formats amount according to market precision (for display/logging)."""
        return f"{Decimal(str(amount)):.{self.amount_precision}f}"

    def round_price(self, price: float) -> float:
        """Rounds price UP or DOWN to the nearest price tick."""
        price_decimal = Decimal(str(price))
        # Determine rounding direction based on typical usage (e.g., SL slightly further, TP slightly closer)
        # For simplicity here, just rounding to nearest tick. Adjust if needed.
        # Using ROUND_HALF_UP as a general case, but specific cases might need ROUND_UP or ROUND_DOWN.
        # For SL buy: round down. For SL sell: round up. For TP buy: round down. For TP sell: round up.
        # Let's stick to a simple rounding for now, order placement logic can refine.
        rounded = (price_decimal / self.price_tick).quantize(Decimal('1'), rounding=ROUND_UP if price_decimal > 0 else ROUND_DOWN) * self.price_tick
        return float(rounded)

    def round_amount(self, amount: float) -> float:
        """Rounds amount DOWN to the nearest amount tick (conservative)."""
        amount_decimal = Decimal(str(amount))
        # Always round amount down to avoid insufficient funds or exceeding limits
        rounded = (amount_decimal / self.amount_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * self.amount_tick
        return float(rounded)
    # --- End Precision Handling ---


    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Helper for smoothed EMA (SWMA(4) -> EMA(length))."""
        if len(series) < length + 4: # Need enough data for rolling window and EMA
            return pd.Series(np.nan, index=series.index)

        # Calculate 4-period Smoothed Moving Average (equivalent to LWMA with weights [1,2,2,1]/6)
        weights = np.array([1, 2, 2, 1]) / 6.0
        # Apply rolling weighted average. Ensure min_periods=4.
        swma = series.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights) if len(x)==4 else np.nan, raw=True)

        # Calculate EMA of the SWMA
        ema_of_swma = ta.ema(swma, length=length, fillna=False) # Use fillna=False initially
        return ema_of_swma.fillna(np.nan) # Explicitly handle NaNs

    def _find_pivots(self, df: pd.DataFrame, left: int, right: int, is_high: bool) -> pd.Series:
        """
        Finds pivot points similar to Pine Script's ta.pivothigh/low.
        Args:
            df: DataFrame containing price data.
            left: Number of bars to the left.
            right: Number of bars to the right.
            is_high: True to find pivot highs, False for pivot lows.
        Returns:
            A pandas Series with pivot prices at the pivot bar index, NaN otherwise.
        """
        if self.ob_source == "Wicks":
             source_col = 'high' if is_high else 'low'
        else: # Bodys
             # Use close for high pivot, open for low pivot (common interpretation)
             source_col = 'close' if is_high else 'open'

        if source_col not in df.columns:
            log.error(f"Source column '{source_col}' not found for pivot calculation.")
            return pd.Series(np.nan, index=df.index)

        source_series = df[source_col]
        pivots = pd.Series(np.nan, index=df.index)

        # Iterate through the series where a full window (left + current + right) is available
        for i in range(left, len(df) - right):
            pivot_val = source_series.iloc[i]
            if pd.isna(pivot_val): continue # Skip if pivot candidate value is NaN

            is_pivot = True
            # Check left bars
            for j in range(1, left + 1):
                left_val = source_series.iloc[i - j]
                if pd.isna(left_val): continue # Skip comparison if left value is NaN
                # For high pivot: left value must be strictly less than pivot value
                # For low pivot: left value must be strictly greater than pivot value
                if (is_high and left_val >= pivot_val) or \
                   (not is_high and left_val <= pivot_val):
                    is_pivot = False; break
            if not is_pivot: continue

            # Check right bars
            for j in range(1, right + 1):
                right_val = source_series.iloc[i + j]
                if pd.isna(right_val): continue # Skip comparison if right value is NaN
                # For high pivot: right value must be less than or equal to pivot value
                # For low pivot: right value must be greater than or equal to pivot value
                if (is_high and right_val > pivot_val) or \
                   (not is_high and right_val < pivot_val):
                    is_pivot = False; break

            if is_pivot:
                pivots.iloc[i] = pivot_val # Store the pivot value at the pivot index
        return pivots

    def update(self, df_input: pd.DataFrame) -> AnalysisResults:
        """
        Calculates indicators and generates signals based on the input DataFrame.
        Args:
            df_input: pandas DataFrame with OHLCV columns and DatetimeIndex.
                      Assumes the last row is the most recent *closed* candle.
        Returns:
            AnalysisResults dictionary containing calculated data and signals.
        """
        if df_input.empty or len(df_input) < self.min_data_len:
            log.warning(f"Not enough data ({len(df_input)}/{self.min_data_len}) for analysis.")
            # Return default state if not enough data
            return AnalysisResults(
                dataframe=df_input, last_signal="HOLD", active_bull_boxes=self.bull_boxes,
                active_bear_boxes=self.bear_boxes, last_close=np.nan, current_trend=self.current_trend,
                trend_changed=False, last_atr=None
            )

        df = df_input.copy()
        log.debug(f"Analyzing {len(df)} candles ending {df.index[-1]}...")

        # --- Volumatic Trend Calculations ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.vol_atr_period, fillna=False)
        df['ema1'] = self._ema_swma(df['close'], length=self.length) # Smoothed EMA
        df['ema2'] = ta.ema(df['close'], length=self.length, fillna=False) # Regular EMA

        # Determine trend based on EMA comparison
        # Use np.where for vectorized comparison, handle NaNs
        df['trend_up'] = np.where(df['ema1'] < df['ema2'], True,
                         np.where(df['ema1'] >= df['ema2'], False, np.nan))
        # Forward fill the trend to handle initial NaNs if EMAs haven't converged
        df['trend_up'] = df['trend_up'].ffill()

        # Detect trend changes
        df['trend_changed'] = (df['trend_up'] != df['trend_up'].shift(1)) & df['trend_up'].notna() & df['trend_up'].shift(1).notna()

        # Get values from the last (most recent closed) candle
        last_row = df.iloc[-1]
        current_trend_up = last_row['trend_up']
        trend_just_changed = last_row['trend_changed']
        last_atr_value = last_row['atr'] if pd.notna(last_row['atr']) else None

        # Update dynamic levels only when trend changes or initially
        if pd.notna(current_trend_up):
            # Update if trend is identified for the first time OR if it changed on this candle
            if self.current_trend is None or (trend_just_changed and current_trend_up != self.current_trend):
                is_initial_trend = self.current_trend is None
                self.current_trend = current_trend_up # Update strategy's trend state
                log.info(f"{Fore.MAGENTA}{'Initial Trend Detected' if is_initial_trend else 'Trend Changed'}! New Trend: {'UP' if current_trend_up else 'DOWN'}. Updating levels...{Style.RESET_ALL}")

                # Recalculate levels based on the candle where trend changed
                # Use the EMA1 and ATR from the candle *before* the change for stability? Or the current one?
                # Let's use the current candle's values for simplicity.
                current_ema1 = last_row['ema1']
                current_atr = last_row['atr']
                if pd.notna(current_ema1) and pd.notna(current_atr) and current_atr > 1e-9: # Ensure ATR is valid
                    atr_mult = 3.0 # Multiplier for main bands
                    vol_atr_mult = 4.0 # Multiplier for vol bands relative to EMA1
                    self.upper = current_ema1 + current_atr * atr_mult
                    self.lower = current_ema1 - current_atr * atr_mult
                    # Vol bands are offset from the *opposite* main band
                    self.lower_vol = self.lower + current_atr * vol_atr_mult
                    self.upper_vol = self.upper - current_atr * vol_atr_mult
                    # Ensure vol bands don't cross main bands
                    if self.lower_vol < self.lower: self.lower_vol = self.lower
                    if self.upper_vol > self.upper: self.upper_vol = self.upper
                    # Calculate step size (used in original Pine for dynamic adjustments, less critical here)
                    self.step_up = (self.lower_vol - self.lower) / 100 if self.lower_vol > self.lower else 0
                    self.step_dn = (self.upper - self.upper_vol) / 100 if self.upper > self.upper_vol else 0
                    log.info(f"Levels Updated @ {df.index[-1]}: U={self.format_price(self.upper)}, L={self.format_price(self.lower)}")
                else:
                     log.warning(f"Could not update levels @ {df.index[-1]} due to NaN/zero EMA1/ATR. Levels reset.")
                     self.upper, self.lower, self.lower_vol, self.upper_vol, self.step_up, self.step_dn = [None] * 6

        # --- Volume Normalization (Optional - not directly used in OB/Trend signals here) ---
        # Calculate rolling volume percentile for context if needed elsewhere
        roll_window = min(self.vol_percentile_len, len(df))
        min_periods_vol = max(1, min(roll_window // 2, 50)) # Require some data but not full window initially
        if roll_window > 0:
            df['vol_percentile_val'] = df['volume'].rolling(window=roll_window, min_periods=min_periods_vol).apply(
                lambda x: np.nanpercentile(x, self.vol_percentile) if np.any(~np.isnan(x) & (x > 0)) else np.nan, raw=True)

            df['vol_norm'] = np.where(
                (df['vol_percentile_val'].notna()) & (df['vol_percentile_val'] > 1e-9),
                (df['volume'] / df['vol_percentile_val'] * 100), 0 # Normalize volume against percentile value
            ).fillna(0).astype(float)
        else:
            df['vol_percentile_val'] = np.nan
            df['vol_norm'] = 0.0


        # --- Pivot Order Block Calculations ---
        df['ph'] = self._find_pivots(df, self.pivot_left_h, self.pivot_right_h, is_high=True)
        df['pl'] = self._find_pivots(df, self.pivot_left_l, self.pivot_right_l, is_high=False)

        # --- Create and Manage Order Blocks ---
        # Only need to check recent bars for *new* pivots that could form OBs
        # Check bars from (current - right_lookback) index for pivots
        check_start_idx = max(0, len(df) - max(self.pivot_right_h, self.pivot_right_l) - 5) # Check a bit further back just in case
        new_boxes_created_count = 0
        df['int_index'] = range(len(df)) # Use integer index for OB tracking

        # Iterate through potential pivot locations in the recent part of the dataframe
        for i in range(check_start_idx, len(df)):
            # Check for Bearish Box from Pivot High
            # Pivot occurs at index 'i', detected after 'pivot_right_h' bars pass
            pivot_occur_int_idx = i
            if pd.notna(df['ph'].iloc[pivot_occur_int_idx]):
                # Check if this pivot index already has a bear box
                if not any(b['id'] == pivot_occur_int_idx for b in self.bear_boxes):
                    ob_candle = df.iloc[pivot_occur_int_idx]
                    # Define OB boundaries based on source
                    if self.ob_source == "Wicks":
                        top_price = ob_candle['high']
                        bottom_price = ob_candle['close'] # Bear OB: High to Close of up candle before down move
                    else: # Bodys
                        top_price = ob_candle['close']
                        bottom_price = ob_candle['open'] # Bear OB: Body of up candle

                    if pd.notna(top_price) and pd.notna(bottom_price):
                        # Ensure top > bottom
                        if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price
                        # Check if OB has a meaningful range (greater than ~tick size)
                        if Decimal(str(abs(top_price - bottom_price))) > self.price_tick / 10:
                            new_box = OrderBlock(id=pivot_occur_int_idx, type='bear', left_idx=pivot_occur_int_idx,
                                                 right_idx=len(df)-1, # Initially valid until current bar
                                                 top=top_price, bottom=bottom_price, active=True, closed_idx=None)
                            self.bear_boxes.append(new_box)
                            new_boxes_created_count += 1
                            log.info(f"{Fore.RED}New Bear OB {pivot_occur_int_idx} @ {df.index[pivot_occur_int_idx]}: T={self.format_price(top_price)}, B={self.format_price(bottom_price)}{Style.RESET_ALL}")

            # Check for Bullish Box from Pivot Low
            pivot_occur_int_idx = i
            if pd.notna(df['pl'].iloc[pivot_occur_int_idx]):
                 # Check if this pivot index already has a bull box
                 if not any(b['id'] == pivot_occur_int_idx for b in self.bull_boxes):
                    ob_candle = df.iloc[pivot_occur_int_idx]
                    # Define OB boundaries based on source
                    if self.ob_source == "Wicks":
                        top_price = ob_candle['open'] # Bull OB: Open to Low of down candle before up move
                        bottom_price = ob_candle['low']
                    else: # Bodys
                        top_price = ob_candle['open']
                        bottom_price = ob_candle['close'] # Bull OB: Body of down candle

                    if pd.notna(top_price) and pd.notna(bottom_price):
                        # Ensure top > bottom
                        if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price
                        # Check if OB has a meaningful range
                        if Decimal(str(abs(top_price - bottom_price))) > self.price_tick / 10:
                            new_box = OrderBlock(id=pivot_occur_int_idx, type='bull', left_idx=pivot_occur_int_idx,
                                                 right_idx=len(df)-1, # Initially valid until current bar
                                                 top=top_price, bottom=bottom_price, active=True, closed_idx=None)
                            self.bull_boxes.append(new_box)
                            new_boxes_created_count += 1
                            log.info(f"{Fore.GREEN}New Bull OB {pivot_occur_int_idx} @ {df.index[pivot_occur_int_idx]}: T={self.format_price(top_price)}, B={self.format_price(bottom_price)}{Style.RESET_ALL}")

        # Manage existing boxes based on the latest close price
        current_close = last_row['close']
        current_bar_int_idx = len(df) - 1
        if pd.notna(current_close):
            closed_bull_count, closed_bear_count = 0, 0
            # Check Bull Boxes: Invalidate if close goes below the bottom
            for box in self.bull_boxes:
                if box['active']:
                    if current_close < box['bottom']:
                        box['active'] = False
                        box['closed_idx'] = current_bar_int_idx
                        closed_bull_count += 1
                    else:
                        # Keep extending the validity (right_idx) if still active
                        box['right_idx'] = current_bar_int_idx
            # Check Bear Boxes: Invalidate if close goes above the top
            for box in self.bear_boxes:
                if box['active']:
                    if current_close > box['top']:
                        box['active'] = False
                        box['closed_idx'] = current_bar_int_idx
                        closed_bear_count += 1
                    else:
                        # Keep extending the validity (right_idx) if still active
                        box['right_idx'] = current_bar_int_idx

            if closed_bull_count: log.info(f"{Fore.YELLOW}Closed {closed_bull_count} Bull OB(s) due to price breaking below.{Style.RESET_ALL}")
            if closed_bear_count: log.info(f"{Fore.YELLOW}Closed {closed_bear_count} Bear OB(s) due to price breaking above.{Style.RESET_ALL}")

        # Prune Order Blocks to keep memory usage reasonable
        # Keep only the most recent 'max_boxes' active boxes and maybe some recent inactive ones for history
        active_bull = sorted([b for b in self.bull_boxes if b['active']], key=lambda x: x['id'], reverse=True)
        inactive_bull = sorted([b for b in self.bull_boxes if not b['active']], key=lambda x: x['id'], reverse=True)
        self.bull_boxes = active_bull[:self.max_boxes] + inactive_bull[:self.max_boxes * 2] # Keep more inactive for context

        active_bear = sorted([b for b in self.bear_boxes if b['active']], key=lambda x: x['id'], reverse=True)
        inactive_bear = sorted([b for b in self.bear_boxes if not b['active']], key=lambda x: x['id'], reverse=True)
        self.bear_boxes = active_bear[:self.max_boxes] + inactive_bear[:self.max_boxes * 2]

        # --- Signal Generation ---
        signal = "HOLD"
        active_bull_boxes = [b for b in self.bull_boxes if b['active']] # Use pruned list
        active_bear_boxes = [b for b in self.bear_boxes if b['active']]

        # Only generate signals if trend is defined and close price is valid
        if self.current_trend is not None and pd.notna(current_close):
            # Priority 1: Exit signal if trend just changed against the desired position state
            if trend_just_changed:
                if not self.current_trend and self.last_signal_state == "BUY": # Trend changed DOWN while wanting LONG
                    signal = "EXIT_LONG"
                elif self.current_trend and self.last_signal_state == "SELL": # Trend changed UP while wanting SHORT
                    signal = "EXIT_SHORT"

            # Priority 2: Entry signals if trend allows and not already in desired state
            if signal == "HOLD": # Only check for entry if no exit signal was generated
                if self.current_trend: # Trend is UP -> Look for Long Entry
                    if self.last_signal_state != "BUY": # Only enter if not already trying to be long
                        # Check if price entered any active Bullish OB
                        for box in active_bull_boxes:
                            # Entry condition: Current close is within the Bull OB range
                            if box['bottom'] <= current_close <= box['top']:
                                signal = "BUY"
                                break # Take the first valid OB entry
                elif not self.current_trend: # Trend is DOWN -> Look for Short Entry
                    if self.last_signal_state != "SELL": # Only enter if not already trying to be short
                        # Check if price entered any active Bearish OB
                        for box in active_bear_boxes:
                            # Entry condition: Current close is within the Bear OB range
                            if box['bottom'] <= current_close <= box['top']:
                                signal = "SELL"
                                break # Take the first valid OB entry

        # Update internal state based on the generated signal
        if signal == "BUY":
            self.last_signal_state = "BUY"
        elif signal == "SELL":
            self.last_signal_state = "SELL"
        elif signal in ["EXIT_LONG", "EXIT_SHORT"]:
            self.last_signal_state = "HOLD" # Reset state after exit

        # Log the generated signal if it's not HOLD
        if signal != "HOLD":
             color = Fore.YELLOW if "EXIT" in signal else (Fore.GREEN if signal == "BUY" else Fore.RED)
             trend_str = 'UP' if self.current_trend else 'DOWN' if self.current_trend is not None else 'UNDETERMINED'
             log.warning(f"{color}{Style.BRIGHT}*** {signal} Signal generated at {self.format_price(current_close)} (Trend: {trend_str}) ***{Style.RESET_ALL}")
        else:
            log.debug(f"Signal: HOLD (Internal State: {self.last_signal_state}, Trend: {'UP' if self.current_trend else 'DOWN' if self.current_trend is not None else 'UNDETERMINED'})")


        df.drop(columns=['int_index'], inplace=True, errors='ignore') # Cleanup temporary index column

        # Return all relevant results
        return AnalysisResults(
            dataframe=df, # Return the updated dataframe with indicators
            last_signal=signal,
            active_bull_boxes=active_bull_boxes, # Return currently active boxes
            active_bear_boxes=active_bear_boxes,
            last_close=current_close,
            current_trend=self.current_trend,
            trend_changed=trend_just_changed,
            last_atr=last_atr_value
        )
EOF

# --- Create main.py ---
info "Creating ${YELLOW}${MAIN_FILE}${NC}..."
# Use 'EOF' to prevent shell expansion within the Python code
cat << 'EOF' > "$MAIN_FILE"
# -*- coding: utf-8 -*-
"""
Pyrmethus Volumatic Trend + OB Trading Bot (CCXT Async Version - Enhanced)

Disclaimer: Trading involves risk. This bot is provided for educational purposes
and demonstration. Use at your own risk. Test thoroughly on paper/testnet before
using real funds. Ensure you understand the strategy and code.
"""
import asyncio
import datetime
import json
import logging
import os
import signal
import sys
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from typing import Any, Dict, Optional, Set

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Import strategy class and type hints
from strategy import AnalysisResults, VolumaticOBStrategy

# --- Initialize Colorama & Decimal Precision ---
init(autoreset=True)
getcontext().prec = 28 # Set high decimal precision for calculations

# --- Load Environment Variables ---
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
# Read testnet flag, default to False if not set or invalid
TESTNET_STR = os.getenv("BYBIT_TESTNET", "False").lower()
TESTNET = TESTNET_STR == "true"

# --- Global Variables ---
config: Dict[str, Any] = {}
exchange: Optional[ccxt.Exchange] = None # Type hint for CCXT exchange instance
strategy_instance: Optional[VolumaticOBStrategy] = None
market: Optional[Dict[str, Any]] = None # CCXT market structure
latest_dataframe: Optional[pd.DataFrame] = None # Holds OHLCV data
# Store position info using Decimal for precision
current_position: Dict[str, Any] = {"size": Decimal(0), "side": "None", "entry_price": Decimal(0), "timestamp": 0.0}
last_position_check_time: float = 0.0 # Track REST API calls
last_health_check_time: float = 0.0
last_ws_update_time: float = 0.0 # Track WebSocket health
running_tasks: Set[asyncio.Task] = set() # Store background tasks for cancellation
stop_event = asyncio.Event() # Event to signal shutdown

# --- Locks for Shared Resources ---
# Use asyncio locks to prevent race conditions in async operations
data_lock = asyncio.Lock() # Protects latest_dataframe
position_lock = asyncio.Lock() # Protects current_position and REST position fetches
order_lock = asyncio.Lock() # Protects order placement/cancellation

# --- Logging Setup ---
log = logging.getLogger("PyrmethusVolumaticBotCCXT")
# Default level, will be overridden by config
log_level = logging.INFO

def setup_logging(level_str: str = "INFO"):
    """Configures logging for the application."""
    global log_level
    try:
        log_level = getattr(logging, level_str.upper(), logging.INFO)
    except AttributeError:
        print(f"Warning: Invalid log level '{level_str}' in config. Using INFO.")
        log_level = logging.INFO

    log.setLevel(log_level)
    # Prevent adding multiple handlers if re-configured
    if not log.handlers:
        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        log.addHandler(ch)

        # Optional File Handler (configure path as needed)
        # log_filename = f"bot_ccxt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        # try:
        #     fh = logging.FileHandler(log_filename)
        #     fh.setLevel(log_level)
        #     fh.setFormatter(formatter)
        #     log.addHandler(fh)
        # except Exception as e:
        #     log.error(f"Failed to set up file logging: {e}")

    # Prevent log messages from propagating to the root logger
    log.propagate = False

# --- Configuration Loading ---
def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
            conf = json.load(f)
            # Basic validation for essential keys
            required_keys = ["exchange", "symbol", "timeframe", "order", "strategy", "data", "checks"]
            if not all(key in conf for key in required_keys):
                 raise ValueError("Config file missing required top-level keys: "
                                  f"exchange, symbol, timeframe, order, strategy, data, checks")
            # Validate nested keys if necessary
            if 'params' not in conf.get('strategy', {}):
                 raise ValueError("Config file missing 'strategy.params'.")
            if 'stop_loss' not in conf.get('strategy', {}):
                 raise ValueError("Config file missing 'strategy.stop_loss'.")
            if not all(k in conf.get('order', {}) for k in ['type', 'risk_per_trade_percent', 'leverage']):
                 raise ValueError("Config file missing required keys in 'order'.")

            log.info(f"Configuration loaded successfully from '{path}'.")
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
    global exchange # Allow modification of the global variable
    exchange_id = config.get('exchange', 'bybit').lower()
    account_type = config.get('account_type', 'contract')

    if not API_KEY or not API_SECRET:
        log.critical("CRITICAL: API Key or Secret not found in environment variables (.env file).")
        return None

    if not hasattr(ccxt, exchange_id):
        log.critical(f"CRITICAL: Exchange '{exchange_id}' is not supported by CCXT.")
        return None

    try:
        log.info(f"Connecting to CCXT exchange '{exchange_id}' (Account: {account_type}, Testnet: {TESTNET})...")
        exchange_config = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': account_type, # 'spot', 'margin', 'future', 'swap', 'contract', 'unified'
                'adjustForTimeDifference': True, # Auto-sync clock with server
                 # Bybit V5 specific options (check CCXT docs for updates)
                'accounts': {'unified': 'UNIFIED', 'contract': 'CONTRACT', 'spot': 'SPOT'}.get(account_type, 'CONTRACT'),
                'recvWindow': 10000, # Increase recvWindow if timestamp errors occur (default 5000)
            }
        }
        # Conditionally add Bybit specific options if needed
        # if exchange_id == 'bybit':
        #     exchange_config['options']['enableUnifiedMargin'] = (account_type == 'unified')
        #     exchange_config['options']['enableUnifiedAccount'] = (account_type == 'unified')

        exchange = getattr(ccxt, exchange_id)(exchange_config)

        if TESTNET:
            log.warning("Using Testnet mode.")
            exchange.set_sandbox_mode(True)

        # Test connection by loading markets (also fetches server time)
        await exchange.load_markets()
        log.info(f"{Fore.GREEN}Successfully connected to {exchange.name}. Loaded {len(exchange.markets)} markets.{Style.RESET_ALL}")
        return exchange
    except ccxt.AuthenticationError as e:
        log.critical(f"CRITICAL: CCXT Authentication Error: {e}. Check API keys, permissions, and ensure IP whitelist (if used) is correct.")
        return None
    except ccxt.NetworkError as e:
         log.critical(f"CRITICAL: CCXT Network Error connecting to exchange: {e}. Check internet connection and firewall.")
         return None
    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize CCXT exchange: {e}", exc_info=True)
        return None

async def load_exchange_market(symbol: str) -> Optional[Dict[str, Any]]:
    """Loads or re-loads market data for a specific symbol."""
    global market # Allow modification of the global variable
    if not exchange:
        log.error("Cannot load market, exchange not connected.")
        return None
    try:
        await exchange.load_markets(True) # Force reload to get latest info
        if symbol in exchange.markets:
            market = exchange.markets[symbol]
            # Validate essential market data
            if not market or not market.get('precision') or not market.get('limits'):
                 log.error(f"Market data for {symbol} is incomplete or missing precision/limits.")
                 return None
            if market.get('active') is False:
                 log.error(f"Market {symbol} is not active on the exchange.")
                 return None

            log.info(f"Market data loaded/updated for {symbol}.")
            # Log key details at DEBUG level
            log.debug(f"Market Details ({symbol}):\n"
                      f"  Precision: Price={market.get('precision',{}).get('price')}, Amount={market.get('precision',{}).get('amount')}\n"
                      f"  Limits: Amount(min={market.get('limits',{}).get('amount',{}).get('min')}, max={market.get('limits',{}).get('amount',{}).get('max')}), "
                      f"Cost(min={market.get('limits',{}).get('cost',{}).get('min')}), "
                      f"Leverage(max={market.get('limits',{}).get('leverage',{}).get('max')})\n"
                      f"  Type: {market.get('type')}, Contract: {market.get('contract')}")
            return market
        else:
            log.error(f"Symbol '{symbol}' not found in loaded markets for {exchange.name}.")
            available_symbols = list(exchange.markets.keys())
            log.error(f"Available symbols sample: {available_symbols[:10]}...") # Show a sample
            return None
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        log.error(f"Failed to load market data for {symbol}: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error loading market data for {symbol}: {e}", exc_info=True)
        return None

async def fetch_initial_data(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches historical OHLCV data using CCXT."""
    if not exchange:
        log.error("Cannot fetch data, exchange not connected.")
        return None
    log.info(f"Fetching initial {limit} candles for {symbol} ({timeframe})...")
    try:
        # CCXT fetch_ohlcv returns list: [[timestamp, open, high, low, close, volume]]
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            log.warning(f"Received empty list from fetch_ohlcv for {symbol}, {timeframe}. No initial data.")
            # Return an empty DataFrame with correct columns
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        # Ensure numeric types
        df = df.astype(float)

        # Check for NaNs which might indicate gaps or exchange issues
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            log.warning(f"NaN values found in fetched OHLCV data:\n{nan_counts[nan_counts > 0]}")
            # Option: Fill NaNs (e.g., forward fill) or drop rows, depending on strategy needs
            # df = df.ffill() # Example: Forward fill
            log.warning("Proceeding with NaN values. Strategy should handle them.")

        log.info(f"Fetched {len(df)} initial candles. From {df.index.min()} to {df.index.max()}")
        return df
    except ccxt.NetworkError as e:
        log.error(f"Network error fetching initial klines: {e}")
        return None
    except ccxt.ExchangeError as e:
         log.error(f"Exchange error fetching initial klines: {e}")
         # Check if the error is about the symbol/timeframe combination
         if "not supported" in str(e).lower():
             log.error(f"The symbol '{symbol}' or timeframe '{timeframe}' might not be supported by {exchange.name}.")
         return None
    except Exception as e:
        log.error(f"Unexpected error fetching initial klines: {e}", exc_info=True)
        return None

async def get_current_position(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches and updates the current position state asynchronously via REST API."""
    global current_position, last_position_check_time
    if not exchange or not market:
        log.warning("Cannot get position, exchange or market not ready.")
        return current_position # Return last known state

    # Rate limit REST checks to avoid hitting API limits
    now = time.monotonic()
    check_interval = config.get('checks', {}).get('position_check_interval', 30)
    if now - last_position_check_time < check_interval:
        # log.debug("Position check skipped (rate limit). Returning cached state.")
        return current_position # Return last known state

    log.debug(f"Fetching position for {symbol} via REST...")
    async with position_lock: # Lock to prevent concurrent updates from REST/WS
        try:
            # Use fetch_positions for Bybit V5 (even for single symbol) as fetch_position might be deprecated/different
            # Filter by the specific symbol if multiple positions are returned
            all_positions = await exchange.fetch_positions([symbol]) # Pass symbol in list

            pos_data = None
            if all_positions:
                # Find the position matching the exact symbol (e.g., 'BTC/USDT:USDT')
                for p in all_positions:
                    if p.get('symbol') == symbol:
                        pos_data = p
                        break

            if pos_data:
                # Parse position data carefully - structure varies! Use Decimal.
                # Common fields: 'contracts' (size in base), 'contractSize' (value of 1 contract),
                # 'side' ('long'/'short'), 'entryPrice', 'leverage', 'unrealizedPnl', 'initialMargin', etc.
                size_str = pos_data.get('contracts', '0') or '0' # Size in base currency (e.g., BTC)
                side = pos_data.get('side', 'none').lower() # 'long', 'short', or 'none'
                entry_price_str = pos_data.get('entryPrice', '0') or '0'

                # Safely convert to Decimal
                size = Decimal(size_str)
                entry_price = Decimal(entry_price_str)

                # Update global state
                current_position = {
                    "size": size,
                    "side": "Buy" if side == 'long' else "Sell" if side == 'short' else "None",
                    "entry_price": entry_price,
                    "timestamp": time.time() # Record time of successful check
                }
                log.debug(f"Fetched Position: Size={size}, Side={current_position['side']}, Entry={entry_price}")

            else: # No position found for the symbol
                 if current_position['size'] != Decimal(0):
                     log.info(f"Position for {symbol} now reported as flat (previously {current_position['side']} {current_position['size']}).")
                 else:
                     log.debug(f"No position found for {symbol}.")
                 current_position = {"size": Decimal(0), "side": "None", "entry_price": Decimal(0), "timestamp": time.time()}

            last_position_check_time = now # Update time only on successful check
            return current_position

        except ccxt.NetworkError as e:
            log.warning(f"Network error fetching position: {e}. Returning last known state.")
            # Return cached state on temporary network issues
            return current_position
        except ccxt.ExchangeError as e:
            log.error(f"Exchange error fetching position: {e}. Assuming flat (use with caution).")
            # This is risky. If the exchange error persists, we might take wrong actions.
            # Consider adding retry logic or stopping the bot if this happens frequently.
            current_position = {"size": Decimal(0), "side": "None", "entry_price": Decimal(0), "timestamp": time.time()}
            return current_position
        except Exception as e:
            log.error(f"Unexpected error fetching position: {e}", exc_info=True)
            # Return cached state on unexpected errors
            return current_position

async def get_wallet_balance(quote_currency: str = "USDT") -> Optional[Decimal]:
    """Fetches available equity/balance in the specified quote currency."""
    if not exchange: return None
    log.debug(f"Fetching wallet balance for {quote_currency}...")
    try:
        # fetch_balance structure depends heavily on exchange and account type
        balance_data = await exchange.fetch_balance()

        # --- Adapt parsing based on expected structure (e.g., Bybit Unified/Contract) ---
        # Option 1: Look directly in the top-level free/total for the currency
        total_equity = Decimal(balance_data.get('total', {}).get(quote_currency, '0'))
        free_balance = Decimal(balance_data.get('free', {}).get(quote_currency, '0'))

        # Option 2: Look in Bybit V5 specific 'info' structure if needed
        # Example path (might change): balance_data['info']['result']['list'][0]['equity']
        bybit_equity = Decimal(0)
        if 'info' in balance_data and isinstance(balance_data['info'], dict):
             result = balance_data['info'].get('result', {})
             if isinstance(result, dict) and 'list' in result and isinstance(result['list'], list) and len(result['list']) > 0:
                 account_info = result['list'][0]
                 if isinstance(account_info, dict):
                     bybit_equity = Decimal(account_info.get('equity', '0')) # Total equity for the account type

        # --- Determine which balance to use ---
        # Prefer total equity if available and positive, as it reflects margin use
        if bybit_equity > 0:
            log.debug(f"Using Bybit 'equity': {bybit_equity} {quote_currency}")
            return bybit_equity
        elif total_equity > 0:
            log.debug(f"Using CCXT 'total' balance: {total_equity} {quote_currency}")
            return total_equity
        elif free_balance > 0:
             # Using free balance might underestimate risk capacity if margin is used
             log.warning(f"Total equity not found or zero, using 'free' balance: {free_balance} {quote_currency}")
             return free_balance
        else:
             log.error(f"Could not determine a valid balance/equity for {quote_currency}. Found 0.")
             return Decimal(0) # Return 0 if no balance found

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        log.error(f"Could not fetch wallet balance: {e}")
        return None # Indicate failure to fetch
    except (InvalidOperation, TypeError) as e:
         log.error(f"Error converting balance data to Decimal: {e}")
         return None
    except Exception as e:
        log.error(f"Unexpected error fetching balance: {e}", exc_info=True)
        return None

async def calculate_order_qty(entry_price: float, sl_price: float, risk_percent: float) -> Optional[float]:
    """Calculates order quantity based on risk, SL distance, and equity."""
    if not market or not strategy_instance:
        log.error("Cannot calculate quantity: Market or strategy instance not available.")
        return None

    # Use Decimal for precise calculations
    sl_decimal = Decimal(str(sl_price))
    entry_decimal = Decimal(str(entry_price))
    risk_percent_decimal = Decimal(str(risk_percent))

    # Ensure SL is meaningfully different from entry
    if abs(sl_decimal - entry_decimal) < strategy_instance.price_tick:
        log.error(f"Stop loss price {sl_price} is too close to entry price {entry_price}. Cannot calculate quantity.")
        return None

    # Get current equity
    quote_currency = market.get('quote', 'USDT') # e.g., USDT in BTC/USDT
    balance = await get_wallet_balance(quote_currency)
    if balance is None or balance <= 0:
        log.error(f"Cannot calculate order quantity: Invalid or zero balance ({balance}) for {quote_currency}.")
        return None

    try:
        # Calculate risk amount in quote currency
        risk_amount = balance * (risk_percent_decimal / Decimal(100))
        # Calculate stop loss distance in quote currency per unit of base currency
        sl_distance_per_unit = abs(entry_decimal - sl_decimal)

        if sl_distance_per_unit <= 0:
            raise ValueError("Stop loss distance is zero or negative.")

        # Calculate quantity in base asset (e.g., BTC for BTC/USDT)
        # Qty (Base) = Risk Amount (Quote) / SL Distance per Unit (Quote/Base)
        qty_base = risk_amount / sl_distance_per_unit

    except (InvalidOperation, ValueError, ZeroDivisionError, TypeError) as e:
        log.error(f"Error during quantity calculation math: {e}")
        log.error(f"Inputs: balance={balance}, risk%={risk_percent_decimal}, entry={entry_decimal}, sl={sl_decimal}")
        return None

    # Round the calculated quantity DOWN to the market's amount precision/tick size
    qty_rounded = strategy_instance.round_amount(float(qty_base))

    # Check against market limits (min/max order size in base currency)
    min_qty = float(market.get('limits', {}).get('amount', {}).get('min', 0))
    max_qty = float(market.get('limits', {}).get('amount', {}).get('max', float('inf')))

    qty_final = qty_rounded

    if qty_final <= 0:
        log.error(f"Calculated order quantity is zero or negative ({qty_final}) after rounding. Min Qty: {min_qty}")
        # Maybe try using min_qty if risk allows? Or just fail. Let's fail for safety.
        return None

    if qty_final < min_qty:
        log.warning(f"Calculated qty {qty_final} is below market minimum ({min_qty}).")
        # Option 1: Use min_qty (increases risk)
        # Option 2: Abort trade
        # Let's choose Option 1 for now, but log the increased risk clearly.
        qty_final = min_qty
        actual_risk_amount = Decimal(str(min_qty)) * sl_distance_per_unit
        actual_risk_percent = (actual_risk_amount / balance) * 100 if balance > 0 else 0
        log.warning(f"Adjusting order quantity to minimum: {qty_final}. "
                    f"Actual Risk: {actual_risk_amount:.2f} {quote_currency} ({actual_risk_percent:.2f}%)")
    elif qty_final > max_qty:
        log.warning(f"Calculated qty {qty_final} exceeds market maximum ({max_qty}). Adjusting down.")
        qty_final = max_qty # Use max allowed quantity

    log.info(f"Calculated Order Qty: {strategy_instance.format_amount(qty_final)} {market.get('base', '')} "
             f"(Balance={balance:.2f}, Risk={risk_percent}%, RiskAmt={risk_amount:.2f}, SLDist={sl_distance_per_unit:.{strategy_instance.price_precision}f})")
    return qty_final

async def place_order(symbol: str, side: str, qty: float, price: Optional[float]=None, sl_price: Optional[float]=None, tp_price: Optional[float]=None) -> Optional[Dict]:
    """Places an order using CCXT create_order with SL/TP params if supported."""
    if not exchange or not strategy_instance or not market:
        log.error("Cannot place order: Exchange, strategy, or market not ready.")
        return None

    mode = config.get("mode", "Live").lower()
    if mode == "paper":
        # Simulate order placement in paper trading mode
        sl_str = f" SL={strategy_instance.format_price(sl_price)}" if sl_price else ""
        tp_str = f" TP={strategy_instance.format_price(tp_price)}" if tp_price else ""
        price_str = f" @{strategy_instance.format_price(price)}" if price and config['order']['type'].lower() == 'limit' else ""
        log.warning(f"[PAPER MODE] Simulating {side.upper()} {config['order']['type']} order: "
                    f"{strategy_instance.format_amount(qty)} {symbol}{price_str}{sl_str}{tp_str}")
        # Return a simulated order structure
        return {
            "id": f"paper_{int(time.time())}",
            "symbol": symbol,
            "side": side,
            "amount": qty,
            "price": price,
            "status": "closed", # Assume immediate fill for paper trading simplicity
            "filled": qty,
            "average": price if price else (await exchange.fetch_ticker(symbol)).get('last', 0), # Simulate fill price
            "info": {"paperTrade": True, "simulated": True}
        }

    # --- Live/Testnet Order Placement ---
    async with order_lock: # Prevent concurrent order placements
        order_type = config['order']['type'].lower() # 'market' or 'limit'
        amount = strategy_instance.round_amount(qty) # Round amount down
        limit_price = strategy_instance.round_price(price) if price and order_type == 'limit' else None

        # Final validation before placing order
        if amount <= 0:
             log.error(f"Attempted to place order with zero/negative amount after rounding: {amount}")
             return None
        min_qty = float(market.get('limits', {}).get('amount', {}).get('min', 0))
        if amount < min_qty:
             log.error(f"Final order amount {amount} is still below minimum {min_qty}. Aborting order.")
             # This can happen if min_qty was used in calculation but rounding brought it down again.
             # Or if min_qty itself is zero/negative in market data.
             return None

        # --- Prepare CCXT Params ---
        # Base parameters for create_order
        order_params = {}

        # Parameters for SL/TP (syntax varies significantly by exchange and API version)
        # Using Bybit V5 Unified/Contract syntax as an example:
        # SL/TP are often passed within the 'params' dictionary for create_order
        ccxt_params = {
             # 'positionIdx': 0, # 0: One-Way Mode, 1: Hedge Buy, 2: Hedge Sell. Required for Bybit hedge mode. Assume One-Way.
             # Check Bybit docs for current requirements (e.g., linear vs inverse, unified vs contract)
        }
        if sl_price:
            sl_price_rounded = strategy_instance.round_price(sl_price)
            # Determine trigger direction based on order side (1: sell trigger, 2: buy trigger)
            trigger_direction = 2 if side.lower() == 'buy' else 1 # If buying, SL is triggered by price falling (sell trigger direction)
            ccxt_params.update({
                'stopLoss': str(sl_price_rounded), # Use string representation for price
                # 'slTriggerBy': config['order'].get('sl_trigger_type', 'LastPrice'), # MarkPrice, IndexPrice, LastPrice
                # 'tpslMode': 'Full', # Partial or Full position TP/SL
                # 'slOrderType': 'Market', # Bybit might require specifying SL order type
            })
            # CCXT often abstracts trigger types, but check specific exchange needs
            log.info(f"Prepared SL: Price={sl_price_rounded}") # Trigger type might be implicit or set elsewhere

        if tp_price:
             tp_price_rounded = strategy_instance.round_price(tp_price)
             trigger_direction = 1 if side.lower() == 'buy' else 2 # If buying, TP is triggered by price rising (buy trigger direction) - Check Bybit docs! This might be wrong. TP trigger is usually opposite side.
             # TP Trigger: If long, TP is a sell limit/market order triggered when price rises. If short, TP is a buy limit/market.
             ccxt_params.update({
                 'takeProfit': str(tp_price_rounded),
                 # 'tpTriggerBy': config['order'].get('tp_trigger_type', 'LastPrice'),
                 # 'tpOrderType': 'Market',
             })
             log.info(f"Prepared TP: Price={tp_price_rounded}")

        # Log the attempt
        log.warning(f"{Fore.CYAN}Attempting to place {side.upper()} {order_type.upper()} order:{Style.RESET_ALL}\n"
                    f"  Symbol: {symbol}\n"
                    f"  Amount: {amount}\n"
                    f"  Limit Price: {limit_price if limit_price else 'N/A'}\n"
                    f"  SL Price: {ccxt_params.get('stopLoss', 'N/A')}\n"
                    f"  TP Price: {ccxt_params.get('takeProfit', 'N/A')}\n"
                    f"  Params: {ccxt_params}")

        try:
            # Place the order using ccxt.create_order
            order = await exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=limit_price, # Pass None for market orders
                params=ccxt_params # Pass exchange-specific params here
            )
            log.info(f"{Fore.GREEN}Order placed successfully! ID: {order.get('id')}{Style.RESET_ALL}")
            log.debug(f"Order details: {order}")

            # Force position check soon after placing order to confirm state
            global last_position_check_time
            last_position_check_time = 0

            return order # Return the order details dictionary

        except ccxt.InsufficientFunds as e:
            log.error(f"{Fore.RED}Order Failed: Insufficient Funds.{Style.RESET_ALL} {e}")
            # Check available balance, leverage, and potential open orders locking margin.
            await get_wallet_balance(market['quote']) # Log current balance
            return None
        except ccxt.InvalidOrder as e:
             log.error(f"{Fore.RED}Order Failed: Invalid Order Parameters.{Style.RESET_ALL} Check config, calculations, and market limits. {e}")
             log.error(f"Order details attempted: symbol={symbol}, type={order_type}, side={side}, amount={amount}, price={limit_price}, params={ccxt_params}")
             # Common issues: Price/amount precision, below min size/cost, invalid SL/TP params for the exchange.
             return None
        except ccxt.ExchangeNotAvailable as e:
             log.error(f"{Fore.RED}Order Failed: Exchange Not Available (Maintenance?).{Style.RESET_ALL} {e}")
             # Implement retry logic or pause trading.
             return None
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            log.error(f"{Fore.RED}Order Failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
            # Consider retrying network errors. Exchange errors might be permanent (e.g., invalid symbol).
            return None
        except Exception as e:
            log.error(f"{Fore.RED}Unexpected error placing order: {e}{Style.RESET_ALL}", exc_info=True)
            return None

async def close_position(symbol: str, position_data: Dict[str, Any]) -> Optional[Dict]:
    """Closes the current position using a reduce-only market order."""
    if not exchange or not strategy_instance or not market:
        log.error("Cannot close position: Exchange, strategy, or market not ready.")
        return None

    mode = config.get("mode", "Live").lower()
    current_size = position_data.get('size', Decimal(0))
    current_side = position_data.get('side', 'None') # Expect 'Buy' or 'Sell'

    if mode == "paper":
        if current_size > 0:
            log.warning(f"[PAPER MODE] Simulating closing {current_side} position for {symbol} (Size: {current_size})")
            # Simulate successful close
            return {"id": f"paper_close_{int(time.time())}", "info": {"paperTrade": True, "simulatedClose": True}}
        else:
            log.info("[PAPER MODE] Attempted to close position, but already flat.")
            return {"info": {"alreadyFlat": True}}

    # --- Live/Testnet Position Close ---
    async with order_lock: # Ensure only one closing order attempt at a time
        if current_size <= 0 or current_side == 'None':
            log.info(f"Attempted to close position for {symbol}, but position data shows it's already flat.")
            return {"info": {"alreadyFlat": True}}

        # Determine the side and amount for the closing order
        side_to_close = 'sell' if current_side == 'Buy' else 'buy'
        # Use the exact size from the fetched position data
        amount_to_close = float(current_size)

        log.warning(f"{Fore.YELLOW}Attempting to close {current_side} position for {symbol} (Size: {amount_to_close}). Placing {side_to_close.upper()} Market order...{Style.RESET_ALL}")

        # --- Cancel Existing SL/TP Orders First (Important!) ---
        # If SL/TP were placed as separate orders (not attached), cancel them before closing.
        # If SL/TP were attached via params, the reduceOnly close should handle it (verify exchange behavior).
        # Example: Cancelling conditional orders on Bybit (syntax might vary)
        try:
            log.debug(f"Attempting to cancel existing conditional orders (SL/TP) for {symbol} before closing...")
            # This might require specific params depending on the exchange API version
            # Example for Bybit V5: cancel_all_orders with {'orderFilter': 'StopOrder'} or similar
            # Check CCXT documentation for the correct method and parameters
            # await exchange.cancel_all_orders(symbol, params={'orderFilter': 'StopOrder'}) # Adjust params as needed
            # Simpler approach: Cancel all open orders for the symbol (use with caution if other strategies run)
            await exchange.cancel_all_orders(symbol)
            log.info(f"Successfully cancelled open orders for {symbol}.")
            await asyncio.sleep(0.5) # Short delay to allow cancellation processing
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            log.warning(f"Could not cancel orders before closing position (might be none or API issue): {e}")
        except Exception as e:
            log.error(f"Unexpected error cancelling orders: {e}", exc_info=True)


        # --- Place the Reduce-Only Market Order ---
        params = {
            'reduceOnly': True,
            # 'positionIdx': 0 # May be needed for Bybit hedge mode
        }
        try:
            order = await exchange.create_order(
                symbol=symbol,
                type='market', # Use market order for immediate close
                side=side_to_close,
                amount=amount_to_close,
                params=params
            )
            log.info(f"{Fore.GREEN}Position close order placed successfully! ID: {order.get('id')}{Style.RESET_ALL}")
            log.debug(f"Close Order details: {order}")

            # Force position check soon after closing attempt
            global last_position_check_time
            last_position_check_time = 0
            return order

        except ccxt.InvalidOrder as e:
             # This often happens if the position was already closed manually or by SL/TP
             # Or if the size changed between fetch and close attempt.
             log.warning(f"Close order failed, likely position already closed or size changed: {e}")
             last_position_check_time = 0 # Force position check to confirm state
             return {"info": {"error": str(e), "alreadyFlatOrChanged": True}}
        except ccxt.InsufficientFunds as e:
             # Should not happen with reduceOnly, but log if it does
             log.error(f"{Fore.RED}Close order failed: Insufficient Funds (unexpected for reduceOnly). {e}{Style.RESET_ALL}")
             last_position_check_time = 0
             return None
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            log.error(f"{Fore.RED}Close order failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
            # Consider retry logic for network errors
            return None
        except Exception as e:
            log.error(f"{Fore.RED}Unexpected error closing position: {e}{Style.RESET_ALL}", exc_info=True)
            return None

async def set_leverage(symbol: str, leverage: int):
    """Sets leverage for the specified symbol using CCXT (if supported)."""
    if not exchange or not market:
        log.warning("Cannot set leverage: Exchange or market not ready.")
        return
    if not exchange.has.get('setLeverage'):
         log.warning(f"Exchange {exchange.name} does not support setting leverage via CCXT method 'setLeverage'. Manual setting might be required.")
         return

    # Validate leverage against market limits if available
    max_leverage = market.get('limits', {}).get('leverage', {}).get('max')
    if max_leverage is not None and not 1 <= leverage <= max_leverage:
         log.error(f"Invalid leverage {leverage}. Must be between 1 and {max_leverage} for {symbol}. Leverage not set.")
         return

    log.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
    try:
        # Note: CCXT's set_leverage abstracts underlying calls. Behavior depends on exchange.
        # Some exchanges require setting buy/sell leverage separately or have mode requirements.
        # Bybit V5: set_leverage usually works for the symbol directly in one-way mode.
        await exchange.set_leverage(leverage, symbol)
        log.info(f"{Fore.GREEN}Leverage for {symbol} set to {leverage}x request sent.{Style.RESET_ALL} (Confirmation may depend on exchange response)")

        # Optional: Verify leverage by fetching position data immediately after setting
        # await asyncio.sleep(1) # Short delay
        # pos_check = await get_current_position(symbol)
        # if pos_check and 'info' in pos_check and pos_check['info'].get('leverage') == str(leverage):
        #     log.info("Leverage confirmed via position check.")
        # elif pos_check:
        #     log.warning(f"Leverage verification failed. Position shows leverage: {pos_check.get('leverage', 'N/A')}")

    except ccxt.ExchangeError as e:
         # Handle common errors like "leverage not modified"
         # Bybit V5 error code for "Leverage not modified": 110044
         if "not modified" in str(e).lower() or "110044" in str(e):
              log.warning(f"Leverage for {symbol} already set to {leverage}x (Exchange response: Not modified).")
         else:
              log.error(f"Failed to set leverage for {symbol}: {e}")
    except Exception as e:
        log.error(f"Unexpected error setting leverage: {e}", exc_info=True)


# --- WebSocket Watcher Loops ---
async def watch_kline_loop(symbol: str, timeframe: str):
    """Watches for new OHLCV candles via WebSocket and triggers processing."""
    global last_ws_update_time
    if not exchange or not exchange.has.get('watchOHLCV'):
        log.error("Kline watcher cannot start: Exchange does not support watchOHLCV.")
        return # Exit task if not supported

    log.info(f"Starting Kline watcher for {symbol} ({timeframe})...")
    while not stop_event.is_set():
        try:
            # watch_ohlcv returns a list of *closed* candles since the last call
            candles = await exchange.watch_ohlcv(symbol, timeframe)
            # [[timestamp, open, high, low, close, volume]]
            if not candles:
                log.debug("watch_ohlcv returned empty list.")
                continue

            now_mono = time.monotonic()
            if now_mono - last_ws_update_time > 5: # Log if WS was quiet for a bit
                 log.debug(f"Kline WS received data after {now_mono - last_ws_update_time:.1f}s silence.")
            last_ws_update_time = now_mono # Update health check timestamp

            # Process each received closed candle
            for candle_data in candles:
                try:
                    ts_ms, o, h, l, c, v = candle_data
                    # Validate data types before processing
                    if not all(isinstance(x, (int, float)) for x in [ts_ms, o, h, l, c, v]):
                        log.warning(f"Received invalid data types in candle: {candle_data}. Skipping.")
                        continue

                    ts = pd.to_datetime(ts_ms, unit='ms')
                    new_data = {'open': float(o), 'high': float(h), 'low': float(l), 'close': float(c), 'volume': float(v)}
                    log.debug(f"WS Kline Received: T={ts}, O={o}, H={h}, L={l}, C={c}, V={v}")

                    # Process the confirmed candle data asynchronously
                    await process_candle(ts, new_data)

                except (ValueError, TypeError) as e:
                    log.error(f"Error processing individual candle data {candle_data}: {e}")
                    continue # Skip this candle and process the next

        except ccxt.NetworkError as e:
            log.warning(f"Kline Watcher Network Error: {e}. Reconnecting...")
            await asyncio.sleep(5) # Wait before implicit reconnection by watch_ohlcv
        except ccxt.ExchangeError as e:
            log.warning(f"Kline Watcher Exchange Error: {e}. Retrying...")
            await asyncio.sleep(5)
        except asyncio.CancelledError:
             log.info("Kline watcher task cancelled.")
             break # Exit loop cleanly on cancellation
        except Exception as e:
            log.error(f"Unexpected error in Kline watcher: {e}", exc_info=True)
            await asyncio.sleep(10) # Longer sleep on unexpected errors before retrying
    log.info("Kline watcher loop finished.")

async def process_candle(timestamp: pd.Timestamp, data: Dict[str, float]):
    """Adds candle data to the DataFrame and triggers strategy analysis."""
    global latest_dataframe # Allow modification
    async with data_lock: # Ensure exclusive access to the dataframe
        if latest_dataframe is None:
            log.warning("DataFrame not initialized, skipping candle processing.")
            return

        # Check if timestamp already exists (e.g., duplicate WS message or overlap with initial fetch)
        if timestamp in latest_dataframe.index:
             # Update the existing row - useful if WS sends updates for the same closed candle
             log.debug(f"Updating existing candle data for {timestamp}.")
             latest_dataframe.loc[timestamp, list(data.keys())] = list(data.values())
        else:
             # Append new candle data
             log.debug(f"Adding new candle {timestamp} to DataFrame.")
             # Create a new DataFrame row with the correct index type
             new_row = pd.DataFrame([data], index=pd.DatetimeIndex([timestamp]))
             latest_dataframe = pd.concat([latest_dataframe, new_row])

             # Prune old data to maintain max_df_len
             max_len = config.get('data', {}).get('max_df_len', 2000)
             if len(latest_dataframe) > max_len:
                 latest_dataframe = latest_dataframe.iloc[-max_len:]
                 # log.debug(f"DataFrame pruned to {len(latest_dataframe)} rows.")

        # --- Trigger Strategy Analysis on the updated DataFrame ---
        if strategy_instance:
            log.info(f"Running analysis on DataFrame ending with candle: {timestamp}")
            # Analyze a copy to avoid modifying the locked dataframe during analysis
            df_copy = latest_dataframe.copy()
            try:
                # Run the strategy's update method
                analysis_results = strategy_instance.update(df_copy)

                # Process the generated signals asynchronously
                # Create a new task to handle signal processing without blocking candle updates
                signal_task = asyncio.create_task(process_signals(analysis_results))
                running_tasks.add(signal_task)
                # Remove task from set when done to prevent memory leak
                signal_task.add_done_callback(running_tasks.discard)

            except Exception as e:
                log.error(f"Error during strategy analysis update: {e}", exc_info=True)
        else:
             log.warning("Strategy instance not available for analysis.")


async def watch_positions_loop(symbol: str):
    """(Optional) Watches for position updates via WebSocket."""
    global current_position, last_ws_update_time, last_position_check_time
    if not exchange or not exchange.has.get('watchPositions'):
        log.info("Position watcher skipped: Exchange does not support watchPositions.")
        return

    log.info(f"Starting Position watcher for {symbol}...")
    while not stop_event.is_set():
        try:
            # watch_positions usually returns a list of all positions for the account type
            positions_updates = await exchange.watch_positions([symbol]) # Watch specific symbol if supported
            now_mono = time.monotonic()
            if now_mono - last_ws_update_time > 5: log.debug(f"Position WS received data after {now_mono - last_ws_update_time:.1f}s silence.")
            last_ws_update_time = now_mono # Update health check timestamp

            if not positions_updates: continue

            pos_update = None
            for p in positions_updates:
                if p.get('symbol') == symbol:
                     pos_update = p
                     break # Found our symbol

            if pos_update:
                log.info(f"{Fore.CYAN}Position update via WS: {pos_update}{Style.RESET_ALL}")
                # Parse the update carefully
                try:
                    size_str = pos_update.get('contracts', '0') or '0'
                    side = pos_update.get('side', 'none').lower()
                    entry_price_str = pos_update.get('entryPrice', '0') or '0'
                    ws_size = Decimal(size_str)
                    ws_side = "Buy" if side == 'long' else "Sell" if side == 'short' else "None"
                    ws_entry_price = Decimal(entry_price_str)

                    # Update internal state cautiously - REST check remains the source of truth for critical actions
                    async with position_lock:
                        # Log if WS state differs significantly from last known REST state
                        if ws_size != current_position['size'] or ws_side != current_position['side']:
                            log.warning(f"WS position update differs from cached state. "
                                        f"WS: {ws_side} {ws_size}. Cache: {current_position['side']} {current_position['size']}. "
                                        f"Forcing REST check.")
                            # Force a REST check soon to confirm the change
                            last_position_check_time = 0
                        # Optionally update the cached state with WS data for faster reflection,
                        # but rely on REST for decisions.
                        # current_position['size'] = ws_size
                        # current_position['side'] = ws_side
                        # current_position['entry_price'] = ws_entry_price
                        # current_position['timestamp'] = time.time() # Mark as WS update time?

                except (InvalidOperation, TypeError) as e:
                    log.error(f"Error parsing WS position update data {pos_update}: {e}")
            else:
                 # This might happen if the position for the symbol closes
                 log.debug(f"Received position update via WS, but not for {symbol} (or position closed).")
                 # If we previously had a position, force a REST check
                 async with position_lock:
                     if current_position['size'] != Decimal(0):
                         log.warning(f"WS position update no longer includes {symbol}. Forcing REST check.")
                         last_position_check_time = 0


        except ccxt.NetworkError as e: log.warning(f"Position Watcher Network Error: {e}. Reconnecting..."); await asyncio.sleep(5)
        except ccxt.ExchangeError as e: log.warning(f"Position Watcher Exchange Error: {e}. Retrying..."); await asyncio.sleep(5)
        except asyncio.CancelledError: log.info("Position watcher task cancelled."); break
        except Exception as e: log.error(f"Unexpected error in Position watcher: {e}", exc_info=True); await asyncio.sleep(10)
    log.info("Position watcher loop finished.")

async def watch_orders_loop(symbol: str):
    """(Optional) Watches for order updates (fills, cancellations) via WebSocket."""
    global last_ws_update_time, last_position_check_time
    if not exchange or not exchange.has.get('watchOrders'):
        log.info("Order watcher skipped: Exchange does not support watchOrders.")
        return

    log.info(f"Starting Order watcher for {symbol}...")
    while not stop_event.is_set():
        try:
            orders = await exchange.watch_orders(symbol)
            now_mono = time.monotonic()
            if now_mono - last_ws_update_time > 5: log.debug(f"Order WS received data after {now_mono - last_ws_update_time:.1f}s silence.")
            last_ws_update_time = now_mono # Update health check timestamp

            if not orders: continue

            for order_update in orders:
                 # Process order updates - log fills, SL/TP triggers, cancellations
                 status = order_update.get('status') # 'open', 'closed', 'canceled', 'expired', 'rejected'
                 order_id = order_update.get('id')
                 filled = order_update.get('filled', 0.0)
                 avg_price = order_update.get('average')
                 order_type = order_update.get('type')
                 order_side = order_update.get('side')
                 log.info(f"{Fore.CYAN}Order Update via WS [{symbol}]: ID={order_id}, Side={order_side}, Type={order_type}, Status={status}, Filled={filled}, AvgPrice={avg_price}{Style.RESET_ALL}")

                 # If an order is filled or closed/canceled, it might affect position state
                 if status in ['closed', 'canceled'] and order_id:
                      log.warning(f"{Fore.YELLOW}Order {order_id} ({order_side} {order_type}) reached terminal state '{status}' via WS.{Style.RESET_ALL}")
                      # Force a position check to get the most accurate state after order execution/cancellation
                      log.info("Forcing position check after order update.")
                      async with position_lock: # Use lock just to modify check time
                          last_position_check_time = 0

        except ccxt.NetworkError as e: log.warning(f"Order Watcher Network Error: {e}. Reconnecting..."); await asyncio.sleep(5)
        except ccxt.ExchangeError as e: log.warning(f"Order Watcher Exchange Error: {e}. Retrying..."); await asyncio.sleep(5)
        except asyncio.CancelledError: log.info("Order watcher task cancelled."); break
        except Exception as e: log.error(f"Unexpected error in Order watcher: {e}", exc_info=True); await asyncio.sleep(10)
    log.info("Order watcher loop finished.")

# --- Signal Processing & Execution (Async) ---
async def process_signals(results: AnalysisResults):
    """Processes strategy signals and executes trades based on position state."""
    if stop_event.is_set():
        log.warning("Signal processing skipped: Stop event is set.")
        return
    if not results or not strategy_instance or not market:
        log.warning("Signal processing skipped: Missing analysis results, strategy, or market data.")
        return

    signal = results['last_signal']
    last_close = results['last_close']
    last_atr = results['last_atr']
    symbol = config['symbol']

    log.debug(f"Processing Signal: {signal}, Last Close: {last_close:.{strategy_instance.price_precision}f}, Last ATR: {last_atr}")

    if pd.isna(last_close):
        log.warning("Cannot process signal: Last close price is NaN.")
        return

    # --- Get Current Position State (Crucial Step) ---
    # Use the reliable REST API fetch before taking any action based on the signal
    pos_data = await get_current_position(symbol)
    if not pos_data: # Indicates fetch error
        log.error("Could not get reliable position data. Skipping signal action to avoid errors.")
        return

    # Use Decimal for position size comparison
    current_pos_size = pos_data.get('size', Decimal(0))
    current_pos_side = pos_data.get('side', 'None') # 'Buy', 'Sell', 'None'
    is_long = current_pos_side == 'Buy' and current_pos_size > Decimal(0)
    is_short = current_pos_side == 'Sell' and current_pos_size > Decimal(0)
    is_flat = not is_long and not is_short

    log.info(f"Processing signal '{signal}' | Current Position: {'Long' if is_long else 'Short' if is_short else 'Flat'} (Size: {current_pos_size})")

    # --- Execute Actions based on Signal and Current Position State ---
    tp_ratio = Decimal(str(config['order'].get('tp_ratio', 2.0)))
    risk_percent = float(config['order'].get('risk_per_trade_percent', 1.0))
    sl_method = config.get('strategy', {}).get('stop_loss', {}).get('method', 'ATR')
    sl_atr_multiplier = float(config.get('strategy', {}).get('stop_loss', {}).get('atr_multiplier', 1.5))

    # --- BUY Signal ---
    if signal == "BUY" and is_flat:
        log.warning(f"{Fore.GREEN}{Style.BRIGHT}BUY Signal received - Attempting to Enter Long.{Style.RESET_ALL}")
        # Calculate SL
        sl_price_raw = None
        if sl_method == "ATR" and last_atr and last_atr > 0:
            sl_price_raw = last_close - (last_atr * sl_atr_multiplier)
        elif sl_method == "OB":
             # Find the highest bottom of active bull boxes below the current price
             relevant_obs = [b['bottom'] for b in results.get('active_bull_boxes', []) if b['bottom'] < last_close]
             if relevant_obs:
                  ob_sl_level = max(relevant_obs) # Use the highest support level below price
                  # Add a small buffer (e.g., fraction of ATR or price tick)
                  sl_buffer = (last_atr * 0.1) if last_atr else (strategy_instance.price_tick * Decimal(5))
                  sl_price_raw = float(Decimal(str(ob_sl_level)) - sl_buffer)
             else:
                  log.warning("OB SL method selected, but no relevant Bull OB found below price. Falling back to ATR.")
                  if last_atr and last_atr > 0: sl_price_raw = last_close - (last_atr * sl_atr_multiplier * 1.5) # Wider ATR fallback

        if sl_price_raw is None or sl_price_raw >= last_close:
             log.error(f"Invalid SL price calculated for BUY signal (SL={sl_price_raw}, Close={last_close}). Aborting entry.")
             return
        sl_price = strategy_instance.round_price(sl_price_raw) # Round SL price appropriately (e.g., down for long SL)

        # Calculate TP
        sl_distance = Decimal(str(last_close)) - Decimal(str(sl_price))
        tp_price_raw = float(Decimal(str(last_close)) + (sl_distance * tp_ratio)) if sl_distance > 0 else None
        tp_price = strategy_instance.round_price(tp_price_raw) if tp_price_raw else None # Round TP (e.g., down for long TP)

        # Calculate Quantity
        qty = await calculate_order_qty(last_close, sl_price, risk_percent)
        if qty and qty > 0:
            # Place the order
            await place_order(symbol, 'buy', qty,
                              price=last_close if config['order']['type'].lower() == 'limit' else None,
                              sl_price=sl_price, tp_price=tp_price)
        else:
            log.error("BUY order cancelled: Quantity calculation failed or resulted in zero.")

    # --- SELL Signal ---
    elif signal == "SELL" and is_flat:
        log.warning(f"{Fore.RED}{Style.BRIGHT}SELL Signal received - Attempting to Enter Short.{Style.RESET_ALL}")
        # Calculate SL
        sl_price_raw = None
        if sl_method == "ATR" and last_atr and last_atr > 0:
             sl_price_raw = last_close + (last_atr * sl_atr_multiplier)
        elif sl_method == "OB":
            # Find the lowest top of active bear boxes above the current price
            relevant_obs = [b['top'] for b in results.get('active_bear_boxes', []) if b['top'] > last_close]
            if relevant_obs:
                  ob_sl_level = min(relevant_obs) # Use the lowest resistance level above price
                  sl_buffer = (last_atr * 0.1) if last_atr else (strategy_instance.price_tick * Decimal(5))
                  sl_price_raw = float(Decimal(str(ob_sl_level)) + sl_buffer)
            else:
                  log.warning("OB SL method selected, but no relevant Bear OB found above price. Falling back to ATR.")
                  if last_atr and last_atr > 0: sl_price_raw = last_close + (last_atr * sl_atr_multiplier * 1.5) # Wider ATR fallback

        if sl_price_raw is None or sl_price_raw <= last_close:
            log.error(f"Invalid SL price calculated for SELL signal (SL={sl_price_raw}, Close={last_close}). Aborting entry.")
            return
        sl_price = strategy_instance.round_price(sl_price_raw) # Round SL (e.g., up for short SL)

        # Calculate TP
        sl_distance = Decimal(str(sl_price)) - Decimal(str(last_close))
        tp_price_raw = float(Decimal(str(last_close)) - (sl_distance * tp_ratio)) if sl_distance > 0 else None
        tp_price = strategy_instance.round_price(tp_price_raw) if tp_price_raw else None # Round TP (e.g., up for short TP)

        # Calculate Quantity
        qty = await calculate_order_qty(last_close, sl_price, risk_percent)
        if qty and qty > 0:
            # Place the order
            await place_order(symbol, 'sell', qty,
                              price=last_close if config['order']['type'].lower() == 'limit' else None,
                              sl_price=sl_price, tp_price=tp_price)
        else:
            log.error("SELL order cancelled: Quantity calculation failed or resulted in zero.")

    # --- EXIT_LONG Signal ---
    elif signal == "EXIT_LONG" and is_long:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_LONG Signal received - Attempting to Close Long Position.{Style.RESET_ALL}")
        await close_position(symbol, pos_data) # Pass the fetched position data

    # --- EXIT_SHORT Signal ---
    elif signal == "EXIT_SHORT" and is_short:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_SHORT Signal received - Attempting to Close Short Position.{Style.RESET_ALL}")
        await close_position(symbol, pos_data) # Pass the fetched position data

    # --- No Action / Already in State ---
    elif signal == "HOLD":
        log.debug("HOLD Signal - No action taken.")
    elif signal == "BUY" and is_long:
        log.debug("BUY Signal received, but already Long. No action.")
    elif signal == "SELL" and is_short:
        log.debug("SELL Signal received, but already Short. No action.")
    elif signal == "EXIT_LONG" and not is_long:
        log.debug("EXIT_LONG Signal received, but not Long. No action.")
    elif signal == "EXIT_SHORT" and not is_short:
        log.debug("EXIT_SHORT Signal received, but not Short. No action.")
    else:
        # Should not happen if logic is correct
        log.debug(f"Signal '{signal}' received but no matching action criteria met (Position: {current_pos_side} {current_pos_size}).")


# --- Periodic Health Check ---
async def periodic_check_loop():
    """Runs periodic checks for WebSocket health, stale positions, etc."""
    global last_health_check_time, last_position_check_time, last_ws_update_time
    check_interval = config.get('checks', {}).get('health_check_interval', 60)
    pos_check_interval = config.get('checks', {}).get('position_check_interval', 30)
    # Consider WS dead if no message received for slightly longer than health check interval
    ws_timeout = check_interval * 1.5

    log.info(f"Starting Periodic Check loop (Interval: {check_interval}s)")
    while not stop_event.is_set():
        await asyncio.sleep(check_interval) # Wait for the interval
        if stop_event.is_set(): break # Exit if stop signal received during sleep

        now_mono = time.monotonic()
        log.debug(f"Running periodic checks (Time: {now_mono:.1f})...")
        last_health_check_time = now_mono

        # 1. Check WebSocket Health (based on last message time from any WS watcher)
        time_since_last_ws = now_mono - last_ws_update_time if last_ws_update_time > 0 else ws_timeout
        if time_since_last_ws >= ws_timeout:
             log.warning(f"{Fore.RED}WebSocket potentially stale! No updates received for {time_since_last_ws:.1f}s (Timeout: {ws_timeout}s).{Style.RESET_ALL}")
             # Action: Could try to reconnect WS or trigger a more serious alert.
             # The main loop already checks if essential tasks (like kline watcher) have died.
        else:
             log.debug(f"WebSocket health check OK (last update {time_since_last_ws:.1f}s ago).")

        # 2. Force Position Check if REST data is stale
        # This ensures position is checked even if WS updates are missed or delayed.
        time_since_last_pos_check = now_mono - last_position_check_time if last_position_check_time > 0 else pos_check_interval
        if time_since_last_pos_check >= pos_check_interval:
             log.info("Periodic check forcing REST position update...")
             # Run the check non-blockingly
             pos_check_task = asyncio.create_task(get_current_position(config['symbol']))
             running_tasks.add(pos_check_task)
             pos_check_task.add_done_callback(running_tasks.discard)

        # 3. Add other checks as needed:
        #    - Check available balance periodically?
        #    - Check exchange status endpoint?
        #    - Check for excessive error logs?

        log.debug("Periodic checks complete.")

    log.info("Periodic check loop finished.")


# --- Graceful Shutdown ---
async def shutdown(signal_type: Optional[signal.Signals] = None):
    """Cleans up resources, cancels tasks, and exits gracefully."""
    if stop_event.is_set():
        log.warning("Shutdown already in progress.")
        return

    signal_name = f"signal {signal_type.name}" if signal_type else "request"
    log.warning(f"Shutdown initiated by {signal_name}...")
    stop_event.set() # Signal all loops and tasks to stop

    # Cancel all running asyncio tasks collected in running_tasks
    tasks_to_cancel = list(running_tasks)
    if tasks_to_cancel:
        log.info(f"Cancelling {len(tasks_to_cancel)} running background tasks...")
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
        # Wait for tasks to finish cancelling
        results = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        log.info("Background tasks cancellation complete.")
        # Log any exceptions that occurred during task cancellation/execution
        for i, result in enumerate(results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                log.error(f"Task {tasks_to_cancel[i].get_name()} raised an exception during shutdown: {result}", exc_info=result)


    # Close the CCXT exchange connection
    if exchange and hasattr(exchange, 'close') and not getattr(exchange, 'closed', True):
        log.info("Closing CCXT exchange connection...")
        try:
            await exchange.close()
            log.info("Exchange connection closed.")
        except Exception as e:
            log.error(f"Error closing exchange connection during shutdown: {e}", exc_info=True)

    # Optional: Implement logic to close open positions on exit (use with extreme caution!)
    # Consider if this is desired behavior. Manual intervention might be safer.
    # await close_open_position_on_exit()

    log.warning("Shutdown sequence complete. Exiting.")
    # Allow logs to flush before exiting
    await asyncio.sleep(0.5)
    # Explicitly exit - needed if shutdown is called not from main loop exit
    sys.exit(0)

async def close_open_position_on_exit():
     """Placeholder for logic to close any open position during shutdown."""
     log.warning("Checking for open position to close on exit (USE WITH CAUTION)...")
     # Ensure exchange is still usable (might fail if connection closed early)
     if not exchange or not market or getattr(exchange, 'closed', True):
         log.error("Cannot check/close position on exit: Exchange not available.")
         return

     try:
         # Fetch position one last time
         pos_data = await get_current_position(config['symbol'])
         if pos_data and pos_data.get('size', Decimal(0)) > Decimal(0):
             log.warning(f"Found open {pos_data['side']} position (Size: {pos_data['size']}). Attempting to close...")
             await close_position(config['symbol'], pos_data)
         else:
             log.info("No open position found to close on exit.")
     except Exception as e:
         log.error(f"Error during position close on exit: {e}", exc_info=True)


# --- Main Application Entry Point ---
async def main():
    """Main asynchronous function to initialize and run the bot."""
    global config, exchange, market, strategy_instance, latest_dataframe, running_tasks

    print(Fore.MAGENTA + Style.BRIGHT + "\n~~~ Pyrmethus Volumatic+OB Trading Bot Starting (CCXT Async) ~~~" + Style.RESET_ALL)
    print(f"Timestamp: {datetime.datetime.now()}")

    # Load configuration first
    config = load_config() # Exits on failure

    # Setup logging based on config
    setup_logging(config.get("log_level", "INFO"))
    log.info("Logging configured.")
    log.debug(f"Full Config: {json.dumps(config, indent=2)}")

    # Validate API Keys early
    if not API_KEY or not API_SECRET:
        log.critical("CRITICAL: BYBIT_API_KEY or BYBIT_API_SECRET not set in .env file.")
        sys.exit(1)
    log.info(f"API Key found (ending with ...{API_KEY[-4:]})")

    # Connect to exchange
    exchange = await connect_ccxt()
    if not exchange:
        log.critical("Failed to connect to exchange. Exiting.")
        sys.exit(1)

    # Load market info for the target symbol
    market = await load_exchange_market(config['symbol'])
    if not market:
        log.critical(f"Failed to load market data for {config['symbol']}. Exiting.")
        await exchange.close() # Clean up connection
        sys.exit(1)

    # Attempt to set leverage (log warning/error if fails, but continue)
    await set_leverage(config['symbol'], config['order']['leverage'])

    # Initialize Strategy Engine
    try:
        strategy_params = config.get('strategy', {}).get('params', {})
        strategy_instance = VolumaticOBStrategy(market=market, **strategy_params)
    except Exception as e:
        log.critical(f"Failed to initialize strategy: {e}. Check config.json['strategy']['params'].", exc_info=True)
        await exchange.close()
        sys.exit(1)

    # Fetch Initial Historical Data
    async with data_lock: # Lock dataframe during initial population
        initial_fetch_limit = config.get('data', {}).get('fetch_limit', 750)
        latest_dataframe = await fetch_initial_data(
            config['symbol'],
            config['timeframe'],
            initial_fetch_limit
        )
        if latest_dataframe is None: # Indicates fetch error
            log.critical("Failed to fetch initial market data. Exiting.")
            await exchange.close()
            sys.exit(1)
        elif len(latest_dataframe) < strategy_instance.min_data_len:
            log.warning(f"Initial data fetched ({len(latest_dataframe)}) is less than minimum required by strategy ({strategy_instance.min_data_len}). Strategy will need more data from WebSocket.")
        else:
            # Run initial analysis on historical data if enough candles were fetched
            log.info("Running initial analysis on historical data...")
            try:
                # Make a copy for analysis
                df_copy = latest_dataframe.copy()
                initial_results = strategy_instance.update(df_copy)
                trend_str = 'UP' if initial_results['current_trend'] else 'DOWN' if initial_results['current_trend'] is not None else 'UNDETERMINED'
                log.info(f"Initial Analysis Complete: Trend={trend_str}, Last Close={initial_results['last_close']:.{strategy_instance.price_precision}f}, Initial Signal={initial_results['last_signal']}")
            except Exception as e:
                 log.error(f"Error during initial strategy analysis: {e}", exc_info=True)


    # Fetch initial position state
    await get_current_position(config['symbol'])
    log.info(f"Initial Position State: {current_position['side']} (Size: {current_position['size']})")

    # --- Start Background Tasks ---
    log.info(f"{Fore.CYAN}Setup complete. Starting WebSocket watchers and periodic checks...{Style.RESET_ALL}")
    log.info(f"Trading Mode: {config.get('mode', 'Live')}")
    log.info(f"Symbol: {config['symbol']} | Timeframe: {config['timeframe']}")

    # Create and store tasks for graceful shutdown
    tasks_to_start = [
        # Essential: Kline Watcher
        asyncio.create_task(watch_kline_loop(config['symbol'], config['timeframe']), name="KlineWatcher"),
        # Optional: Position Watcher (can reduce REST calls but adds complexity)
        # asyncio.create_task(watch_positions_loop(config['symbol']), name="PositionWatcher"),
        # Optional: Order Watcher (for faster fill/cancel confirmation)
        # asyncio.create_task(watch_orders_loop(config['symbol']), name="OrderWatcher"),
        # Essential: Periodic Health/Position Check
        asyncio.create_task(periodic_check_loop(), name="PeriodicChecker"),
    ]
    running_tasks.update(tasks_to_start)

    # Keep main running, monitor essential tasks, and handle shutdown signal
    log.info("Main loop running. Monitoring tasks... (Press Ctrl+C to stop)")
    while not stop_event.is_set():
         # Check if essential tasks (like kline watcher) are still running
         kline_task = next((t for t in running_tasks if t.get_name() == "KlineWatcher"), None)

         if kline_task and kline_task.done():
              log.critical(f"{Fore.RED}CRITICAL: Kline watcher task has terminated unexpectedly!{Style.RESET_ALL}")
              try:
                   # This will raise the exception if the task failed
                   kline_task.result()
              except asyncio.CancelledError:
                   log.warning("Kline watcher was cancelled.") # Expected during shutdown
              except Exception as e:
                   log.critical(f"Kline watcher failed with error: {e}", exc_info=True)

              log.critical("Attempting to stop bot gracefully due to essential task failure...")
              # Trigger shutdown without waiting for OS signal
              # Use create_task to avoid blocking main loop if shutdown takes time
              asyncio.create_task(shutdown())
              break # Exit main monitoring loop

         # Heartbeat sleep for the main loop
         await asyncio.sleep(5)

    log.info("Main loop finished.")


if __name__ == "__main__":
    # Get the asyncio event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Setup signal handlers for graceful shutdown (SIGINT: Ctrl+C, SIGTERM: kill)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            # Use create_task to run shutdown coroutine when signal is received
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for SIGTERM
            if sig == signal.SIGINT:
                 # Still try to catch Ctrl+C on Windows
                 signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(shutdown(signal.SIGINT)))
            log.warning(f"Signal handler for {sig.name} not supported on this platform.")


    try:
        # Run the main asynchronous function
        loop.run_until_complete(main())
    except asyncio.CancelledError:
         log.info("Main task cancelled during shutdown.")
    except KeyboardInterrupt: # Catch Ctrl+C if signal handler fails
         log.warning("KeyboardInterrupt caught in main. Initiating shutdown...")
         loop.run_until_complete(shutdown(signal.SIGINT))
    finally:
         # Final check to ensure exchange connection is closed
         if exchange and hasattr(exchange, 'close') and not getattr(exchange, 'closed', True):
             log.warning("Exchange connection still open after main loop exit. Closing now.")
             try:
                 loop.run_until_complete(exchange.close())
             except Exception as e:
                 log.error(f"Error during final exchange close: {e}")

         # Cancel any remaining tasks just in case
         remaining_tasks = [t for t in asyncio.all_tasks(loop=loop) if t is not asyncio.current_task(loop=loop)]
         if remaining_tasks:
             log.warning(f"Cancelling {len(remaining_tasks)} potentially lingering tasks...")
             for task in remaining_tasks:
                 task.cancel()
             loop.run_until_complete(asyncio.gather(*remaining_tasks, return_exceptions=True))

         # Close the loop (important for cleanup)
         # loop.close() # Deprecated since Python 3.9? asyncio handles this better now.
         log.info("Application finished.")

EOF

# --- Final Instructions ---
echo -e "${GREEN}All files created successfully in ${YELLOW}${PROJECT_DIR}${NC}"
echo -e "\n${YELLOW}--- Next Steps ---${NC}"
echo -e "1. ${YELLOW}Edit the ${BLUE}${ENV_FILE}${YELLOW} file and add your actual Bybit API Key and Secret.${NC}"
echo -e "   ${RED}IMPORTANT: Start with Testnet keys (BYBIT_TESTNET=\"True\") and ensure API permissions are correct!${NC}"
echo -e "2. Review ${YELLOW}${CONFIG_FILE}${NC} carefully:"
echo -e "   - Set your desired ${BLUE}symbol${NC} using the CCXT unified format (e.g., BTC/USDT:USDT)."
echo -e "   - Choose the correct ${BLUE}timeframe${NC} (e.g., 5m, 1h)."
echo -e "   - Adjust risk (${BLUE}risk_per_trade_percent${NC}), ${BLUE}leverage${NC}, ${BLUE}tp_ratio${NC}."
echo -e "   - Configure strategy parameters under ${BLUE}strategy.params${NC}."
echo -e "   - Verify ${BLUE}account_type${NC} ('contract', 'unified', 'spot') matches your Bybit setup."
echo -e "   - Start with ${BLUE}mode${NC} set to \"Paper\" or ensure Testnet is enabled in .env."
echo -e "3. Install required Python packages:"
echo -e "   ${GREEN}python -m venv venv${NC}  # Create virtual environment (Recommended)"
echo -e "   ${GREEN}source venv/bin/activate${NC}  # Activate (Linux/macOS)"
echo -e "   ${GREEN}# venv\\Scripts\\activate${NC}  # Activate (Windows)"
echo -e "   ${GREEN}pip install -r ${REQ_FILE}${NC}"
echo -e "4. Run the bot:"
echo -e "   ${GREEN}python ${MAIN_FILE}${NC}"
echo -e "\n${BLUE}Bot setup complete! Remember to test thoroughly on Testnet/Paper mode before using real funds.${NC}"

# Optional: Make the setup script non-executable after running
# This prevents accidental re-runs that might overwrite customized files.
# chmod -x "$(basename "$0")"

exit 0
