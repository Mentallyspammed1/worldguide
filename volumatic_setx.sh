#!/bin/bash
# Purpose: Sets up the directory structure and initial files for the
#          Pyrmethus Volumatic Trend + OB trading bot.

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# Prevent errors in a pipeline from being masked.
set -euo pipefail

# --- Configuration ---
readonly PROJECT_DIR="pyrmethus_volumatic_bot"
readonly ENV_FILE=".env"
readonly CONFIG_FILE="config.json"
readonly STRATEGY_FILE="strategy.py"
readonly MAIN_FILE="main.py"
readonly REQ_FILE="requirements.txt"
# readonly SETUP_SCRIPT_NAME="${0##*/}" # Get the script's own name

# --- Colors for Output ---
# Using printf for better portability and control than echo -e
CLR_BLUE='\033[0;34m'
CLR_GREEN='\033[0;32m'
CLR_YELLOW='\033[1;33m'
CLR_RED='\033[0;31m'
CLR_NC='\033[0m' # No Color

# --- Helper Functions ---
info() {
    printf "${CLR_BLUE}%s${CLR_NC}\n" "$*" >&2
}

success() {
    printf "${CLR_GREEN}%s${CLR_NC}\n" "$*" >&2
}

warn() {
    printf "${CLR_YELLOW}WARN: %s${CLR_NC}\n" "$*" >&2
}

error() {
    printf "${CLR_RED}ERROR: %s${CLR_NC}\n" "$*" >&2
}

die() {
    error "$@"
    exit 1
}

# --- Main Script ---
info "Setting up Pyrmethus Volumatic Bot in directory: ${CLR_YELLOW}${PROJECT_DIR}${CLR_NC}"

# Create project directory
if mkdir -p "$PROJECT_DIR"; then
    success "Directory '$PROJECT_DIR' created or already exists."
else
    die "Could not create directory '$PROJECT_DIR'. Check permissions."
fi

# Change into the directory; die if it fails
cd "$PROJECT_DIR" || die "Could not change directory to '$PROJECT_DIR'."
info "Changed working directory to $(pwd)"

# --- Create requirements.txt ---
info "Creating ${CLR_YELLOW}${REQ_FILE}${CLR_NC}..."
cat << EOF > "$REQ_FILE"
# Core dependencies
pybit>=5.5.0 # Use a recent version of pybit V5
python-dotenv
pandas>=1.5.0 # Ensure modern pandas features
numpy
colorama

# Strategy specific
pandas-ta

# Optional but recommended
websocket-client>=1.3.0 # Often needed by pybit WS
requests # Good for potential health checks, external API calls etc.
EOF
success "${REQ_FILE} created."

# --- Create .env file ---
info "Creating ${CLR_YELLOW}${ENV_FILE}${CLR_NC} (Remember to fill in your API keys!)"
cat << EOF > "$ENV_FILE"
# Bybit API Credentials (use Testnet keys for development!)
# Register API keys here: https://www.bybit.com/app/user/api-management
BYBIT_API_KEY="YOUR_API_KEY_HERE"
BYBIT_API_SECRET="YOUR_API_SECRET_HERE"

# Optional: Set to True for Testnet, False or leave blank for Mainnet
BYBIT_TESTNET="True"
EOF
success "${ENV_FILE} created. ${CLR_YELLOW}Please edit it with your API credentials.${CLR_NC}"

# --- Create config.json ---
info "Creating ${CLR_YELLOW}${CONFIG_FILE}${CLR_NC}..."
cat << EOF > "$CONFIG_FILE"
{
  "symbol": "BTCUSDT",
  "interval": "5",
  "mode": "Live",
  "log_level": "INFO",

  "order": {
    "type": "Market",
    "risk_per_trade_percent": 1.0,
    "leverage": 5,
    "tp_ratio": 2.0
  },

  "strategy": {
    "class": "VolumaticOBStrategy",
    "params": {
      "length": 40,
      "vol_atr_period": 200,
      "vol_percentile_len": 1000,
      "vol_percentile": 100,
      "ob_source": "Wicks",
      "pivot_left_h": 10,
      "pivot_right_h": 10,
      "pivot_left_l": 10,
      "pivot_right_l": 10,
      "max_boxes": 5
    },
    "stop_loss": {
      "method": "ATR",
      "atr_multiplier": 2.0
    }
  },

  "data": {
      "fetch_limit": 750,
      "max_df_len": 2000
  },

  "websocket": {
      "ping_interval": 20,
      "connect_timeout": 10
  },
  "position_check_interval": 10
}
EOF
success "${CONFIG_FILE} created."

# --- Create strategy.py ---
info "Creating ${CLR_YELLOW}${STRATEGY_FILE}${CLR_NC}..."
# Use 'EOF' (with quotes) to prevent parameter expansion inside the heredoc
cat << 'EOF' > "$STRATEGY_FILE"
# -*- coding: utf-8 -*-
"""
Strategy Definition: Enhanced Volumatic Trend + OB
Calculates indicators, identifies order blocks, and generates trading signals.
"""
import logging
from typing import List, Dict, Optional, Any, TypedDict, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP

import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style

# --- Type Definitions ---
# Consistent type hints improve readability and allow static analysis.
class OrderBlock(TypedDict):
    id: int          # Unique identifier (e.g., index of the candle where OB formed)
    type: str        # 'bull' or 'bear'
    left_idx: int    # Index of bar where OB formed
    right_idx: int   # Index of last bar OB is valid for (updated if active)
    top: float       # Top price of the OB
    bottom: float    # Bottom price of the OB
    active: bool     # Still considered valid?
    closed_idx: Optional[int] # Index where it was invalidated/closed

class AnalysisResults(TypedDict):
    dataframe: pd.DataFrame         # DataFrame with indicators
    last_signal: str                # 'BUY', 'SELL', 'HOLD', 'EXIT_LONG', 'EXIT_SHORT'
    active_bull_boxes: List[OrderBlock]
    active_bear_boxes: List[OrderBlock]
    last_close: float               # Last closing price
    current_trend: Optional[bool]   # True for UP, False for DOWN, None if undetermined
    trend_changed: bool             # True if trend changed on the last candle
    last_atr: Optional[float]       # Last calculated ATR value (for SL/TP)

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
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}") # Specific logger
        if not market_info:
            raise ValueError("Market info (tickSize, qtyStep) is required for strategy initialization.")
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
        self.current_trend: Optional[bool] = None # True=UP, False=DOWN

        # Calculate minimum required data length based on indicator periods
        self.min_data_len = max(
            self.length + 4, # For _ema_swma shift
            self.vol_atr_period,
            self.vol_percentile_len,
            self.pivot_left_h + self.pivot_right_h + 1, # Pivot lookback/forward
            self.pivot_left_l + self.pivot_right_l + 1
        )

        # Extract precision details from market_info using Decimal for accuracy
        try:
            self.tick_size = Decimal(self.market_info['priceFilter']['tickSize'])
            self.qty_step = Decimal(self.market_info['lotSizeFilter']['qtyStep'])
        except (KeyError, TypeError) as e:
            self.log.error(f"Failed to extract tickSize/qtyStep from market_info: {e}")
            raise ValueError("Market info missing required price/lot filter details.") from e

        self.price_precision = self._get_decimal_places(self.tick_size)
        self.qty_precision = self._get_decimal_places(self.qty_step)

        self.log.info(f"{Fore.MAGENTA}Initializing VolumaticOB Strategy Engine...{Style.RESET_ALL}")
        self.log.info(f"Params: TrendLen={self.length}, Pivots={self.pivot_left_h}/{self.pivot_right_h}, "
                      f"MaxBoxes={self.max_boxes}, OB Source={self.ob_source}")
        self.log.info(f"Minimum data points required: {self.min_data_len}")
        self.log.debug(f"Tick Size: {self.tick_size}, Qty Step: {self.qty_step}")
        self.log.debug(f"Price Precision: {self.price_precision}, Qty Precision: {self.qty_precision}")

    def _parse_params(self, params: Dict[str, Any]):
        """Load and type-cast parameters from the config dictionary."""
        self.length = int(params.get('length', 40))
        self.vol_atr_period = int(params.get('vol_atr_period', 200))
        self.vol_percentile_len = int(params.get('vol_percentile_len', 1000))
        self.vol_percentile = int(params.get('vol_percentile', 100))
        self.ob_source = str(params.get('ob_source', "Wicks"))
        self.pivot_left_h = int(params.get('pivot_left_h', 10))
        self.pivot_right_h = int(params.get('pivot_right_h', 10))
        self.pivot_left_l = int(params.get('pivot_left_l', 10))
        self.pivot_right_l = int(params.get('pivot_right_l', 10))
        self.max_boxes = int(params.get('max_boxes', 5))
        # Load SL params too
        sl_config = params.get('stop_loss', {})
        self.sl_method = str(sl_config.get('method', 'ATR'))
        self.sl_atr_multiplier = float(sl_config.get('atr_multiplier', 2.0))

    def _validate_params(self):
        """Perform basic validation of strategy parameters."""
        if self.ob_source not in ["Wicks", "Bodys"]:
            raise ValueError("Invalid 'ob_source'. Must be 'Wicks' or 'Bodys'.")
        lengths = [self.length, self.vol_atr_period, self.vol_percentile_len,
                   self.pivot_left_h, self.pivot_right_h, self.pivot_left_l, self.pivot_right_l]
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
        return abs(decimal_val.as_tuple().exponent) if decimal_val.is_finite() and decimal_val.as_tuple().exponent < 0 else 0

    def round_price(self, price: float, rounding_mode=ROUND_DOWN) -> float:
        """
        Rounds a price according to the market's tickSize.
        Uses ROUND_DOWN by default, suitable for sell SL/TP or conservative entries.
        Use ROUND_UP for buy SL/TP.
        """
        if not isinstance(price, (float, int)) or not np.isfinite(price):
            self.log.warning(f"Invalid price value for rounding: {price}. Returning NaN.")
            return np.nan
        try:
            # Use Decimal for precise rounding based on tick_size
            return float(Decimal(str(price)).quantize(self.tick_size, rounding=rounding_mode))
        except Exception as e:
            self.log.error(f"Error rounding price {price} with tick_size {self.tick_size}: {e}")
            return np.nan # Return NaN on error

    def round_qty(self, qty: float) -> float:
        """
        Rounds quantity DOWN according to the market's qtyStep.
        Ensures the quantity is a valid multiple of the minimum step.
        """
        if not isinstance(qty, (float, int)) or not np.isfinite(qty) or qty < 0:
            self.log.warning(f"Invalid quantity value for rounding: {qty}. Returning 0.")
            return 0.0
        try:
            # Use Decimal division and multiplication to round down to the nearest step
            qty_decimal = Decimal(str(qty))
            rounded_qty = (qty_decimal // self.qty_step) * self.qty_step
            return float(rounded_qty)
        except Exception as e:
            self.log.error(f"Error rounding quantity {qty} with qty_step {self.qty_step}: {e}")
            return 0.0 # Return 0 on error

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
        swma = series.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights) if not np.isnan(x).any() else np.nan, raw=True)

        # Calculate EMA of the SWMA result
        # Use adjust=False for behavior closer to some TA libraries/platforms
        ema_of_swma = ta.ema(swma.dropna(), length=length, adjust=False)

        # Reindex to match original series index, filling gaps with NaN
        return ema_of_swma.reindex(series.index)

    def _find_pivots(self, df: pd.DataFrame, left: int, right: int, is_high: bool) -> pd.Series:
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
             source_col = 'high' if is_high else 'low'
        else: # Bodys
             # Use close for pivot highs, open for pivot lows (common interpretation)
             source_col = 'close' if is_high else 'open'

        if source_col not in df.columns:
            self.log.error(f"Source column '{source_col}' not found in DataFrame for pivot calculation.")
            return pd.Series(np.nan, index=df.index)

        source_series = df[source_col]
        pivots = pd.Series(np.nan, index=df.index)

        # Efficient vectorized approach (faster than Python loop for large data)
        # Pad series to handle boundary conditions during shifts easily
        padded_source = pd.concat([pd.Series([np.nan] * left), source_series, pd.Series([np.nan] * right)])

        # Check left side: current value must be strictly greater/less than left neighbors
        left_check = True
        for i in range(1, left + 1):
            shifted = padded_source.shift(i)
            if is_high:
                left_check &= (padded_source > shifted)
            else:
                left_check &= (padded_source < shifted)

        # Check right side: current value must be greater/less than or equal to right neighbors
        right_check = True
        for i in range(1, right + 1):
            shifted = padded_source.shift(-i)
            if is_high:
                right_check &= (padded_source >= shifted)
            else:
                right_check &= (padded_source <= shifted)

        # Combine checks and align back to original index
        is_pivot = (left_check & right_check).iloc[left:-right] # Remove padding
        is_pivot.index = df.index # Align index

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
            self.log.warning(f"Not enough data ({len(df_input)}/{self.min_data_len}) for analysis. Returning current state.")
            # Return current state without modifying the input DataFrame
            return AnalysisResults(
                dataframe=df_input, # Return original df
                last_signal="HOLD",
                active_bull_boxes=[b for b in self.bull_boxes if b['active']], # Return copies of active boxes
                active_bear_boxes=[b for b in self.bear_boxes if b['active']],
                last_close=df_input['close'].iloc[-1] if not df_input.empty else np.nan,
                current_trend=self.current_trend,
                trend_changed=False,
                last_atr=None # Cannot calculate ATR with insufficient data
            )

        # Work on a copy to avoid modifying the original DataFrame passed in
        df = df_input.copy()
        self.log.debug(f"Analyzing {len(df)} candles ending {df.index[-1]}...")

        # --- Volumatic Trend Calculations ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.vol_atr_period)
        df['ema1'] = self._ema_swma(df['close'], length=self.length)
        df['ema2'] = ta.ema(df['close'], length=self.length, adjust=False) # Use adjust=False

        # Determine trend direction (True=UP, False=DOWN)
        # Use np.select for clarity, ffill to handle initial NaNs
        conditions = [df['ema1'] < df['ema2'], df['ema1'] >= df['ema2']]
        choices = [True, False]
        df['trend_up'] = np.select(conditions, choices, default=np.nan)
        df['trend_up'] = df['trend_up'].ffill() # Forward fill trend after initial calculation

        # Detect trend change, ignoring NaNs and the very first valid trend
        df['trend_changed'] = (df['trend_up'] != df['trend_up'].shift(1)) & df['trend_up'].notna() & df['trend_up'].shift(1).notna()

        # --- Update Levels on Trend Change ---
        last_row = df.iloc[-1]
        current_trend_up = last_row['trend_up']
        trend_just_changed = last_row['trend_changed']
        last_atr_value = last_row['atr']
        current_ema1 = last_row['ema1']

        # Update persistent trend state if the trend value is valid
        if pd.notna(current_trend_up):
            is_initial_trend = self.current_trend is None
            if is_initial_trend:
                self.current_trend = current_trend_up
                self.log.info(f"Initial Trend detected: {'UP' if self.current_trend else 'DOWN'}")
                trend_just_changed = True # Force level update on first detection

            elif trend_just_changed and current_trend_up != self.current_trend:
                self.current_trend = current_trend_up # Update internal trend state
                self.log.info(f"{Fore.MAGENTA}Trend Changed! New Trend: {'UP' if current_trend_up else 'DOWN'}. Updating levels...{Style.RESET_ALL}")
                # Level update logic moved inside the trend change block

            # Update levels IF trend just changed (or it's the initial trend) AND necessary values are valid
            if trend_just_changed and pd.notna(current_ema1) and pd.notna(last_atr_value) and last_atr_value > 1e-9:
                self.upper = current_ema1 + last_atr_value * 3
                self.lower = current_ema1 - last_atr_value * 3
                self.lower_vol = self.lower + last_atr_value * 4
                self.upper_vol = self.upper - last_atr_value * 4
                # Prevent levels from crossing due to large ATR or calculation quirks
                self.lower_vol = max(self.lower_vol, self.lower)
                self.upper_vol = min(self.upper_vol, self.upper)

                self.step_up = (self.lower_vol - self.lower) / 100 if self.lower_vol > self.lower else 0
                self.step_dn = (self.upper - self.upper_vol) / 100 if self.upper > self.upper_vol else 0

                self.log.info(f"Levels Updated @ {df.index[-1]}: U={self.upper:.{self.price_precision}f}, L={self.lower:.{self.price_precision}f}")
            elif trend_just_changed:
                 self.log.warning(f"Could not update levels at {df.index[-1]} due to NaN/zero values (EMA1={current_ema1}, ATR={last_atr_value}). Levels remain unchanged or reset.")
                 # Optionally reset levels if update fails:
                 # self.upper, self.lower, self.lower_vol, self.upper_vol, self.step_up, self.step_dn = [None] * 6

        # --- Volume Normalization ---
        # Calculate rolling percentile of volume
        roll_window = min(self.vol_percentile_len, len(df))
        min_periods_vol = max(1, roll_window // 2) # Require a reasonable number of periods
        # Use np.nanpercentile for robustness against NaNs within the window
        # Rolling apply can be slow; consider alternatives if needed
        df['vol_percentile_val'] = df['volume'].rolling(window=roll_window, min_periods=min_periods_vol).apply(
            lambda x: np.nanpercentile(x[x > 0], self.vol_percentile) if np.any(x > 0) else np.nan,
            raw=True # raw=True might be faster
        )

        # Normalize volume based on the percentile value
        df['vol_norm'] = np.where(
            (df['vol_percentile_val'].notna()) & (df['vol_percentile_val'] > 1e-9), # Avoid division by zero/NaN
            (df['volume'] / df['vol_percentile_val'] * 100),
            0 # Assign 0 if percentile is NaN or (near) zero
        )
        df['vol_norm'] = df['vol_norm'].fillna(0).astype(float) # Ensure float type and fill any remaining NaNs

        # --- Pivot Order Block Calculations ---
        df['ph'] = self._find_pivots(df, self.pivot_left_h, self.pivot_right_h, is_high=True)
        df['pl'] = self._find_pivots(df, self.pivot_left_l, self.pivot_right_l, is_high=False)

        # --- Create and Manage Order Blocks ---
        # Use integer indices for easier list management and box IDs
        df['int_index'] = range(len(df))
        current_bar_int_idx = len(df) - 1

        # Check for newly formed pivots in the recent data
        # A pivot at index `p` is confirmed `right` bars later (at index `p + right`).
        # We only need to check bars where pivots might have just been confirmed.
        check_start_idx = max(0, len(df) - max(self.pivot_right_h, self.pivot_right_l) - 5) # Check recent bars + buffer
        new_boxes_created_count = 0

        for i in range(check_start_idx, len(df)):
            # Check for Bearish Box (Pivot High confirmation)
            # Pivot occurred at index `i - pivot_right_h`
            bear_pivot_occur_idx = i - self.pivot_right_h
            if bear_pivot_occur_idx >= 0 and pd.notna(df['ph'].iloc[bear_pivot_occur_idx]):
                # Check if a box for this pivot index already exists
                if not any(b['id'] == bear_pivot_occur_idx for b in self.bear_boxes):
                    ob_candle = df.iloc[bear_pivot_occur_idx]
                    top_price, bottom_price = np.nan, np.nan

                    if self.ob_source == "Wicks":
                        top_price = ob_candle['high']
                        # Common definition: use close for bear OB bottom (body top)
                        bottom_price = max(ob_candle['open'], ob_candle['close'])
                    else: # Bodys
                        top_price = max(ob_candle['open'], ob_candle['close'])
                        bottom_price = min(ob_candle['open'], ob_candle['close'])

                    # Ensure valid prices and top >= bottom
                    if pd.notna(top_price) and pd.notna(bottom_price) and top_price >= bottom_price:
                        # Check for zero-range OB (e.g., doji) - allow small range based on tick size
                        if abs(top_price - bottom_price) > float(self.tick_size / 10): # Allow OB if range > 1/10th tick
                            new_box = OrderBlock(
                                id=bear_pivot_occur_idx, type='bear', left_idx=bear_pivot_occur_idx,
                                right_idx=current_bar_int_idx, # Valid up to current bar
                                top=top_price, bottom=bottom_price, active=True, closed_idx=None
                            )
                            self.bear_boxes.append(new_box)
                            new_boxes_created_count += 1
                            self.log.info(f"{Fore.RED}New Bear OB {bear_pivot_occur_idx} @ {df.index[bear_pivot_occur_idx]}: T={top_price:.{self.price_precision}f}, B={bottom_price:.{self.price_precision}f}{Style.RESET_ALL}")

            # Check for Bullish Box (Pivot Low confirmation)
            # Pivot occurred at index `i - pivot_right_l`
            bull_pivot_occur_idx = i - self.pivot_right_l
            if bull_pivot_occur_idx >= 0 and pd.notna(df['pl'].iloc[bull_pivot_occur_idx]):
                # Check if a box for this pivot index already exists
                if not any(b['id'] == bull_pivot_occur_idx for b in self.bull_boxes):
                    ob_candle = df.iloc[bull_pivot_occur_idx]
                    top_price, bottom_price = np.nan, np.nan

                    if self.ob_source == "Wicks":
                         # Common definition: use open for bull OB top (body bottom)
                         top_price = min(ob_candle['open'], ob_candle['close'])
                         bottom_price = ob_candle['low']
                    else: # Bodys
                         top_price = max(ob_candle['open'], ob_candle['close'])
                         bottom_price = min(ob_candle['open'], ob_candle['close'])

                    if pd.notna(top_price) and pd.notna(bottom_price) and top_price >= bottom_price:
                        if abs(top_price - bottom_price) > float(self.tick_size / 10):
                            new_box = OrderBlock(
                                id=bull_pivot_occur_idx, type='bull', left_idx=bull_pivot_occur_idx,
                                right_idx=current_bar_int_idx,
                                top=top_price, bottom=bottom_price, active=True, closed_idx=None
                            )
                            self.bull_boxes.append(new_box)
                            new_boxes_created_count += 1
                            self.log.info(f"{Fore.GREEN}New Bull OB {bull_pivot_occur_idx} @ {df.index[bull_pivot_occur_idx]}: T={top_price:.{self.price_precision}f}, B={bottom_price:.{self.price_precision}f}{Style.RESET_ALL}")

        if new_boxes_created_count > 0:
             self.log.debug(f"Created {new_boxes_created_count} new order blocks.")

        # --- Manage existing boxes (close or extend validity) ---
        current_close = last_row['close']
        closed_bull_count = 0
        closed_bear_count = 0

        if pd.notna(current_close): # Only manage boxes if close price is valid
            for box in self.bull_boxes:
                if box['active']:
                    # Invalidate Bull OB if price closes below its bottom
                    if current_close < box['bottom']:
                        box['active'] = False
                        box['closed_idx'] = current_bar_int_idx
                        closed_bull_count += 1
                    else:
                        # Otherwise, extend its validity to the current bar
                        box['right_idx'] = current_bar_int_idx

            for box in self.bear_boxes:
                if box['active']:
                    # Invalidate Bear OB if price closes above its top
                    if current_close > box['top']:
                        box['active'] = False
                        box['closed_idx'] = current_bar_int_idx
                        closed_bear_count +=1
                    else:
                        box['right_idx'] = current_bar_int_idx

            if closed_bull_count > 0: self.log.info(f"{Fore.YELLOW}Closed {closed_bull_count} Bull OBs due to price violation.{Style.RESET_ALL}")
            if closed_bear_count > 0: self.log.info(f"{Fore.YELLOW}Closed {closed_bear_count} Bear OBs due to price violation.{Style.RESET_ALL}")

        # --- Prune Order Blocks ---
        # Keep only the 'max_boxes' most recent *active* boxes of each type.
        # Keep a limited number of recent inactive ones for context/debugging.
        active_bull = sorted([b for b in self.bull_boxes if b['active']], key=lambda x: x['id'], reverse=True)
        inactive_bull = sorted([b for b in self.bull_boxes if not b['active']], key=lambda x: x['id'], reverse=True)
        initial_bull_len = len(self.bull_boxes)
        # Keep max_boxes active + double that number inactive (most recent ones)
        self.bull_boxes = active_bull[:self.max_boxes] + inactive_bull[:self.max_boxes * 2]
        if len(self.bull_boxes) < initial_bull_len: self.log.debug(f"Pruned {initial_bull_len - len(self.bull_boxes)} older Bull OBs.")

        active_bear = sorted([b for b in self.bear_boxes if b['active']], key=lambda x: x['id'], reverse=True)
        inactive_bear = sorted([b for b in self.bear_boxes if not b['active']], key=lambda x: x['id'], reverse=True)
        initial_bear_len = len(self.bear_boxes)
        self.bear_boxes = active_bear[:self.max_boxes] + inactive_bear[:self.max_boxes * 2]
        if len(self.bear_boxes) < initial_bear_len: self.log.debug(f"Pruned {initial_bear_len - len(self.bear_boxes)} older Bear OBs.")

        # --- Signal Generation ---
        signal = "HOLD" # Default signal
        active_bull_boxes = [b for b in self.bull_boxes if b['active']] # Get currently active boxes
        active_bear_boxes = [b for b in self.bear_boxes if b['active']]

        # Check conditions only if trend and close price are valid
        if self.current_trend is not None and pd.notna(current_close):
            # 1. Check for Trend Change Exit first
            if trend_just_changed:
                # Check internal state to see if we were in a position that needs exiting
                if not self.current_trend and self.last_signal_state == "BUY": # Trend flipped DOWN while intended LONG
                    signal = "EXIT_LONG"
                    self.log.warning(f"{Fore.YELLOW}{Style.BRIGHT}*** EXIT LONG Signal (Trend Flip to DOWN) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}")
                elif self.current_trend and self.last_signal_state == "SELL": # Trend flipped UP while intended SHORT
                    signal = "EXIT_SHORT"
                    self.log.warning(f"{Fore.YELLOW}{Style.BRIGHT}*** EXIT SHORT Signal (Trend Flip to UP) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}")

            # 2. Check for Entries only if not exiting and not already in the desired state
            if signal == "HOLD": # Only look for entries if not exiting
                if self.current_trend: # Trend is UP -> Look for Long Entries
                    if self.last_signal_state != "BUY": # Only enter if not already intending long
                        # Check if price entered any active Bull OB
                        for box in active_bull_boxes:
                            # Check if close is within the box range (inclusive)
                            if box['bottom'] <= current_close <= box['top']:
                                signal = "BUY"
                                self.log.info(f"{Fore.GREEN}{Style.BRIGHT}*** BUY Signal (Trend UP + Price in Bull OB {box['id']}) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}")
                                break # Take the first valid signal

                elif not self.current_trend: # Trend is DOWN -> Look for Short Entries
                    if self.last_signal_state != "SELL": # Only enter if not already intending short
                        # Check if price entered any active Bear OB
                        for box in active_bear_boxes:
                            if box['bottom'] <= current_close <= box['top']:
                                signal = "SELL"
                                self.log.info(f"{Fore.RED}{Style.BRIGHT}*** SELL Signal (Trend DOWN + Price in Bear OB {box['id']}) at {current_close:.{self.price_precision}f} ***{Style.RESET_ALL}")
                                break # Take the first valid signal

        # Update internal state *only* if signal implies a state change
        if signal == "BUY":
            self.last_signal_state = "BUY"
        elif signal == "SELL":
            self.last_signal_state = "SELL"
        elif signal in ["EXIT_LONG", "EXIT_SHORT"]:
            self.last_signal_state = "HOLD" # After exit, we are neutral until a new entry signal

        self.log.debug(f"Signal Generated: {signal} (Internal State: {self.last_signal_state})")

        # Drop the temporary integer index column before returning
        df.drop(columns=['int_index'], inplace=True, errors='ignore')

        # Return results
        return AnalysisResults(
            dataframe=df, # Return the DataFrame with calculated indicators
            last_signal=signal,
            active_bull_boxes=active_bull_boxes, # Return only active boxes
            active_bear_boxes=active_bear_boxes,
            last_close=current_close if pd.notna(current_close) else np.nan,
            current_trend=self.current_trend,
            trend_changed=trend_just_changed,
            last_atr=last_atr_value if pd.notna(last_atr_value) else None
        )
EOF
success "${STRATEGY_FILE} created."

# --- Create main.py ---
info "Creating ${CLR_YELLOW}${MAIN_FILE}${CLR_NC}..."
# Use 'EOF' (with quotes) to prevent parameter expansion inside the heredoc
cat << 'EOF' > "$MAIN_FILE"
# -*- coding: utf-8 -*-
"""
Pyrmethus Volumatic Trend + OB Trading Bot for Bybit V5 API

Connects to Bybit, fetches data, runs the strategy analysis,
and executes trades based on generated signals.
Includes WebSocket integration for real-time candle updates.
"""
import os
import sys
import json
import time
import datetime
import logging
import signal
import threading
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation
from typing import Dict, Optional, Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from colorama import init, Fore, Style
from pybit.unified_trading import HTTP, WebSocket

# Import strategy class and type hints from strategy.py
try:
    from strategy import VolumaticOBStrategy, AnalysisResults
except ImportError as e:
    print(f"ERROR: Could not import from strategy.py: {e}", file=sys.stderr)
    print("Ensure strategy.py is in the same directory.", file=sys.stderr)
    sys.exit(1)

# --- Initialize Colorama ---
# autoreset=True ensures color resets after each print
init(autoreset=True)

# --- Load Environment Variables ---
# Load API keys and settings from .env file in the same directory
if load_dotenv():
    print("Loaded environment variables from .env file.")
else:
    print("WARN: .env file not found or empty. API keys might be missing.")

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
TESTNET = os.getenv("BYBIT_TESTNET", "False").lower() in ("true", "1", "t")

# --- Global Variables & State ---
config: Dict[str, Any] = {}
session: Optional[HTTP] = None
ws: Optional[WebSocket] = None
ws_thread: Optional[threading.Thread] = None
ws_connected = threading.Event() # Signals WebSocket connection status
stop_event = threading.Event() # Used for graceful shutdown coordination
latest_dataframe: Optional[pd.DataFrame] = None
strategy_instance: Optional[VolumaticOBStrategy] = None
market_info: Optional[Dict[str, Any]] = None
data_lock = threading.Lock() # Protects access to latest_dataframe
# position_lock = threading.Lock() # Use if modifying position state based on WS (currently not)
order_lock = threading.Lock() # Protects order placement/closing logic to prevent race conditions
last_position_check_time: float = 0
POSITION_CHECK_INTERVAL: int = 10 # Default, overridden by config
# --- Logging Setup ---
# Use a single logger instance throughout the application
log = logging.getLogger("PyrmethusVolumaticBot")
log_level = logging.INFO # Default, will be overridden by config

def setup_logging(level_str="INFO"):
    """Configures logging format, level, and handlers."""
    global log_level
    try:
        log_level = getattr(logging, level_str.upper())
    except AttributeError:
        log_level = logging.INFO
        print(f"WARN: Invalid log level '{level_str}'. Defaulting to INFO.", file=sys.stderr)

    log.setLevel(log_level)

    # Prevent adding multiple handlers if called again
    if not log.handlers:
        # Console Handler (StreamHandler)
        ch = logging.StreamHandler(sys.stdout) # Use stdout for console logs
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        log.addHandler(ch)

        # Optional: File Handler
        # log_filename = f"bot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        # try:
        #     fh = logging.FileHandler(log_filename)
        #     fh.setLevel(log_level)
        #     fh.setFormatter(formatter)
        #     log.addHandler(fh)
        # except Exception as e:
        #     log.error(f"Failed to set up file logging to {log_filename}: {e}")

    # Prevent messages from propagating to the root logger if handlers are set
    log.propagate = False

# --- Configuration Loading ---
def load_config(path="config.json") -> Dict[str, Any]:
    """Loads and validates the configuration from a JSON file."""
    global POSITION_CHECK_INTERVAL
    try:
        with open(path, 'r') as f:
            conf = json.load(f)
        log.info(f"Configuration loaded successfully from '{path}'.")

        # Basic validation (add more checks as needed)
        required_keys = ["symbol", "interval", "mode", "log_level", "order", "strategy", "data", "websocket"]
        if not all(key in conf for key in required_keys):
            raise ValueError(f"Config file missing one or more required top-level keys: {required_keys}")
        if not all(key in conf["order"] for key in ["risk_per_trade_percent", "leverage"]):
             raise ValueError("Config file missing required keys in 'order' section.")
        if not all(key in conf["strategy"] for key in ["params", "stop_loss"]):
             raise ValueError("Config file missing required keys in 'strategy' section.")
        if not all(key in conf["data"] for key in ["fetch_limit", "max_df_len"]):
             raise ValueError("Config file missing required keys in 'data' section.")

        # Update global interval if set in config
        POSITION_CHECK_INTERVAL = int(conf.get("position_check_interval", 10))
        if POSITION_CHECK_INTERVAL <= 0:
             log.warning("position_check_interval must be positive. Using default 10s.")
             POSITION_CHECK_INTERVAL = 10

        return conf
    except FileNotFoundError:
        log.critical(f"CRITICAL: Configuration file '{path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log.critical(f"CRITICAL: Configuration file '{path}' contains invalid JSON: {e}")
        sys.exit(1)
    except ValueError as e:
        log.critical(f"CRITICAL: Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        log.critical(f"CRITICAL: Unexpected error loading configuration: {e}", exc_info=True)
        sys.exit(1)

# --- Bybit API Interaction ---
def connect_bybit() -> Optional[HTTP]:
    """Establishes and tests connection to Bybit HTTP API."""
    if not API_KEY or not API_SECRET:
        log.critical("CRITICAL: Bybit API Key or Secret not found. Check .env file or environment variables.")
        sys.exit(1)
    try:
        log.info(f"Connecting to Bybit {'Testnet' if TESTNET else 'Mainnet'} HTTP API...")
        s = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET)
        # Test connection with a simple read-only call
        server_time_resp = s.get_server_time()
        if server_time_resp.get('retCode') == 0:
            server_ts = int(server_time_resp['result']['timeNano']) / 1e9
            log.info(f"Successfully connected. Server time: {datetime.datetime.fromtimestamp(server_ts)}")
            return s
        else:
            log.critical(f"CRITICAL: Failed to connect or verify connection: {server_time_resp.get('retMsg', 'Unknown Error')}")
            return None
    except Exception as e:
        log.critical(f"CRITICAL: Exception during Bybit HTTP API connection: {e}", exc_info=True)
        return None

def get_market_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches and validates instrument information for the specified symbol."""
    if not session:
        log.error("HTTP session not available for get_market_info.")
        return None
    try:
        log.debug(f"Fetching market info for {symbol}...")
        response = session.get_instruments_info(category="linear", symbol=symbol)
        if response.get('retCode') == 0 and response['result'].get('list'):
            info = response['result']['list'][0]
            log.info(f"Fetched market info for {symbol}.")
            log.debug(f"Market Info Details: {json.dumps(info, indent=2)}")

            # Validate presence of required filter details for precision and order limits
            price_filter = info.get('priceFilter', {})
            lot_filter = info.get('lotSizeFilter', {})
            if not price_filter.get('tickSize') or \
               not lot_filter.get('qtyStep') or \
               not lot_filter.get('minOrderQty') or \
               not lot_filter.get('maxOrderQty'):
                log.error(f"Market info for {symbol} is missing required filter details "
                          "(tickSize, qtyStep, minOrderQty, maxOrderQty). Cannot proceed.")
                return None
            # Convert to Decimal early for consistency
            try:
                 info['priceFilter']['tickSize'] = Decimal(price_filter['tickSize'])
                 info['lotSizeFilter']['qtyStep'] = Decimal(lot_filter['qtyStep'])
                 info['lotSizeFilter']['minOrderQty'] = Decimal(lot_filter['minOrderQty'])
                 info['lotSizeFilter']['maxOrderQty'] = Decimal(lot_filter['maxOrderQty'])
            except (InvalidOperation, TypeError) as e:
                 log.error(f"Could not convert market info filter values to Decimal for {symbol}: {e}")
                 return None

            return info
        else:
            log.error(f"Failed to get market info for {symbol}: {response.get('retMsg', 'Unknown Error')} "
                      f"(Code: {response.get('retCode', 'N/A')})")
            return None
    except Exception as e:
        # Log traceback only if debug level is enabled
        log.error(f"Exception fetching market info for {symbol}: {e}", exc_info=(log_level <= logging.DEBUG))
        return None

def fetch_initial_data(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetches historical Klines data and prepares the initial DataFrame."""
    if not session:
         log.error("HTTP session not available for fetch_initial_data.")
         return None
    log.info(f"Fetching initial {limit} klines for {symbol} (interval: {interval})...")
    try:
        response = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        if response.get('retCode') == 0 and response['result'].get('list'):
            kline_list = response['result']['list']
            if not kline_list:
                 log.warning(f"Received empty kline list from Bybit for initial fetch of {symbol}/{interval}.")
                 # Return an empty DataFrame with expected columns
                 return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'turnover'])

            df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            # Convert types immediately after creation
            df = df.astype({
                'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
                'low': 'float64', 'close': 'float64', 'volume': 'float64', 'turnover': 'float64'
            })
            # Convert timestamp (milliseconds) to DatetimeIndex
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            # Bybit V5 returns oldest first, so sort chronologically just in case
            df = df.sort_index()
            log.info(f"Fetched {len(df)} initial candles. Data spans from {df.index.min()} to {df.index.max()}")
            return df
        else:
            log.error(f"Failed to fetch initial klines for {symbol}/{interval}: {response.get('retMsg', 'Error')} "
                      f"(Code: {response.get('retCode', 'N/A')})")
            return None
    except Exception as e:
        log.error(f"Exception fetching initial klines: {e}", exc_info=(log_level <= logging.DEBUG))
        return None

def get_current_position(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetches current position details for the symbol. Includes rate limiting.
    Returns a dictionary representing the position, or None if rate limited/error.
    A 'flat' position is represented with size 0 and side 'None'.
    """
    global last_position_check_time
    if not session:
        log.error("HTTP session not available for get_current_position.")
        # Return a representation of a flat position on session error
        return {"size": Decimal(0), "side": "None", "avgPrice": Decimal(0), "liqPrice": Decimal(0), "unrealisedPnl": Decimal(0)}

    now = time.time()
    # Check if the interval has passed since the last check
    if now - last_position_check_time < POSITION_CHECK_INTERVAL:
        # log.debug("Skipping position check due to rate limit.")
        return None # Indicate check was skipped

    log.debug(f"Fetching position for {symbol}...")
    try:
        response = session.get_positions(category="linear", symbol=symbol)
        # Update check time regardless of success/failure to prevent spamming on errors
        last_position_check_time = time.time()

        if response.get('retCode') == 0 and response['result'].get('list'):
            # Bybit returns a list, even when filtering by symbol. Assume first entry is the relevant one.
            position = response['result']['list'][0]
            # Convert relevant fields to Decimal for precise calculations
            pos_data = {
                "size": Decimal(position.get('size', '0')),
                "side": position.get('side', 'None'), # 'Buy', 'Sell', or 'None'
                "avgPrice": Decimal(position.get('avgPrice', '0')),
                "liqPrice": Decimal(position.get('liqPrice', '0')) if position.get('liqPrice') else Decimal(0),
                "unrealisedPnl": Decimal(position.get('unrealisedPnl', '0')),
                "markPrice": Decimal(position.get('markPrice', '0')), # Useful for context
                "leverage": Decimal(position.get('leverage', '0')), # Confirm leverage setting
                # Add other fields if needed
            }
            log.debug(f"Position Data: Size={pos_data['size']}, Side={pos_data['side']}, AvgPrice={pos_data['avgPrice']}")
            return pos_data
        elif response.get('retCode') == 110001: # Parameter error (e.g., invalid symbol)
             log.error(f"Parameter error fetching position for {symbol}. Is symbol valid? {response.get('retMsg', '')}")
             # Assume flat on symbol error to prevent incorrect actions
             return {"size": Decimal(0), "side": "None", "avgPrice": Decimal(0), "liqPrice": Decimal(0), "unrealisedPnl": Decimal(0)}
        else:
            log.error(f"Failed to get position for {symbol}: {response.get('retMsg', 'Error')} "
                      f"(Code: {response.get('retCode', 'N/A')})")
            # Return flat representation on API error, but log severity
            return {"size": Decimal(0), "side": "None", "avgPrice": Decimal(0), "liqPrice": Decimal(0), "unrealisedPnl": Decimal(0)}
    except Exception as e:
        last_position_check_time = time.time() # Update time even on exception
        log.error(f"Exception fetching position for {symbol}: {e}", exc_info=(log_level <= logging.DEBUG))
        # Return flat representation on exception
        return {"size": Decimal(0), "side": "None", "avgPrice": Decimal(0), "liqPrice": Decimal(0), "unrealisedPnl": Decimal(0)}

def get_wallet_balance(account_type="UNIFIED", coin="USDT") -> Optional[Decimal]:
    """Fetches account equity for risk calculation (UNIFIED account type)."""
    if not session:
        log.error("HTTP session not available for get_wallet_balance.")
        return None
    try:
        # V5 Unified Trading uses get_wallet_balance
        response = session.get_wallet_balance(accountType=account_type, coin=coin)
        if response.get('retCode') == 0 and response['result'].get('list'):
            # Unified account balance info is usually in the first item of the list
            balance_info = response['result']['list'][0]
            # Use 'equity' as the basis for risk calculation in Unified account
            if 'equity' in balance_info:
                equity = Decimal(balance_info['equity'])
                log.debug(f"Account Equity ({coin}): {equity}")
                if equity < 0:
                    log.warning(f"Account equity is negative: {equity}")
                return equity
            else:
                log.warning(f"Could not find 'equity' field in wallet balance response for {account_type} account.")
                # Fallback: Try 'totalAvailableBalance'? Less accurate for risk based on margin.
                if 'totalAvailableBalance' in balance_info:
                     avail_balance = Decimal(balance_info['totalAvailableBalance'])
                     log.warning(f"Falling back to totalAvailableBalance: {avail_balance}")
                     return avail_balance
                log.error("Neither 'equity' nor 'totalAvailableBalance' found in balance response.")
                return None
        else:
            log.error(f"Failed to get wallet balance: {response.get('retMsg', 'Unknown Error')} "
                      f"(Code: {response.get('retCode', 'N/A')})")
            return None
    except Exception as e:
        log.error(f"Exception fetching wallet balance: {e}", exc_info=(log_level <= logging.DEBUG))
        return None

def calculate_order_qty(entry_price: float, sl_price: float, risk_percent: float) -> Optional[float]:
    """
    Calculates order quantity based on risk percentage, SL distance, and account equity.
    Rounds down to the nearest valid quantity step and checks against min/max limits.
    """
    if not market_info or not strategy_instance:
        log.error("Market info or strategy instance not available for quantity calculation.")
        return None
    if not all(isinstance(p, (float, int)) and np.isfinite(p) for p in [entry_price, sl_price, risk_percent]):
        log.error(f"Invalid input for quantity calculation: entry={entry_price}, sl={sl_price}, risk%={risk_percent}")
        return None

    try:
        entry_decimal = Decimal(str(entry_price))
        sl_decimal = Decimal(str(sl_price))
        tick_size = strategy_instance.tick_size

        # Ensure SL is meaningfully different from entry
        if abs(sl_decimal - entry_decimal) < tick_size:
            log.error(f"Stop loss price {sl_price:.{strategy_instance.price_precision}f} is too close to entry price "
                      f"{entry_price:.{strategy_instance.price_precision}f} (tick size: {tick_size}). Cannot calculate quantity.")
            return None

        balance = get_wallet_balance()
        if balance is None or balance <= 0:
            log.error(f"Cannot calculate order quantity: Invalid or zero balance ({balance}).")
            return None

        risk_amount = balance * (Decimal(str(risk_percent)) / 100)
        sl_distance_points = abs(entry_decimal - sl_decimal)

        if sl_distance_points == 0:
             log.error("Stop loss distance calculated as zero. Cannot calculate quantity.")
             return None

        # For Linear contracts (XXX/USDT), PnL is in Quote currency (USDT).
        # Loss per contract = Qty (in Base) * SL_Distance (in Quote)
        # We want: Qty * SL_Distance <= Risk Amount
        # Qty (in Base Asset, e.g., BTC) = Risk Amount / SL_Distance
        qty_base = risk_amount / sl_distance_points

    except (InvalidOperation, ZeroDivisionError, TypeError) as e:
        log.error(f"Error during quantity calculation math: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors
        log.error(f"Unexpected error during quantity calculation: {e}", exc_info=(log_level <= logging.DEBUG))
        return None

    # Round down to the minimum quantity step using the strategy's helper
    qty_rounded = strategy_instance.round_qty(float(qty_base))

    # Check against min/max order quantity from market_info
    min_qty = float(market_info['lotSizeFilter']['minOrderQty'])
    max_qty = float(market_info['lotSizeFilter']['maxOrderQty'])

    if qty_rounded < min_qty:
        log.warning(f"Calculated quantity {qty_rounded:.{strategy_instance.qty_precision}f} is below minimum ({min_qty}).")
        # Decision: Use min_qty (higher risk) or skip trade?
        # Current behavior: Use min_qty but warn about increased risk.
        qty_final = min_qty
        # Recalculate actual risk if using min_qty
        actual_risk_amount = Decimal(str(min_qty)) * sl_distance_points
        actual_risk_percent = (actual_risk_amount / balance) * 100 if balance > 0 else Decimal(0)
        log.warning(f"Using minimum quantity {min_qty:.{strategy_instance.qty_precision}f}. "
                    f"Actual Risk: {actual_risk_amount:.2f} USDT ({actual_risk_percent:.2f}%)")

    elif qty_rounded > max_qty:
        log.warning(f"Calculated quantity {qty_rounded:.{strategy_instance.qty_precision}f} exceeds maximum ({max_qty}). Using maximum.")
        qty_final = max_qty
    else:
        qty_final = qty_rounded

    if qty_final <= 0:
        log.error(f"Final calculated quantity is zero or negative ({qty_final}). Cannot place order.")
        return None

    log.info(f"Calculated Order Qty: {qty_final:.{strategy_instance.qty_precision}f} "
             f"(Balance={balance:.2f}, Risk={risk_percent}%, RiskAmt={risk_amount:.2f}, "
             f"SLDist={sl_distance_points:.{strategy_instance.price_precision}f})")
    return qty_final

def place_order(symbol: str, side: str, qty: float, price: Optional[float]=None, sl_price: Optional[float]=None, tp_price: Optional[float]=None) -> Optional[Dict]:
    """
    Places an order (Market or Limit) via Bybit API with optional SL/TP.
    Handles rounding, quantity checks, and basic error reporting.
    Uses an order_lock to prevent concurrent order placements.
    """
    if not session or not strategy_instance or not market_info:
        log.error("Cannot place order: Session, strategy instance, or market info missing.")
        return None
    if config.get("mode", "Live").lower() == "paper":
        log.warning(f"[PAPER MODE] Simulating {side} order placement: Qty={qty}, Symbol={symbol}, Price={price}, SL={sl_price}, TP={tp_price}")
        # Simulate a successful response for paper trading state management
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": f"paper_{int(time.time())}"}, "paperTrade": True}

    # Acquire lock to ensure only one order placement happens at a time
    with order_lock:
        order_type = config['order']['type']
        min_qty = float(market_info['lotSizeFilter']['minOrderQty'])

        # Final quantity rounding and validation before placing order
        qty_rounded = strategy_instance.round_qty(qty)
        if qty_rounded <= 0:
             log.error(f"Attempted to place order with zero or negative rounded quantity ({qty_rounded}). Original qty: {qty}.")
             return None
        if qty_rounded < min_qty:
            log.warning(f"Final quantity {qty_rounded:.{strategy_instance.qty_precision}f} is less than min qty {min_qty}. Adjusting to minimum.")
            qty_rounded = min_qty

        # Determine reference price for SL/TP validation (use provided price for limit, estimate for market)
        # For market orders, the exact entry isn't known, use the last close or current mark price as estimate
        ref_entry_price = price if order_type == "Limit" and price else get_current_position(symbol).get('markPrice', None) # Fallback to mark price
        if ref_entry_price is None: # If still None, maybe use last candle close?
             with data_lock:
                  if latest_dataframe is not None and not latest_dataframe.empty:
                       ref_entry_price = latest_dataframe['close'].iloc[-1]
        if ref_entry_price is None:
             log.error("Could not determine a reference entry price for SL/TP validation. Skipping SL/TP.")
             sl_price = None
             tp_price = None
        else:
             ref_entry_price = float(ref_entry_price) # Ensure float

        # Prepare order parameters dictionary
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side, # "Buy" or "Sell"
            "orderType": order_type,
            "qty": str(qty_rounded), # API requires quantity as a string
            "timeInForce": "GTC", # GoodTillCancel is common for entries with SL/TP
            "reduceOnly": False, # This is an entry order
            "positionIdx": 0 # Required for one-way position mode
        }

        # Add price for Limit orders
        if order_type == "Limit":
            if price and isinstance(price, (float, int)) and np.isfinite(price):
                # Round limit price according to tick size (use default rounding for limit orders)
                limit_price_rounded = strategy_instance.round_price(price, rounding_mode=ROUND_UP if side == "Buy" else ROUND_DOWN) # Be slightly aggressive on limit entry
                params["price"] = str(limit_price_rounded)
            else:
                log.error(f"Limit order requires a valid price. Got: {price}. Order cancelled.")
                return None

        # Add SL/TP using Bybit's parameters, with validation
        if sl_price and isinstance(sl_price, (float, int)) and np.isfinite(sl_price):
            # Round SL price (away from entry: DOWN for Buy, UP for Sell)
            sl_rounding = ROUND_DOWN if side == "Buy" else ROUND_UP
            sl_price_rounded = strategy_instance.round_price(sl_price, rounding_mode=sl_rounding)

            # Validate SL relative to reference entry price
            if ref_entry_price is not None:
                if (side == "Buy" and sl_price_rounded >= ref_entry_price) or \
                   (side == "Sell" and sl_price_rounded <= ref_entry_price):
                    log.error(f"Invalid SL price {sl_price_rounded:.{strategy_instance.price_precision}f} for {side} order "
                              f"relative to reference entry {ref_entry_price:.{strategy_instance.price_precision}f}. SL skipped.")
                else:
                    params["stopLoss"] = str(sl_price_rounded)
                    log.info(f"Setting StopLoss at: {sl_price_rounded}")
            else: # Cannot validate if ref_entry_price is unknown
                 params["stopLoss"] = str(sl_price_rounded)
                 log.warning(f"Setting StopLoss at: {sl_price_rounded} (Could not validate against entry price).")


        if tp_price and isinstance(tp_price, (float, int)) and np.isfinite(tp_price):
            # Round TP price (towards profit: UP for Buy, DOWN for Sell)
            tp_rounding = ROUND_UP if side == "Buy" else ROUND_DOWN
            tp_price_rounded = strategy_instance.round_price(tp_price, rounding_mode=tp_rounding)

            # Validate TP relative to reference entry price
            if ref_entry_price is not None:
                if (side == "Buy" and tp_price_rounded <= ref_entry_price) or \
                   (side == "Sell" and tp_price_rounded >= ref_entry_price):
                    log.error(f"Invalid TP price {tp_price_rounded:.{strategy_instance.price_precision}f} for {side} order "
                              f"relative to reference entry {ref_entry_price:.{strategy_instance.price_precision}f}. TP skipped.")
                else:
                    params["takeProfit"] = str(tp_price_rounded)
                    log.info(f"Setting TakeProfit at: {tp_price_rounded}")
            else: # Cannot validate if ref_entry_price is unknown
                 params["takeProfit"] = str(tp_price_rounded)
                 log.warning(f"Setting TakeProfit at: {tp_price_rounded} (Could not validate against entry price).")


        log.warning(f"Placing {side} {order_type} order: Qty={params['qty']} {symbol} "
                    f"{'@'+str(params.get('price')) if 'price' in params else '(Market)'} "
                    f"SL={params.get('stopLoss', 'N/A')} TP={params.get('takeProfit', 'N/A')}")
        try:
            response = session.place_order(**params)
            log.debug(f"Place Order Response: {response}")

            if response.get('retCode') == 0:
                order_id = response['result'].get('orderId')
                log.info(f"{Fore.GREEN}Order placed successfully! OrderID: {order_id}{Style.RESET_ALL}")
                # Optional: Store order ID for tracking, wait for fill via WS?
                return response
            else:
                # Log specific Bybit error messages
                error_msg = response.get('retMsg', 'Unknown Error')
                error_code = response.get('retCode', 'N/A')
                log.error(f"{Fore.RED}Failed to place order: {error_msg} (Code: {error_code}){Style.RESET_ALL}")
                # Provide hints for common errors
                if error_code == 110007: log.error("Hint: Check available margin, leverage, and potential open orders.")
                if "position mode not modified" in error_msg: log.error("Hint: Ensure Bybit account is set to One-Way position mode for linear perpetuals.")
                if "risk limit" in error_msg.lower(): log.error("Hint: Position size might exceed Bybit's risk limits for the current tier. Check Bybit settings.")
                return response # Return error response for potential handling upstream
        except Exception as e:
            log.error(f"Exception occurred during order placement: {e}", exc_info=(log_level <= logging.DEBUG))
            return None

def close_position(symbol: str, position_data: Dict[str, Any]) -> Optional[Dict]:
    """
    Closes an existing position using a reduce-only market order.
    Uses an order_lock to prevent concurrent closing attempts.
    """
    if not session or not strategy_instance:
        log.error("Cannot close position: Session or strategy instance missing.")
        return None
    if config.get("mode", "Live").lower() == "paper":
        log.warning(f"[PAPER MODE] Simulating closing position for {symbol} (Size: {position_data.get('size', 'N/A')}, Side: {position_data.get('side', 'N/A')})")
        # Simulate success
        return {"retCode": 0, "retMsg": "OK", "result": {"orderId": f"paper_close_{int(time.time())}"}, "paperTrade": True}

    # Acquire lock to ensure only one closing order happens at a time
    with order_lock:
        current_size = position_data.get('size', Decimal(0))
        current_side = position_data.get('side', 'None') # 'Buy' or 'Sell'

        if current_side == 'None' or current_size <= 0:
             log.warning(f"Attempting to close position for {symbol}, but position data indicates it's already flat or size is zero. No action taken.")
             return {"retCode": 0, "retMsg": "Position already flat or zero size", "result": {}, "alreadyFlat": True}

        # Determine the side needed to close the position
        side_to_close = "Sell" if current_side == "Buy" else "Buy"
        # Quantity to close is the current position size, converted to float for rounding, then string for API
        qty_to_close_float = float(current_size)
        qty_to_close_rounded = strategy_instance.round_qty(qty_to_close_float) # Round down just in case

        if qty_to_close_rounded <= 0:
             log.error(f"Calculated quantity to close for {symbol} is zero or negative ({qty_to_close_rounded}). Cannot place closing order.")
             return None

        log.warning(f"Attempting to close {current_side} position for {symbol} (Size: {current_size}). Placing {side_to_close} Market order (Reduce-Only)...")
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side_to_close,
            "orderType": "Market", # Use Market order for immediate closure
            "qty": str(qty_to_close_rounded), # API requires string quantity
            "reduceOnly": True, # CRITICAL: Ensures this order only reduces/closes the position
            "positionIdx": 0 # Required for one-way mode
        }
        try:
            # Optional: Cancel existing SL/TP orders before closing market order
            # This might prevent conflicts or unexpected partial fills of SL/TP
            log.info(f"Attempting to cancel existing Stop Orders (SL/TP) for {symbol} before closing...")
            response_cancel = session.cancel_all_orders(category="linear", symbol=symbol, orderFilter="StopOrder")
            log.debug(f"Cancel SL/TP Response: {response_cancel}")
            if response_cancel.get('retCode') != 0:
                 # Log warning but proceed with close attempt anyway
                 log.warning(f"Could not cancel stop orders before closing: {response_cancel.get('retMsg', 'Error')}. Proceeding with close.")
            time.sleep(0.5) # Brief pause after cancellation attempt

            # Place the closing market order
            response = session.place_order(**params)
            log.debug(f"Close Position Order Response: {response}")

            if response.get('retCode') == 0:
                order_id = response['result'].get('orderId')
                log.info(f"{Fore.YELLOW}Position close order placed successfully! OrderID: {order_id}{Style.RESET_ALL}")
                return response
            else:
                error_msg = response.get('retMsg', 'Unknown Error')
                error_code = response.get('retCode', 'N/A')
                log.error(f"{Fore.RED}Failed to place close order: {error_msg} (Code: {error_code}){Style.RESET_ALL}")
                # Handle common reduce-only errors (often mean position changed or closed already)
                if error_code in [110043, 3400070, 110025]: # Various reduce-only / position size errors
                    log.warning("Reduce-only error likely means position size changed or closed between check and execution. Re-checking position soon.")
                    global last_position_check_time # Force re-check sooner
                    last_position_check_time = 0
                return response # Return error response
        except Exception as e:
            log.error(f"Exception occurred during position closing: {e}", exc_info=(log_level <= logging.DEBUG))
            return None

def set_leverage(symbol: str, leverage: int):
    """Sets the leverage for the specified symbol (requires one-way mode)."""
    if not session:
        log.error("HTTP session not available for set_leverage.")
        return
    # Validate leverage value (adjust range based on Bybit's limits if necessary)
    if not 1 <= leverage <= 100:
         log.error(f"Invalid leverage value: {leverage}. Must be between 1 and 100 (check Bybit limits).")
         return

    log.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
    try:
        # Bybit V5 requires setting buy and sell leverage equally for one-way mode
        response = session.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=str(float(leverage)), # API expects string representation
            sellLeverage=str(float(leverage))
        )
        log.debug(f"Set Leverage Response: {response}")
        if response.get('retCode') == 0:
            log.info(f"Leverage for {symbol} set to {leverage}x successfully.")
        else:
            # Common error: 110044 means leverage not modified (it was already set to this value)
            if response.get('retCode') == 110044:
                 log.warning(f"Leverage for {symbol} already set to {leverage}x (Code: 110044 - Not modified).")
            # Error 110045 might indicate trying to change leverage with open position/orders
            elif response.get('retCode') == 110045:
                 log.error(f"Failed to set leverage for {symbol}: Cannot modify leverage with open positions or orders. (Code: 110045)")
            else:
                 log.error(f"Failed to set leverage for {symbol}: {response.get('retMsg', 'Unknown Error')} "
                           f"(Code: {response.get('retCode', 'N/A')})")
    except Exception as e:
        log.error(f"Exception setting leverage: {e}", exc_info=(log_level <= logging.DEBUG))

# --- WebSocket Handling ---
def handle_ws_message(msg: Dict):
    """Callback function to process incoming WebSocket messages."""
    # log.debug(f"WS Recv: {msg}") # Very verbose, enable only for deep debugging
    global latest_dataframe
    if stop_event.is_set(): return # Don't process if shutdown is initiated

    topic = msg.get("topic", "")
    data = msg.get("data", [])

    # --- Handle Kline Updates ---
    # Topic format: kline.{interval}.{symbol}
    if topic.startswith(f"kline.{config['interval']}.{config['symbol']}"):
        if not data: return # Ignore empty data pushes
        # Bybit V5 Kline WS pushes one candle object per message in the 'data' list
        kline_item = data[0]
        # Process only confirmed (closed) candles
        if not kline_item.get('confirm', False):
            # log.debug("Ignoring unconfirmed kline update.")
            return

        try:
            # Extract data for the confirmed candle
            ts_ms = int(kline_item['start'])
            ts = pd.to_datetime(ts_ms, unit='ms')

            # --- Acquire Lock to Update DataFrame ---
            with data_lock:
                if latest_dataframe is None:
                    log.warning("DataFrame not initialized yet, skipping WS kline processing.")
                    return

                # Check if this candle timestamp already exists (can happen on reconnect)
                if ts in latest_dataframe.index:
                    # log.debug(f"Ignoring duplicate confirmed candle via WS: {ts}")
                    return

                log.debug(f"Confirmed Kline received via WS: T={ts}, O={kline_item['open']}, H={kline_item['high']}, L={kline_item['low']}, C={kline_item['close']}, V={kline_item['volume']}")

                new_data = {
                    'open': float(kline_item['open']),
                    'high': float(kline_item['high']),
                    'low': float(kline_item['low']),
                    'close': float(kline_item['close']),
                    'volume': float(kline_item['volume']),
                    'turnover': float(kline_item.get('turnover', 0.0)) # Turnover might not always be present
                }
                # Create a new DataFrame row with the timestamp as index
                new_row = pd.DataFrame([new_data], index=[ts])

                # Append the new row
                latest_dataframe = pd.concat([latest_dataframe, new_row])

                # Prune the DataFrame to maintain max length
                max_len = config['data']['max_df_len']
                if len(latest_dataframe) > max_len:
                    latest_dataframe = latest_dataframe.iloc[-max_len:]
                    # log.debug(f"DataFrame pruned to {len(latest_dataframe)} rows.")

                # --- Trigger Analysis on the Updated DataFrame ---
                if strategy_instance:
                    log.info(f"Running analysis on new confirmed candle: {ts}")
                    # Pass a copy to the strategy to prevent modification issues if analysis takes time
                    df_copy = latest_dataframe.copy()
                else:
                    df_copy = None # No strategy instance yet

            # --- Process Signals (outside data_lock) ---
            if df_copy is not None and strategy_instance:
                try:
                    analysis_results = strategy_instance.update(df_copy)
                    # Process the generated signals to potentially execute trades
                    process_signals(analysis_results)
                except Exception as e:
                    # Log the full traceback for strategy errors
                    log.error(f"Error during strategy analysis triggered by WS update: {e}", exc_info=True)

        except (KeyError, ValueError, TypeError) as e:
            log.error(f"Error parsing kline data from WS message: {e} - Data: {kline_item}")
        except Exception as e:
             log.error(f"Unexpected error handling kline WS message: {e}", exc_info=True)

    # --- Handle Position Updates ---
    # Topic format: position.{symbol}
    elif topic.startswith("position"):
         if data:
             for pos_update in data:
                 # Filter for the symbol we are trading
                 if pos_update.get('symbol') == config['symbol']:
                      log.info(f"{Fore.CYAN}Position update via WS: Size={pos_update.get('size')}, Side={pos_update.get('side')}, "
                               f"AvgPrice={pos_update.get('avgPrice')}, PnL={pos_update.get('unrealisedPnl')}{Style.RESET_ALL}")
                      # OPTIONAL: Directly update internal state based on WS.
                      # Requires careful locking (position_lock) and state management.
                      # Safer approach: Trigger a faster HTTP position check if needed.
                      # global last_position_check_time
                      # last_position_check_time = 0 # Force check on next main loop iteration
                      pass # Currently just logging the update

    # --- Handle Order Updates ---
    # Topic format: order (catches all order updates for the account)
    elif topic.startswith("order"):
        if data:
             for order_update in data:
                 # Filter for the symbol we are trading
                 if order_update.get('symbol') == config['symbol']:
                     order_status = order_update.get('orderStatus')
                     order_id = order_update.get('orderId')
                     log.info(f"{Fore.CYAN}Order update via WS: ID={order_id}, Status={order_status}, "
                              f"Type={order_update.get('orderType')}, Side={order_update.get('side')}, "
                              f"Price={order_update.get('price')}, Qty={order_update.get('qty')}, "
                              f"AvgPrice={order_update.get('avgPrice')}{Style.RESET_ALL}")
                     # Can use this to track fills, cancellations, SL/TP triggers etc.
                     # Example: If orderStatus is 'Filled' and it matches an expected entry/exit order ID.
                     pass # Currently just logging the update

    # --- Handle Connection Status / Authentication ---
    elif msg.get("op") == "auth":
        if msg.get("success"):
            log.info(f"{Fore.GREEN}WebSocket authenticated successfully.{Style.RESET_ALL}")
        else:
            log.error(f"WebSocket authentication failed: {msg.get('ret_msg', 'No reason provided')}")
            # Consider stopping the bot or attempting reconnect if auth fails persistently
    elif msg.get("op") == "subscribe":
         if msg.get("success"):
            subscribed_topics = msg.get('ret_msg') or msg.get('args') # Location varies slightly
            log.info(f"{Fore.GREEN}WebSocket subscribed successfully to: {subscribed_topics}{Style.RESET_ALL}")
            ws_connected.set() # Signal that connection and subscription are likely successful
         else:
            log.error(f"WebSocket subscription failed: {msg.get('ret_msg', 'No reason provided')}")
            ws_connected.clear() # Signal potential connection issue
            # Consider stopping or retrying
    elif msg.get("op") == "pong":
        log.debug("WebSocket Pong received (heartbeat OK)")
    elif "success" in msg and not msg.get("success"):
        # Catch other potential operation failures
        log.error(f"WebSocket operation failed: {msg}")


def run_websocket_loop():
    """Target function for the WebSocket thread."""
    global ws
    if not ws:
        log.error("WebSocket object not initialized before starting thread.")
        return
    log.info("WebSocket thread started. Running forever...")
    while not stop_event.is_set():
        try:
            # run_forever blocks until exit() is called or an error occurs
            ws.run_forever(ping_interval=config['websocket']['ping_interval'])
        except Exception as e:
            log.error(f"WebSocket run_forever error: {e}", exc_info=True)
            if stop_event.is_set():
                break # Exit loop if stopping
            log.info("Attempting to reconnect WebSocket after error in 10 seconds...")
            time.sleep(10)
            # Need to re-initialize and re-subscribe if run_forever fails
            # This simple loop might not be enough, a full restart might be needed
            if not stop_event.is_set():
                 log.warning("Simple reconnect attempt. Consider full WS restart logic.")
                 # Attempt to re-establish connection (may need re-init of ws object)
                 try:
                      if ws: ws.exit() # Ensure old one is closed
                 except: pass
                 # Re-initialize and start (simplified, might need full start_websocket logic)
                 ws = WebSocket(testnet=TESTNET, channel_type="private", api_key=API_KEY, api_secret=API_SECRET)
                 ws.websocket_data.add_handler(handle_ws_message)
                 topics = [
                    f"kline.{config['interval']}.{config['symbol']}",
                    f"position.{config['symbol']}",
                    "order"
                 ]
                 ws.subscribe(topics)
                 # Loop will retry run_forever
        # Add a small sleep if run_forever exits cleanly but stop_event is not set (shouldn't happen often)
        if not stop_event.is_set():
             time.sleep(1)
    log.info("WebSocket thread finished.")


def start_websocket_connection() -> bool:
    """Initializes and starts the WebSocket connection in a separate thread."""
    global ws, ws_thread, ws_connected
    if not API_KEY or not API_SECRET:
        log.error("Cannot start WebSocket: API credentials missing.")
        return False
    if ws_thread and ws_thread.is_alive():
        log.warning("WebSocket thread is already running.")
        return True # Already running

    log.info("Initializing WebSocket connection...")
    ws_connected.clear() # Reset connection status event

    try:
        # Use "private" channel for accessing private topics like position and order
        ws = WebSocket(testnet=TESTNET, channel_type="private", api_key=API_KEY, api_secret=API_SECRET)

        # Define required subscriptions
        kline_topic = f"kline.{config['interval']}.{config['symbol']}"
        position_topic = f"position.{config['symbol']}" # Specific symbol position updates
        order_topic = "order" # All order updates for the account

        topics_to_subscribe = [kline_topic, position_topic, order_topic]

        # Register the central message handler
        ws.websocket_data.add_handler(handle_ws_message)

        # Subscribe to topics *before* starting the connection loop
        ws.subscribe(topics_to_subscribe)

        # Start the WebSocket processing loop in a daemon thread
        # Daemon=True allows the main program to exit even if this thread is running
        ws_thread = threading.Thread(target=run_websocket_loop, daemon=True, name="WebSocketThread")
        ws_thread.start()

        # Wait briefly for the connection and subscription confirmation
        connect_timeout = config['websocket'].get('connect_timeout', 10)
        log.info(f"Waiting up to {connect_timeout}s for WebSocket connection and subscription confirmation...")
        if ws_connected.wait(timeout=connect_timeout):
            log.info(f"{Fore.GREEN}WebSocket connected and subscribed successfully to topics: {topics_to_subscribe}{Style.RESET_ALL}")
            return True
        else:
            log.error(f"WebSocket did not confirm subscription within {connect_timeout}s. Check connection/credentials.")
            # Attempt cleanup even if connection failed
            stop_websocket_connection()
            return False

    except Exception as e:
        log.critical(f"CRITICAL: Failed to initialize or start WebSocket: {e}", exc_info=True)
        if ws:
            try: ws.exit()
            except: pass
        ws = None
        ws_thread = None
        ws_connected.clear()
        return False

def stop_websocket_connection():
    """Stops the WebSocket connection and joins the thread gracefully."""
    global ws, ws_thread, ws_connected
    if not ws and not (ws_thread and ws_thread.is_alive()):
        log.info("WebSocket already stopped or not initialized.")
        return

    log.info("Stopping WebSocket connection...")
    ws_connected.clear() # Signal connection is down

    if ws:
        try:
            # Optional: Unsubscribe from topics first
            # ws.unsubscribe([...])
            ws.exit() # Signal run_forever to stop
            log.info("WebSocket exit() called.")
        except Exception as e:
            log.error(f"Error calling WebSocket exit(): {e}")
    else:
         log.info("WebSocket object not found, attempting to join thread directly.")

    if ws_thread and ws_thread.is_alive():
        log.info("Waiting for WebSocket thread to join...")
        ws_thread.join(timeout=10) # Wait up to 10 seconds for the thread to finish
        if ws_thread.is_alive():
            log.warning("WebSocket thread did not stop gracefully after 10 seconds.")
        else:
            log.info("WebSocket thread joined successfully.")
    elif ws_thread:
         log.info("WebSocket thread was already stopped.")

    # Clean up global variables
    ws = None
    ws_thread = None
    log.info("WebSocket connection stopped.")

# --- Signal Processing & Trade Execution ---

def calculate_sl_tp(side: str, entry_price: float, last_atr: Optional[float], results: AnalysisResults) -> Tuple[Optional[float], Optional[float]]:
    """Calculates Stop Loss and Take Profit prices based on strategy config."""
    if not strategy_instance or not market_info: return None, None
    if not isinstance(entry_price, (float, int)) or not np.isfinite(entry_price): return None, None

    sl_price_raw = None
    tp_price_raw = None
    sl_method = strategy_instance.sl_method
    sl_atr_multiplier = strategy_instance.sl_atr_multiplier
    tp_ratio = Decimal(str(config['order'].get('tp_ratio', 2.0)))
    tick_size = strategy_instance.tick_size

    # --- Calculate Stop Loss ---
    if sl_method == "ATR":
        if last_atr and last_atr > 0:
            sl_distance = last_atr * sl_atr_multiplier
            sl_price_raw = entry_price - sl_distance if side == "Buy" else entry_price + sl_distance
        else:
            log.error(f"Cannot calculate ATR Stop Loss: Invalid last_atr value ({last_atr}).")
            return None, None # Cannot proceed without valid SL

    elif sl_method == "OB":
        sl_buffer_atr_fraction = Decimal("0.1") # Buffer as fraction of ATR
        sl_buffer_price_fraction = Decimal("0.0005") # Buffer as fraction of price (fallback)
        buffer = Decimal(str(last_atr * float(sl_buffer_atr_fraction))) if last_atr else Decimal(str(entry_price)) * sl_buffer_price_fraction
        buffer = max(buffer, tick_size) # Ensure buffer is at least one tick

        if side == "Buy":
            # Find lowest bottom of active bull OBs below entry
            relevant_obs = [b for b in results['active_bull_boxes'] if b['bottom'] < entry_price]
            if relevant_obs:
                lowest_bottom = min(Decimal(str(b['bottom'])) for b in relevant_obs)
                sl_price_raw = float(lowest_bottom - buffer)
            else:
                log.warning("OB SL method chosen for BUY, but no active Bull OB found below entry. Falling back to ATR.")
                if last_atr and last_atr > 0:
                    sl_price_raw = entry_price - (last_atr * sl_atr_multiplier)
                else: log.error("Cannot set SL: Fallback ATR is unavailable."); return None, None
        else: # side == "Sell"
            # Find highest top of active bear OBs above entry
            relevant_obs = [b for b in results['active_bear_boxes'] if b['top'] > entry_price]
            if relevant_obs:
                highest_top = max(Decimal(str(b['top'])) for b in relevant_obs)
                sl_price_raw = float(highest_top + buffer)
            else:
                log.warning("OB SL method chosen for SELL, but no active Bear OB found above entry. Falling back to ATR.")
                if last_atr and last_atr > 0:
                    sl_price_raw = entry_price + (last_atr * sl_atr_multiplier)
                else: log.error("Cannot set SL: Fallback ATR is unavailable."); return None, None

    if sl_price_raw is None:
        log.error("Stop Loss price could not be calculated. Cannot determine trade parameters.")
        return None, None

    # --- Validate and Round SL ---
    # Ensure SL is on the correct side of the entry price
    if (side == "Buy" and sl_price_raw >= entry_price) or \
       (side == "Sell" and sl_price_raw <= entry_price):
        log.error(f"Calculated SL price {sl_price_raw} is not logical for a {side} trade from entry {entry_price}. Using fallback ATR SL.")
        if last_atr and last_atr > 0:
            sl_distance = last_atr * sl_atr_multiplier
            sl_price_raw = entry_price - sl_distance if side == "Buy" else entry_price + sl_distance
            # Re-check after fallback
            if (side == "Buy" and sl_price_raw >= entry_price) or \
               (side == "Sell" and sl_price_raw <= entry_price):
                 log.error("Fallback ATR SL is also invalid. Cannot determine SL.")
                 return None, None
        else:
            log.error("Cannot set SL: Fallback ATR is unavailable.")
            return None, None

    # Round SL price (away from entry: DOWN for Buy, UP for Sell)
    sl_rounding = ROUND_DOWN if side == "Buy" else ROUND_UP
    sl_price = strategy_instance.round_price(sl_price_raw, rounding_mode=sl_rounding)
    if pd.isna(sl_price):
         log.error("Failed to round SL price.")
         return None, None
    log.info(f"Calculated SL price for {side}: {sl_price}")

    # --- Calculate Take Profit ---
    sl_distance_decimal = abs(Decimal(str(entry_price)) - Decimal(str(sl_price)))
    if sl_distance_decimal > 0 and tp_ratio > 0:
        tp_distance = sl_distance_decimal * tp_ratio
        tp_price_decimal = Decimal(str(entry_price)) + tp_distance if side == "Buy" else Decimal(str(entry_price)) - tp_distance
        tp_price_raw = float(tp_price_decimal)

        # Round TP price (towards profit: UP for Buy, DOWN for Sell)
        tp_rounding = ROUND_UP if side == "Buy" else ROUND_DOWN
        tp_price = strategy_instance.round_price(tp_price_raw, rounding_mode=tp_rounding)
        if pd.isna(tp_price):
             log.warning("Failed to round TP price. TP will not be set.")
             tp_price = None
        else:
             log.info(f"Calculated TP price for {side}: {tp_price} (Ratio: {tp_ratio})")
    else:
        log.warning("Cannot calculate TP: SL distance is zero or TP ratio is not positive.")
        tp_price = None

    return sl_price, tp_price


def process_signals(results: AnalysisResults):
    """
    Processes the strategy signals, calculates order parameters,
    checks current position state, and executes trades or closes positions.
    """
    if not results or not strategy_instance or not market_info:
        log.warning("Signal processing skipped: Missing results, strategy instance, or market info.")
        return
    if stop_event.is_set():
        log.warning("Signal processing skipped: Stop event is set.")
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
    # Crucial for deciding whether to enter or exit. Rate limiting is handled internally.
    position_data = get_current_position(symbol)
    if position_data is None:
        log.warning("Position check skipped due to rate limit. Will re-evaluate on next candle.")
        return # Wait for next cycle if check was skipped

    # If get_current_position returns the 'flat'/'error' dict, use it
    current_pos_size = position_data.get('size', Decimal(0))
    current_pos_side = position_data.get('side', 'None') # 'Buy', 'Sell', or 'None'

    is_long = current_pos_side == 'Buy' and current_pos_size > 0
    is_short = current_pos_side == 'Sell' and current_pos_size > 0 # V5 size is positive for short too
    is_flat = not is_long and not is_short

    log.info(f"Current Position: {'Long' if is_long else 'Short' if is_short else 'Flat'} (Size: {current_pos_size})")

    # --- Execute Actions Based on Signal and Position State ---

    # BUY Signal: Enter Long if Flat
    if signal == "BUY" and is_flat:
        log.warning(f"{Fore.GREEN}{Style.BRIGHT}BUY Signal Received - Attempting to Enter Long.{Style.RESET_ALL}")
        sl_price, tp_price = calculate_sl_tp("Buy", last_close, last_atr, results)
        if sl_price is None:
            log.error("Failed to calculate valid SL for BUY signal. Order cancelled.")
            return

        qty = calculate_order_qty(last_close, sl_price, config['order']['risk_per_trade_percent'])
        if qty and qty > 0:
            # Use last_close as the reference price for Limit orders if applicable
            limit_price = last_close if config['order']['type'] == "Limit" else None
            place_order(symbol, "Buy", qty, price=limit_price, sl_price=sl_price, tp_price=tp_price)
        else:
            log.error("Order quantity calculation failed or resulted in zero/negative. Cannot place BUY order.")

    # SELL Signal: Enter Short if Flat
    elif signal == "SELL" and is_flat:
        log.warning(f"{Fore.RED}{Style.BRIGHT}SELL Signal Received - Attempting to Enter Short.{Style.RESET_ALL}")
        sl_price, tp_price = calculate_sl_tp("Sell", last_close, last_atr, results)
        if sl_price is None:
            log.error("Failed to calculate valid SL for SELL signal. Order cancelled.")
            return

        qty = calculate_order_qty(last_close, sl_price, config['order']['risk_per_trade_percent'])
        if qty and qty > 0:
            limit_price = last_close if config['order']['type'] == "Limit" else None
            place_order(symbol, "Sell", qty, price=limit_price, sl_price=sl_price, tp_price=tp_price)
        else:
            log.error("Order quantity calculation failed or resulted in zero/negative. Cannot place SELL order.")

    # EXIT_LONG Signal: Close Long Position if Currently Long
    elif signal == "EXIT_LONG" and is_long:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_LONG Signal Received - Attempting to Close Long Position.{Style.RESET_ALL}")
        close_position(symbol, position_data) # Pass the fetched position data

    # EXIT_SHORT Signal: Close Short Position if Currently Short
    elif signal == "EXIT_SHORT" and is_short:
        log.warning(f"{Fore.YELLOW}{Style.BRIGHT}EXIT_SHORT Signal Received - Attempting to Close Short Position.{Style.RESET_ALL}")
        close_position(symbol, position_data)

    # HOLD Signal or Signal Matches Current State -> No Action Needed
    elif signal == "HOLD":
        log.debug("HOLD Signal - No trade action.")
    elif signal == "BUY" and is_long:
        log.debug("BUY Signal received, but already Long. No action.")
    elif signal == "SELL" and is_short:
        log.debug("SELL Signal received, but already Short. No action.")
    elif signal == "EXIT_LONG" and not is_long:
         log.debug("EXIT_LONG Signal received, but not Long. No action.")
    elif signal == "EXIT_SHORT" and not is_short:
         log.debug("EXIT_SHORT Signal received, but not Short. No action.")
    # Log cases where entry signal received while already in opposite position (should ideally be handled by EXIT first)
    elif signal == "BUY" and is_short:
         log.warning("BUY Signal received while Short. Strategy should have generated EXIT_SHORT first. No action.")
    elif signal == "SELL" and is_long:
         log.warning("SELL Signal received while Long. Strategy should have generated EXIT_LONG first. No action.")


# --- Graceful Shutdown ---
def handle_shutdown_signal(signum, frame):
    """Handles termination signals (SIGINT, SIGTERM) for graceful shutdown."""
    if stop_event.is_set(): # Prevent running multiple times if signal received repeatedly
         log.warning("Shutdown already in progress.")
         return
    log.warning(f"Shutdown signal {signal.Signals(signum).name} ({signum}) received. Initiating graceful shutdown...")
    stop_event.set() # Signal all loops and threads to stop

    # 1. Stop WebSocket first to prevent processing new data/signals
    stop_websocket_connection()

    # 2. Optional: Implement logic to manage open positions/orders on shutdown
    #    USE WITH EXTREME CAUTION - unexpected closure can lead to losses.
    close_on_exit = False # Make this configurable if needed (e.g., in config.json)
    if close_on_exit and config.get("mode", "Live").lower() != "paper":
        log.warning("Attempting to close open position on exit (close_on_exit=True)...")
        # Need to ensure we get the latest position data, might need retry if rate limited
        pos_data = None
        for attempt in range(3): # Try a few times
             pos_data = get_current_position(config['symbol'])
             if pos_data is not None: # Got data (or confirmed flat), break retry
                  break
             log.warning(f"Position check rate limited during shutdown. Retrying in {POSITION_CHECK_INTERVAL/2}s...")
             time.sleep(POSITION_CHECK_INTERVAL / 2) # Wait briefly

        if pos_data and pos_data.get('size', Decimal(0)) > 0:
             log.warning(f"Found open {pos_data.get('side')} position (Size: {pos_data.get('size')}). Attempting market close.")
             close_response = close_position(config['symbol'], pos_data)
             if close_response and close_response.get('retCode') == 0:
                  log.info("Position close order placed successfully during shutdown.")
             else:
                  log.error("Failed to place position close order during shutdown.")
        elif pos_data:
             log.info("No open position found to close on exit.")
        else:
             log.error("Could not determine position status during shutdown due to repeated errors/rate limit.")

    # 3. Final log message and exit
    log.info("Shutdown sequence complete. Exiting.")
    # Give logs a moment to flush before exiting
    logging.shutdown()
    time.sleep(0.5)
    sys.exit(0)

# --- Main Execution Block ---
if __name__ == "__main__":
    print(Fore.MAGENTA + Style.BRIGHT + "\n~~~ Pyrmethus Volumatic+OB Trading Bot Initializing ~~~" + Style.RESET_ALL)

    # Load configuration first, as it dictates logging level etc.
    config = load_config() # Exits on critical error

    # Setup logging based on config
    setup_logging(config.get("log_level", "INFO"))
    log.info(f"Logging level set to: {logging.getLevelName(log.level)}")
    log.debug(f"Full Config: {json.dumps(config, indent=2)}")

    # Register signal handlers for graceful shutdown (Ctrl+C, kill)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    log.info("Registered shutdown signal handlers.")

    # Connect to Bybit HTTP API
    session = connect_bybit()
    if not session:
        # connect_bybit already logs critical error and exits
        sys.exit(1)

    # Get Market Info (Symbol, Precision, Limits)
    market_info = get_market_info(config['symbol'])
    if not market_info:
        log.critical(f"Could not retrieve valid market info for symbol '{config['symbol']}'. Check symbol and API connection. Exiting.")
        sys.exit(1)

    # Set Leverage (Important!) - Do this before initializing strategy if strategy uses it
    # Ensure leverage is set before potentially placing orders
    set_leverage(config['symbol'], config['order']['leverage'])

    # Initialize Strategy Engine
    try:
        # Pass market info and strategy-specific parameters
        strategy_instance = VolumaticOBStrategy(market_info=market_info, **config['strategy'])
    except (ValueError, KeyError, TypeError) as e:
        log.critical(f"Failed to initialize strategy: {e}. Check config.json strategy params. Exiting.", exc_info=True)
        sys.exit(1)
    except Exception as e:
         log.critical(f"Unexpected error initializing strategy: {e}", exc_info=True)
         sys.exit(1)

    # Fetch Initial Historical Data
    with data_lock: # Protect dataframe initialization
        latest_dataframe = fetch_initial_data(
            config['symbol'],
            config['interval'],
            config['data']['fetch_limit']
        )

    if latest_dataframe is None: # Check for None specifically (indicates fetch error)
        log.critical("Failed to fetch initial historical data. Cannot proceed. Exiting.")
        sys.exit(1)
    if latest_dataframe.empty:
        log.warning("Fetched initial data, but the DataFrame is empty. "
                    "Check symbol/interval on Bybit. Bot will attempt to run using WebSocket data, "
                    "but strategy may need more history.")
        # Allow continuing, WS might populate data, but strategy needs min_data_len
    elif len(latest_dataframe) < strategy_instance.min_data_len:
         log.warning(f"Initial data fetched ({len(latest_dataframe)} candles) is less than minimum required by strategy "
                     f"({strategy_instance.min_data_len}). Strategy calculations may be inaccurate until more data arrives via WebSocket.")
         # Allow continuing

    # Run Initial Analysis on historical data (if enough data exists)
    # This pre-fills indicators and establishes the initial trend/state
    initial_analysis_done = False
    if latest_dataframe is not None and not latest_dataframe.empty and len(latest_dataframe) >= strategy_instance.min_data_len:
        log.info("Running initial analysis on historical data...")
        with data_lock: # Access dataframe safely
            # Pass a copy for analysis
            initial_results = strategy_instance.update(latest_dataframe.copy())
        if initial_results:
            log.info(f"Initial Analysis Complete: Current Trend Estimate = {'UP' if initial_results['current_trend'] else 'DOWN' if initial_results['current_trend'] is False else 'Undetermined'}, "
                     f"Last Signal State = {strategy_instance.last_signal_state}")
            initial_analysis_done = True
        else:
             log.error("Initial analysis failed.")
    else:
         log.info("Skipping initial analysis due to insufficient historical data.")

    # Start WebSocket Connection
    if not start_websocket_connection():
        log.critical("Failed to start WebSocket connection. Exiting.")
        # Ensure cleanup if WS fails to start
        stop_websocket_connection()
        sys.exit(1)

    # --- Bot Running Loop ---
    log.info(f"{Fore.CYAN}{Style.BRIGHT}Bot is now running for {config['symbol']} ({config['interval']}). Mode: {config['mode']}. Waiting for signals...{Style.RESET_ALL}")
    log.info("Press Ctrl+C to stop gracefully.")

    # Main loop primarily keeps the script alive and performs periodic health checks.
    # Most logic is driven by WebSocket updates handled in handle_ws_message.
    while not stop_event.is_set():
        try:
            # 1. Check WebSocket Health (is the thread alive?)
            if ws_thread and not ws_thread.is_alive():
                log.error("WebSocket thread appears to have died unexpectedly!")
                if not stop_event.is_set():
                    log.info("Attempting to restart WebSocket connection...")
                    # Attempt full stop/start cycle
                    stop_websocket_connection()
                    time.sleep(5) # Wait before restarting
                    if not start_websocket_connection():
                        log.critical("Failed to restart WebSocket after failure. Stopping bot.")
                        handle_shutdown_signal(signal.SIGTERM, None) # Trigger shutdown
                    else:
                         log.info("WebSocket connection restarted successfully.")

            # 2. Periodic Position Check (Fallback/Verification)
            # Although WS provides position updates, a periodic check ensures sync.
            now = time.time()
            if now - last_position_check_time > POSITION_CHECK_INTERVAL * 1.5: # Check slightly less often than limit
                log.debug("Performing periodic position check (verification)...")
                # Fetch position data, result is handled internally by get_current_position (updates time, logs)
                get_current_position(config['symbol'])

            # 3. Sleep efficiently until the next check or stop signal
            # Wait for a short duration or until the stop_event is set
            stop_event.wait(timeout=10.0) # Check every 10 seconds

        except KeyboardInterrupt: # Allow Ctrl+C to break the loop and trigger shutdown
             log.warning("KeyboardInterrupt detected in main loop.")
             if not stop_event.is_set():
                  handle_shutdown_signal(signal.SIGINT, None)
             break # Exit loop
        except Exception as e:
             # Catch unexpected errors in the main loop
             log.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
             log.warning("Attempting to continue after error...")
             # Wait a bit longer after an unexpected error before resuming checks
             time.sleep(15)

    # --- End of Script ---
    # Shutdown sequence is handled by handle_shutdown_signal
    log.info("Main loop terminated.")
    # Final confirmation message (might not be reached if sys.exit called in handler)
    print(Fore.MAGENTA + Style.BRIGHT + "~~~ Pyrmethus Volumatic+OB Trading Bot Stopped ~~~" + Style.RESET_ALL)

EOF
success "${MAIN_FILE} created."

# --- Final Instructions ---
success "All files created successfully in ${CLR_YELLOW}${PROJECT_DIR}${CLR_NC}"
printf "\n${CLR_YELLOW}--- Next Steps ---${CLR_NC}\n"
printf "1. ${CLR_YELLOW}Edit the ${CLR_BLUE}%s${CLR_YELLOW} file and add your actual Bybit API Key and Secret.${CLR_NC}\n" "$ENV_FILE"
printf "   ${CLR_RED}IMPORTANT: Start with Testnet keys (BYBIT_TESTNET=True) to avoid real losses!${CLR_NC}\n"
printf "2. Review ${CLR_YELLOW}%s${CLR_NC} and adjust symbol, interval, risk, leverage, and strategy parameters as needed.\n" "$CONFIG_FILE"
printf "3. Install required Python packages (preferably in a virtual environment):\n"
printf "   ${CLR_GREEN}cd %s${CLR_NC}\n" "$PROJECT_DIR"
printf "   ${CLR_GREEN}python -m venv venv${CLR_NC}\n"
printf "   ${CLR_GREEN}source venv/bin/activate  # On Windows use: venv\\Scripts\\activate${CLR_NC}\n"
printf "   ${CLR_GREEN}pip install -r %s${CLR_NC}\n" "$REQ_FILE"
printf "4. Run the bot:\n"
printf "   ${CLR_GREEN}python %s${CLR_NC}\n" "$MAIN_FILE"
printf "\n${CLR_BLUE}Bot setup complete! Remember to test thoroughly on Testnet before using real funds.${CLR_NC}\n"

# Optional: Make the setup script non-executable after running
# chmod -x "$SETUP_SCRIPT_NAME"

exit 0
