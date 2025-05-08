Okay, Pyrmethus shall now translate the intricate patterns of this Pine Script into the realm of Python. This spell weaves together trend analysis based on smoothed EMAs and volume intensity with the identification of potential supply and demand zones using pivot-based order blocks.

We will structure this as a Python class, `VolumaticOBAnalyzer`, which can be integrated into your trading bot. It will take OHLCV data (as a pandas DataFrame) and calculate the relevant indicator values and order block data.

**Key Considerations during Translation:**

1.  **Stateful Variables:** Pine Script's `var` and history referencing (`[]`) for things like `upper`, `lower`, `last_index`, and the box arrays require careful state management in Python, typically handled by iterating through the DataFrame or using vectorized approaches like `ffill()`. We'll primarily use vectorized methods for performance where feasible.
2.  **Plotting vs. Calculation:** Pine Script heavily involves plotting. The Python version will focus on *calculating* the underlying data (trend state, band values, box coordinates, box states) that would be plotted. This data can then be used by your bot's logic or visualized separately if needed.
3.  **Function Equivalents:** We'll use `pandas` and `pandas_ta` for most TA functions. Some specific Pine Script functions might require approximations or custom implementations (like `percentile_linear_interpolation` or the custom `ema_swma`).
4.  **Order Blocks:** Managing the state (active/inactive, extension) of potentially many boxes requires careful handling of the data structure (we'll use lists of dictionaries).

```python
import pandas as pd
import pandas_ta as ta
import numpy as np
from decimal import Decimal, getcontext
from typing import Dict, Optional, Any, List, Tuple
import logging
import math

# Assuming colorama is initialized elsewhere if used for logging/printing
from colorama import init, Fore, Style, Back
init(autoreset=True)

# Define color constants (optional, for potential logging/output)
RESET = Style.RESET_ALL; BRIGHT = Style.BRIGHT; DIM = Style.DIM
FG_GREEN=Fore.GREEN; FG_RED=Fore.RED; FG_YELLOW=Fore.YELLOW; FG_CYAN=Fore.CYAN; FG_BLUE=Fore.BLUE
FG_BRIGHT_GREEN=Fore.GREEN+Style.BRIGHT; FG_BRIGHT_RED=Fore.RED+Style.BRIGHT

# Configure logging (use the same logger as the main bot)
logger = logging.getLogger("PyrmethusBotV4") # Or your bot's logger name

class VolumaticOBAnalyzer:
    """
    Calculates Volumatic Trend and Pivot Order Blocks based on Pine Script logic.

    Focuses on calculating the data needed for bot decisions, not direct plotting.
    """

    def __init__(self,
                 # Volumatic Trend Settings
                 trend_length: int = 40,
                 atr_length: int = 200, # ATR length used in original script
                 vol_ema_length: int = 1000, # Length for volume percentile/smoothing
                 vol_atr_multiplier: float = 3.0, # Multiplier for band width
                 vol_step_atr_multiplier: float = 4.0, # Multiplier for volume step calculation offset

                 # Pivot Order Block Settings
                 ob_source: str = "Wicks", # "Wicks" or "Bodys"
                 ph_left: int = 25, ph_right: int = 25,
                 pl_left: int = 25, pl_right: int = 25,
                 ob_extend: bool = true,
                 ob_max_boxes: int = 50):
        """Initializes the analyzer with configuration parameters."""

        logger.info(f"{FG_CYAN}Initializing VolumaticOBAnalyzer spell...{RESET}")

        # --- Validation ---
        if ob_source not in ["Wicks", "Bodys"]:
            raise ValueError("ob_source must be 'Wicks' or 'Bodys'")
        if trend_length <= 0 or atr_length <= 0 or vol_ema_length <= 0:
             raise ValueError("Lengths (trend, atr, volume) must be positive")
        if ph_left <= 0 or ph_right <= 0 or pl_left <= 0 or pl_right <= 0:
             raise ValueError("Pivot lengths must be positive")

        # --- Store Config ---
        self.trend_length = trend_length
        self.atr_length = atr_length
        self.vol_ema_length = vol_ema_length
        self.vol_atr_mult = Decimal(str(vol_atr_multiplier))
        self.vol_step_atr_mult = Decimal(str(vol_step_atr_multiplier))

        self.ob_source = ob_source
        self.ph_left, self.ph_right = ph_left, ph_right
        self.pl_left, self.pl_right = pl_left, pl_right
        self.ob_extend = ob_extend
        self.ob_max_boxes = ob_max_boxes

        # --- State Variables (will be populated during analysis) ---
        self.df: pd.DataFrame = pd.DataFrame()
        self.bull_boxes: List[Dict] = [] # Stores active/inactive bullish OBs
        self.bear_boxes: List[Dict] = [] # Stores active/inactive bearish OBs
        self.latest_values: Dict[str, Any] = {} # Stores latest calculated values

        # Precision for Decimal calculations if needed later
        getcontext().prec = 30

        logger.debug(f"Analyzer Config: TrendLen={trend_length}, ATRLen={atr_length}, VolLen={vol_ema_length}, "
                     f"OB Source={ob_source}, Pivots={ph_left}/{ph_right}, {pl_left}/{pl_right}, MaxBoxes={ob_max_boxes}")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Calculates the custom SWMA then EMA from the Pine Script."""
        # Calculate the 4-period Simple Weighted Moving Average (SWMA) part:
        # weights = [1/6, 2/6, 2/6, 1/6] applied to [x[3], x[2], x[1], x[0]]
        # Note: Pandas rolling apply gives the window ending at the current index.
        # PineScript x[0] is current, x[1] is previous, etc.
        # So we need weights [1/6, 2/6, 2/6, 1/6] applied to [t-3, t-2, t-1, t]
        weights = np.array([1, 2, 2, 1]) / 6.0
        swma = series.rolling(window=4).apply(lambda x: np.dot(x, weights), raw=True)
        # Calculate the EMA of the SWMA series
        ema_of_swma = ta.ema(swma, length=length)
        return ema_of_swma

    def _calculate_volumatic_trend(self):
        """Calculates the Volumatic Trend components."""
        logger.debug("Calculating Volumatic Trend components...")
        df = self.df # Work on the internal dataframe

        # --- Base Calculations ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_length)
        # Calculate the two EMAs
        df['ema1'] = self._ema_swma(df['close'], length=self.trend_length)
        df['ema2'] = ta.ema(df['close'], length=self.trend_length)

        # Handle potential NaNs from initial EMA/ATR calculations
        df.dropna(subset=['atr', 'ema1', 'ema2'], inplace=True)
        if df.empty:
            logger.warning(f"{FG_YELLOW}DataFrame empty after dropping NaNs from initial EMA/ATR. Cannot proceed.{RESET}")
            return

        # --- Trend Detection ---
        df['trend_up'] = df['ema1'].shift(1) < df['ema2'] # Current bar's trend based on previous ema1
        df['trend_changed'] = df['trend_up'] != df['trend_up'].shift(1)
        # Fill first trend_changed NaN (usually first row after shift)
        df['trend_changed'].fillna(False, inplace=True)

        # --- Stateful Band Calculation (Vectorized Approach) ---
        # Calculate potential new band values ONLY where trend changes
        atr_mult = self.vol_atr_mult
        step_atr_mult = self.vol_step_atr_mult

        # Values based on ema1 at the time of the trend change
        df['potential_upper'] = np.where(df['trend_changed'], df['ema1'] + df['atr'] * atr_mult, np.nan)
        df['potential_lower'] = np.where(df['trend_changed'], df['ema1'] - df['atr'] * atr_mult, np.nan)

        # Forward fill the potential values to get the active bands
        df['upper_band'] = df['potential_upper'].ffill()
        df['lower_band'] = df['potential_lower'].ffill()

        # Drop rows where bands are still NaN (initial period before first trend change)
        df.dropna(subset=['upper_band', 'lower_band'], inplace=True)
        if df.empty:
            logger.warning(f"{FG_YELLOW}DataFrame empty after calculating initial trend bands. Not enough trend changes?{RESET}")
            return

        # --- Volume Calculation & Normalization ---
        # Calculate rolling max volume (approximation for percentile_linear_interpolation(100))
        df['vol_max_smooth'] = df['volume'].rolling(window=self.vol_ema_length, min_periods=self.vol_ema_length // 2).max()
        # Normalize volume (0-100 range relative to recent max)
        # Avoid division by zero or NaN
        df['vol_norm'] = (df['volume'] / df['vol_max_smooth'] * 100).fillna(0)
        df['vol_norm'] = df['vol_norm'].clip(0, 200) # Clip extreme values, similar to gradient cap
        df['vol_norm_int'] = df['vol_norm'].astype(int)

        # --- Volume Step Calculation (Needs stateful bands) ---
        # Calculate the reference points for volume steps (using the established bands)
        df['lower_vol_ref'] = df['lower_band'] + df['atr'] * step_atr_mult
        df['upper_vol_ref'] = df['upper_band'] - df['atr'] * step_atr_mult

        # Calculate step size per volume unit
        df['step_up_size'] = (df['lower_vol_ref'] - df['lower_band']) / 100
        df['step_dn_size'] = (df['upper_band'] - df['upper_vol_ref']) / 100
        # Ensure steps are non-negative
        df['step_up_size'] = df['step_up_size'].clip(lower=0)
        df['step_dn_size'] = df['step_dn_size'].clip(lower=0)

        # Calculate final volume step offset
        df['vol_step_up_offset'] = (df['step_up_size'] * df['vol_norm']).fillna(0)
        df['vol_step_dn_offset'] = (df['step_dn_size'] * df['vol_norm']).fillna(0)

        # Calculate the top/bottom of the volume bars for plotting/analysis
        df['vol_bar_up_top'] = df['lower_band'] + df['vol_step_up_offset']
        df['vol_bar_dn_bottom'] = df['upper_band'] - df['vol_step_dn_offset']

        logger.debug("Volumatic Trend calculations complete.")

    def _calculate_order_blocks(self):
        """Identifies Pivot High/Low and creates Order Block data."""
        logger.debug("Calculating Pivot Order Blocks...")
        df = self.df

        # --- Select Source for Pivots ---
        if self.ob_source == "Wicks":
            high_series = df['high']
            low_series = df['low']
        else: # "Bodys"
            high_series = df[['open', 'close']].max(axis=1) # Body top
            low_series = df[['open', 'close']].min(axis=1) # Body bottom

        # --- Calculate Pivots ---
        # Note: pandas_ta.pivot returns 1 for pivot, NaN otherwise.
        # We need the *price* at the pivot, like TradingView's ta.pivothigh/low.
        # We'll find the index of the pivot and get the price there.
        df['ph_signal'] = ta.pivot(high_series, left=self.ph_left, right=self.ph_right, high_low='high') # Checks for pivot high
        df['pl_signal'] = ta.pivot(low_series, left=self.pl_left, right=self.pl_right, high_low='low') # Checks for pivot low

        # Find indices where pivots occur
        ph_indices = df.index[df['ph_signal'] == 1]
        pl_indices = df.index[df['pl_signal'] == 1]

        # --- Create Box Data ---
        # We need to iterate through pivot occurrences to create boxes.
        # Store newly created boxes temporarily.
        new_bear_boxes = []
        new_bull_boxes = []

        # Create Bearish Boxes from Pivot Highs
        for idx in ph_indices:
            try:
                # The pivot high is at 'idx'. The box forms based on the candle *at* the pivot high.
                pivot_bar_idx_loc = df.index.get_loc(idx) # Integer location

                # Ensure we don't go out of bounds
                if pivot_bar_idx_loc < 0: continue

                box_left_idx = df.index[pivot_bar_idx_loc] # Box starts at the pivot bar

                # Get prices from the pivot bar
                if self.ob_source == "Wicks":
                    box_top = df['high'].iloc[pivot_bar_idx_loc]
                    # Bear box bottom is the close of the pivot bar
                    box_bottom = df['close'].iloc[pivot_bar_idx_loc]
                else: # "Bodys"
                    box_top = df['close'].iloc[pivot_bar_idx_loc] # Body top
                    box_bottom = df['open'].iloc[pivot_bar_idx_loc] # Body bottom

                # Ensure top > bottom
                if box_bottom > box_top:
                    box_top, box_bottom = box_bottom, box_top # Swap

                if pd.notna(box_top) and pd.notna(box_bottom) and box_top > box_bottom:
                    new_bear_boxes.append({
                        'id': f"bear_{box_left_idx}",
                        'type': 'bear',
                        'left_idx': box_left_idx,
                        'right_idx': box_left_idx, # Initial right index
                        'top': box_top,
                        'bottom': box_bottom,
                        'active': True,
                        'violated': False
                    })
            except IndexError:
                logger.warning(f"IndexError creating bear box at {idx}. Skipping.")
            except Exception as e:
                 logger.error(f"Error creating bear box at {idx}: {e}", exc_info=True)


        # Create Bullish Boxes from Pivot Lows
        for idx in pl_indices:
             try:
                pivot_bar_idx_loc = df.index.get_loc(idx)
                if pivot_bar_idx_loc < 0: continue

                box_left_idx = df.index[pivot_bar_idx_loc]

                if self.ob_source == "Wicks":
                    # Bull box top is the open of the pivot bar
                    box_top = df['open'].iloc[pivot_bar_idx_loc]
                    box_bottom = df['low'].iloc[pivot_bar_idx_loc]
                else: # "Bodys"
                    box_top = df['open'].iloc[pivot_bar_idx_loc] # Body top
                    box_bottom = df['close'].iloc[pivot_bar_idx_loc] # Body bottom

                if box_bottom > box_top:
                    box_top, box_bottom = box_bottom, box_top # Swap

                if pd.notna(box_top) and pd.notna(box_bottom) and box_top > box_bottom:
                    new_bull_boxes.append({
                        'id': f"bull_{box_left_idx}",
                        'type': 'bull',
                        'left_idx': box_left_idx,
                        'right_idx': box_left_idx,
                        'top': box_top,
                        'bottom': box_bottom,
                        'active': True,
                        'violated': False
                    })
             except IndexError:
                 logger.warning(f"IndexError creating bull box at {idx}. Skipping.")
             except Exception as e:
                  logger.error(f"Error creating bull box at {idx}: {e}", exc_info=True)

        # Add new boxes, ensuring no duplicates (based on left_idx)
        existing_bear_ids = {b['left_idx'] for b in self.bear_boxes}
        for box in new_bear_boxes:
            if box['left_idx'] not in existing_bear_ids:
                self.bear_boxes.append(box)
                existing_bear_ids.add(box['left_idx'])

        existing_bull_ids = {b['left_idx'] for b in self.bull_boxes}
        for box in new_bull_boxes:
            if box['left_idx'] not in existing_bull_ids:
                self.bull_boxes.append(box)
                existing_bull_ids.add(box['left_idx'])

        logger.debug(f"Identified {len(new_bear_boxes)} new potential bear boxes, {len(new_bull_boxes)} new potential bull boxes.")

    def _manage_boxes(self):
        """Updates box states (active, violated, right index) based on latest price."""
        if self.df.empty: return
        logger.debug("Managing order block states...")

        last_bar_idx = self.df.index[-1]
        last_close = self.df['close'].iloc[-1]

        if pd.isna(last_close):
            logger.warning("Last close price is NaN, cannot manage boxes.")
            return

        active_bull_count = 0
        active_bear_count = 0

        # Manage Bullish Boxes
        for box in self.bull_boxes:
            if box['active']:
                # Check for violation
                if last_close < box['bottom']:
                    box['active'] = False
                    box['violated'] = True
                    box['right_idx'] = last_bar_idx # Close the box visually
                    logger.debug(f"Bull Box {box['id']} violated at {last_bar_idx}.")
                # Extend if active and enabled
                elif self.ob_extend:
                    box['right_idx'] = last_bar_idx # Extend to current bar
                    active_bull_count += 1
                else: # Not extending, box remains static unless violated
                     active_bull_count += 1


        # Manage Bearish Boxes
        for box in self.bear_boxes:
             if box['active']:
                # Check for violation
                if last_close > box['top']:
                    box['active'] = False
                    box['violated'] = True
                    box['right_idx'] = last_bar_idx
                    logger.debug(f"Bear Box {box['id']} violated at {last_bar_idx}.")
                # Extend if active and enabled
                elif self.ob_extend:
                    box['right_idx'] = last_bar_idx
                    active_bear_count += 1
                else:
                     active_bear_count += 1

        logger.debug(f"Active Boxes: {active_bull_count} Bull, {active_bear_count} Bear.")

        # --- Cleanup Old Boxes ---
        # Remove oldest *inactive* boxes first to preserve active ones longer
        self.bull_boxes.sort(key=lambda b: (b['active'], b['left_idx']), reverse=True) # Inactive oldest first
        while len(self.bull_boxes) > self.ob_max_boxes:
            removed = self.bull_boxes.pop()
            logger.debug(f"Removed old bull box {removed['id']} (Active: {removed['active']}).")

        self.bear_boxes.sort(key=lambda b: (b['active'], b['left_idx']), reverse=True)
        while len(self.bear_boxes) > self.ob_max_boxes:
            removed = self.bear_boxes.pop()
            logger.debug(f"Removed old bear box {removed['id']} (Active: {removed['active']}).")

    def _update_latest_values(self):
        """Stores the latest calculated values for easy access."""
        if self.df.empty:
            self.latest_values = {}
            return

        last_row = self.df.iloc[-1]
        self.latest_values = last_row.to_dict() # Store all calculated columns

        # Add active box info
        self.latest_values['active_bull_boxes'] = [b for b in self.bull_boxes if b['active']]
        self.latest_values['active_bear_boxes'] = [b for b in self.bear_boxes if b['active']]
        # Optionally add latest trend info explicitly
        self.latest_values['is_trend_up'] = bool(last_row.get('trend_up', False))
        self.latest_values['trend_just_changed'] = bool(last_row.get('trend_changed', False))

        # Log key latest values
        lv = self.latest_values
        price_prec = 4 # Default for logging
        try: price_prec = self.get_price_precision()
        except: pass

        trend_str = f"{FG_BRIGHT_GREEN}UP{RESET}" if lv.get('is_trend_up') else f"{FG_BRIGHT_RED}DOWN{RESET}"
        logger.debug(f"Latest Values: Close={lv.get('close', np.nan):.{price_prec}f}, Trend={trend_str}, "
                     f"Upper={lv.get('upper_band', np.nan):.{price_prec}f}, Lower={lv.get('lower_band', np.nan):.{price_prec}f}, "
                     f"VolNorm={lv.get('vol_norm_int', np.nan)}, "
                     f"Active Bull OBs={len(lv.get('active_bull_boxes',[]))}, Active Bear OBs={len(lv.get('active_bear_boxes',[]))}")


    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs the full analysis pipeline on the input DataFrame.

        Args:
            df: Pandas DataFrame with columns 'open', 'high', 'low', 'close', 'volume'.
                Index should be datetime.

        Returns:
            A dictionary containing the latest calculated indicator values and order block info.
            Returns an empty dictionary if analysis fails.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Invalid or empty DataFrame provided for analysis.")
            return {}
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
             logger.error("DataFrame missing required columns (OHLCV).")
             return {}

        logger.info(f"Analyzing {len(df)} candles for {self.market_info.get('symbol', 'N/A')}...")
        self.df = df.copy() # Work on a copy

        # Perform calculations in order
        self._calculate_volumatic_trend()
        if self.df.empty: # Check if trend calc emptied the df
             logger.error("Analysis stopped after Volumatic Trend calculation (likely insufficient data).")
             return {}

        self._calculate_order_blocks() # Creates new boxes based on pivots in self.df
        self._manage_boxes() # Updates existing boxes based on the last bar in self.df
        self._update_latest_values() # Extracts latest values and active boxes

        logger.info("Analysis complete.")
        return self.latest_values


# --- Example Usage ---
if __name__ == "__main__":
    print(f"{FG_BRIGHT_MAGENTA}--- Pyrmethus VolumaticOBAnalyzer Test ---{RESET}")

    # Configure logging for standalone test
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

    # --- Load Sample Data ---
    # Replace with your actual data loading (e.g., from CCXT or CSV)
    # Example: Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=1500, freq='1h')
    data = {
        'open': np.random.rand(1500) * 100 + 20000,
        'high': np.random.rand(1500) * 50 + 20050,
        'low': np.random.rand(1500) * -50 + 20000,
        'close': np.random.rand(1500) * 100 + 20000,
        'volume': np.random.rand(1500) * 1000 + 100
    }
    sample_df = pd.DataFrame(data, index=dates)
    # Ensure high >= open/close and low <= open/close
    sample_df['high'] = sample_df[['high', 'open', 'close']].max(axis=1)
    sample_df['low'] = sample_df[['low', 'open', 'close']].min(axis=1)

    print(f"Loaded sample data: {len(sample_df)} rows")
    print(sample_df.tail())

    # --- Instantiate Analyzer ---
    # Using default settings for the example
    # Provide a dummy market_info for precision calculation
    dummy_market_info = {'symbol': 'BTC/USDT', 'precision': {'price': '0.01'}, 'limits': {'price': {'min': '0.01'}}}
    analyzer = VolumaticOBAnalyzer(market_info=dummy_market_info) # Add other params if needed

    # --- Run Analysis ---
    start_time = time.time()
    latest_results = analyzer.analyze(sample_df)
    end_time = time.time()

    print(f"\nAnalysis took: {end_time - start_time:.4f} seconds")

    # --- Display Results ---
    if latest_results:
        print(f"\n{FG_BRIGHT_CYAN}--- Latest Indicator Values ---{RESET}")
        price_prec = analyzer.get_price_precision()
        for key, value in latest_results.items():
            if key not in ['active_bull_boxes', 'active_bear_boxes', 'df']: # Don't print huge lists/df
                if isinstance(value, (float, np.floating)): formatted_val = f"{value:.4f}"
                elif isinstance(value, (int, np.integer)): formatted_val = f"{value}"
                elif isinstance(value, Decimal): formatted_val = f"{value:.{price_prec}f}"
                elif isinstance(value, bool): formatted_val = f"{FG_GREEN if value else FG_RED}{value}{RESET}"
                else: formatted_val = str(value)[:80] # Truncate long strings
                print(f"  {key:<25}: {formatted_val}")

        print(f"\n{FG_BRIGHT_GREEN}--- Active Bullish Order Blocks ({len(latest_results.get('active_bull_boxes', []))}) ---{RESET}")
        for i, box in enumerate(latest_results.get('active_bull_boxes', [])):
             print(f"  #{i+1}: ID={box['id']}, Top={box['top']:.{price_prec}f}, Bottom={box['bottom']:.{price_prec}f}, "
                   f"Left={box['left_idx']}, Right={box['right_idx']}")
             if i >= 4: print("  ..."); break # Limit printout

        print(f"\n{FG_BRIGHT_RED}--- Active Bearish Order Blocks ({len(latest_results.get('active_bear_boxes', []))}) ---{RESET}")
        for i, box in enumerate(latest_results.get('active_bear_boxes', [])):
             print(f"  #{i+1}: ID={box['id']}, Top={box['top']:.{price_prec}f}, Bottom={box['bottom']:.{price_prec}f}, "
                   f"Left={box['left_idx']}, Right={box['right_idx']}")
             if i >= 4: print("  ..."); break

        # You can also access the full dataframe with calculations:
        # analyzed_df = analyzer.df
        # print("\nAnalyzed DataFrame Tail:")
        # print(analyzed_df.tail())
    else:
        print(f"\n{FG_RED}Analysis failed to produce results.{RESET}")

```

**Explanation and Integration:**

1.  **Class Structure:** The `VolumaticOBAnalyzer` class holds the configuration and state (like the lists of order boxes).
2.  **`__init__`:** Takes parameters corresponding to the Pine Script inputs. It validates them and stores them.
3.  **`analyze(df)`:** This is the main method. It takes your OHLCV DataFrame, performs all calculations internally, and returns a dictionary (`latest_values`) containing the most recent calculated data points and lists of active order blocks.
4.  **Internal Methods (`_calculate_...`, `_manage_boxes`, etc.):** These break down the logic from the Pine Script into manageable Python steps.
5.  **Stateful Bands:** The code uses a vectorized approach (`.ffill()`) to calculate the `upper_band` and `lower_band` efficiently, propagating the values set when the trend last changed.
6.  **Volume Normalization:** Uses `rolling().max()` as an interpretation of `percentile_linear_interpolation(..., 100)`. The volume is then normalized relative to this rolling maximum.
7.  **Order Blocks:**
    *   Pivots are detected using `pandas_ta.pivot`.
    *   Box creation iterates through the detected pivot points.
    *   Box management (`_manage_boxes`) iterates through the stored lists (`self.bull_boxes`, `self.bear_boxes`) and updates their `active`, `violated`, and `right_idx` status based on the *latest* closing price in the analyzed DataFrame.
    *   Box cleanup removes the oldest *inactive* boxes first if the maximum number is exceeded.
8.  **`latest_values` Dictionary:** The `analyze` method returns this dictionary, which your bot can easily consume. It includes:
    *   All columns calculated on the DataFrame for the last row (e.g., `close`, `ema1`, `ema2`, `upper_band`, `lower_band`, `vol_norm_int`, `trend_up`, `trend_changed`).
    *   `active_bull_boxes`: A list of dictionaries, each representing an active bullish OB.
    *   `active_bear_boxes`: A list of dictionaries, each representing an active bearish OB.
9.  **Example Usage (`if __name__ == "__main__":`)**: Shows how to:
    *   Load data (replace dummy data with your source).
    *   Instantiate the `VolumaticOBAnalyzer`.
    *   Call `analyzer.analyze(your_df)`.
    *   Access and print the results from the returned dictionary.

**How to Integrate into Your Bot (e.g., v4.1.0):**

1.  **Instantiate:** In your `TradingBot`'s `__init__` or `start` method, after getting `market_info`, create an instance:
    ```python
    # Inside TradingBot class
    self.vol_ob_analyzer = VolumaticOBAnalyzer(
        market_info=self.market_info,
        # Pass relevant config values from self.config.settings
        trend_length=self.config.get('indicators', {}).get('vol_trend_length', 40), # Add these to config.json
        # ... other parameters ...
        ob_max_boxes=self.config.get('ob_max_boxes', 50) # Add to config.json
    )
    ```
2.  **Call in Cycle:** In your `TradingBot._run_cycle` method, after fetching `df`:
    ```python
    # Inside _run_cycle
    if df is not None and not df.empty:
        # Perform the Volumatic/OB analysis
        analysis_results = self.vol_ob_analyzer.analyze(df)

        if not analysis_results:
            self.logger.error(f"{FG_RED}Volumatic/OB Analysis failed. Skip cycle.{RESET}")
            return

        # --- Use the results ---
        latest_close = analysis_results.get('close')
        is_trend_up = analysis_results.get('is_trend_up')
        upper_band = analysis_results.get('upper_band')
        lower_band = analysis_results.get('lower_band')
        active_bull_obs = analysis_results.get('active_bull_boxes', [])
        active_bear_obs = analysis_results.get('active_bear_boxes', [])

        # Modify your signal generation logic in TradingAnalyzer or directly here
        # to incorporate is_trend_up, band proximity, OB proximity, etc.
        # Example: Check if price is near an active bullish OB and trend is up
        signal_action = "HOLD" # Default
        if is_trend_up:
            for ob in active_bull_obs:
                # Check if close is near or inside the OB top/bottom
                if latest_close and ob['bottom'] <= latest_close <= ob['top'] * Decimal("1.005"): # Example proximity check
                     self.logger.info(f"Price near active Bull OB {ob['id']}. Considering LONG.")
                     # Combine with other signals...
                     signal_action = "BUY" # Simplified example
                     break # Act on first relevant OB found

        # ... rest of your cycle logic using latest_close, signal_action etc. ...

    else:
        self.logger.error(f"{FG_RED}Market data failed. Skip cycle.{RESET}")
        return
    ```

Remember to add the necessary configuration parameters (like `vol_trend_length`, `ob_max_boxes`, etc.) to your `config.json` file. This Python version provides the computational core of the Pine Script indicator, ready for integration.
