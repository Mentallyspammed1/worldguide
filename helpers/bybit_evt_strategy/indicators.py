# --- START OF FILE indicators.py ---

# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Technical Indicators Module (v1.1 - Fixed EVT SuperSmoother)

Provides functions to calculate various technical indicators, primarily leveraging the
`pandas_ta` library for efficiency and breadth. Includes standard indicators,
pivot points, and level calculations. Designed to work with pandas DataFrames
containing OHLCV data.

Key Features:
- Wrappers around `pandas_ta` for common indicators.
- Calculation of Standard and Fibonacci Pivot Points.
- Calculation of Support/Resistance levels based on pivots and Fibonacci retracements.
- Custom Ehlers Volumetric Trend (EVT) implementation with SuperSmoother.
- A master function (`calculate_all_indicators`) to compute indicators based on a config.
- Robust error handling and logging.
- Clear type hinting and documentation.

Assumes input DataFrame has columns: 'open', 'high', 'low', 'close', 'volume'
and a datetime index (preferably UTC).
"""

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta  # type: ignore[import]
    PANDAS_TA_AVAILABLE = True
except ImportError:
    print("Error: 'pandas_ta' library not found. Please install it: pip install pandas_ta", file=sys.stderr)
    PANDAS_TA_AVAILABLE = False
    # sys.exit(1) # Optionally exit if pandas_ta is critical


# --- Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():  # Basic setup if run standalone or before main logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Default level

# --- Constants ---
MIN_PERIODS_DEFAULT = 50  # Default minimum number of data points for reliable calculations

# --- Pivot Point Calculations ---


def calculate_standard_pivot_points(high: float, low: float, close: float) -> dict[str, float]:
    """Calculates standard pivot points for the *next* period based on HLC."""
    if not all(isinstance(v, (int, float)) and pd.notna(v) for v in [high, low, close]):  # Use pd.notna
        logger.warning("Invalid input for standard pivot points (NaN or non-numeric).")
        return {}
    if low > high: logger.warning(f"Low ({low}) > High ({high}) for pivot calc.")
    pivots = {}
    try:
        pivot = (high + low + close) / 3.0; pivots['PP'] = pivot
        pivots['S1'] = (2 * pivot) - high; pivots['R1'] = (2 * pivot) - low
        pivots['S2'] = pivot - (high - low); pivots['R2'] = pivot + (high - low)
        pivots['S3'] = low - 2 * (high - pivot); pivots['R3'] = high + 2 * (pivot - low)
    except Exception as e: logger.error(f"Error calculating standard pivots: {e}", exc_info=True); return {}
    return pivots


def calculate_fib_pivot_points(high: float, low: float, close: float) -> dict[str, float]:
    """Calculates Fibonacci pivot points for the *next* period based on HLC."""
    if not all(isinstance(v, (int, float)) and pd.notna(v) for v in [high, low, close]):  # Use pd.notna
        logger.warning("Invalid input for Fibonacci pivot points (NaN or non-numeric)."); return {}
    if low > high: logger.warning(f"Low ({low}) > High ({high}) for Fib pivot calc.")
    fib_pivots = {}
    try:
        pivot = (high + low + close) / 3.0; fib_range = high - low
        if abs(fib_range) < 1e-9: logger.warning("Zero range, cannot calculate Fib Pivots accurately."); fib_pivots['PP'] = pivot; return fib_pivots
        fib_pivots['PP'] = pivot
        fib_pivots['S1'] = pivot - (0.382 * fib_range); fib_pivots['R1'] = pivot + (0.382 * fib_range)
        fib_pivots['S2'] = pivot - (0.618 * fib_range); fib_pivots['R2'] = pivot + (0.618 * fib_range)
        fib_pivots['S3'] = pivot - (1.000 * fib_range); fib_pivots['R3'] = pivot + (1.000 * fib_range)
    except Exception as e: logger.error(f"Error calculating Fib pivots: {e}", exc_info=True); return {}
    return fib_pivots

# --- Support / Resistance Level Calculation ---


def calculate_levels(df_period: pd.DataFrame, current_price: float | None = None) -> dict[str, Any]:
    """Calculates various support/resistance levels based on historical data."""
    levels: dict[str, Any] = {"support": {}, "resistance": {}, "pivot": None, "fib_retracements": {}, "standard_pivots": {}, "fib_pivots": {}}
    required_cols = ['high', 'low', 'close']
    if df_period is None or df_period.empty or not all(col in df_period.columns for col in required_cols): logger.warning("Cannot calculate levels: Invalid DataFrame."); return levels
    standard_pivots, fib_pivots = {}, {}

    # Use previous candle's data for pivots
    if len(df_period) >= 2:
        try:
            prev_row = df_period.iloc[-2]  # Use second to last row for previous candle HLC
            if not prev_row[required_cols].isnull().any():
                standard_pivots = calculate_standard_pivot_points(prev_row["high"], prev_row["low"], prev_row["close"])
                fib_pivots = calculate_fib_pivot_points(prev_row["high"], prev_row["low"], prev_row["close"])
            else: logger.warning("Previous candle data contains NaN, skipping pivot calculation.")
        except IndexError: logger.warning("IndexError calculating pivots (need >= 2 rows).")
        except Exception as e: logger.error(f"Error calculating pivots: {e}", exc_info=True)
    else: logger.warning("Cannot calculate pivots: Need >= 2 data points.")

    levels["standard_pivots"] = standard_pivots; levels["fib_pivots"] = fib_pivots
    levels["pivot"] = standard_pivots.get('PP') if standard_pivots else fib_pivots.get('PP')

    # Calculate Fib retracements over the whole period
    try:
        period_high = df_period["high"].max(); period_low = df_period["low"].min()
        if pd.notna(period_high) and pd.notna(period_low):
            period_diff = period_high - period_low
            if abs(period_diff) > 1e-9:
                levels["fib_retracements"] = {
                    "High": period_high,
                    "Fib 78.6%": period_low + period_diff * 0.786,
                    "Fib 61.8%": period_low + period_diff * 0.618,
                    "Fib 50.0%": period_low + period_diff * 0.5,
                    "Fib 38.2%": period_low + period_diff * 0.382,
                    "Fib 23.6%": period_low + period_diff * 0.236,
                    "Low": period_low
                }
            else: logger.debug("Period range near zero, skipping Fib retracements.")
        else: logger.warning("Could not calculate Fib retracements due to NaN in period High/Low.")
    except Exception as e: logger.error(f"Error calculating Fib retracements: {e}", exc_info=True)

    # Classify levels relative to current price or pivot
    try:
        # Use current price if provided, otherwise use calculated pivot point
        cp = float(current_price) if current_price is not None and pd.notna(current_price) else levels.get("pivot")

        if cp is not None and pd.notna(cp):
            all_levels = {**{f"Std {k}": v for k, v in standard_pivots.items() if pd.notna(v)},
                          **{f"Fib {k}": v for k, v in fib_pivots.items() if k != 'PP' and pd.notna(v)},
                          **{k: v for k, v in levels["fib_retracements"].items() if pd.notna(v)}}
            for label, value in all_levels.items():
                if value < cp: levels["support"][label] = value
                elif value > cp: levels["resistance"][label] = value
        else: logger.debug("Cannot classify S/R relative to current price/pivot (price or pivot is None/NaN).")
    except Exception as e: logger.error(f"Error classifying S/R levels: {e}", exc_info=True)

    # Sort levels for readability
    levels["support"] = dict(sorted(levels["support"].items(), key=lambda item: item[1], reverse=True))
    levels["resistance"] = dict(sorted(levels["resistance"].items(), key=lambda item: item[1]))
    return levels


# --- Custom Indicator Example ---

def calculate_vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series | None:
    """Calculates Volume Weighted Moving Average (VWMA)."""
    if not isinstance(close, pd.Series) or not isinstance(volume, pd.Series): return None
    if close.empty or volume.empty or len(close) != len(volume): return None  # Basic validation
    if length <= 0: logger.error(f"VWMA length must be positive: {length}"); return None
    if len(close) < length: logger.debug(f"VWMA data length {len(close)} < period {length}. Result will have NaNs.")  # Allow calculation

    try:
        pv = close * volume
        # Use min_periods=length to ensure enough data for the window sum
        cumulative_pv = pv.rolling(window=length, min_periods=length).sum()
        cumulative_vol = volume.rolling(window=length, min_periods=length).sum()
        # Avoid division by zero: replace 0 volume with NaN before dividing
        vwma = cumulative_pv / cumulative_vol.replace(0, np.nan)
        vwma.name = f"VWMA_{length}"
        return vwma
    except Exception as e:
        logger.error(f"Error calculating VWMA(length={length}): {e}", exc_info=True)
        return None


def ehlers_volumetric_trend(df: pd.DataFrame, length: int, multiplier: float | int) -> pd.DataFrame:
    """Calculate Ehlers Volumetric Trend using VWMA and SuperSmoother filter.
    Adds columns: 'vwma_X', 'smooth_vwma_X', 'evt_trend_X', 'evt_buy_X', 'evt_sell_X'.
    """
    required_cols = ['close', 'volume']
    if not all(col in df.columns for col in required_cols) or df.empty:
         logger.warning("EVT skipped: Missing columns or empty df.")
         return df
    if length <= 1 or multiplier <= 0:
        logger.warning(f"EVT skipped: Invalid params (len={length}, mult={multiplier}).")
        return df
    # Need at least length + 2 rows for smoother calculation
    if len(df) < length + 2:
        logger.warning(f"EVT skipped: Insufficient data rows ({len(df)}) for length {length}.")
        return df

    df_out = df.copy()
    vwma_col = f'vwma_{length}'
    smooth_col = f'smooth_vwma_{length}'
    trend_col = f'evt_trend_{length}'
    buy_col = f'evt_buy_{length}'
    sell_col = f'evt_sell_{length}'

    try:
        vwma = calculate_vwma(df_out['close'], df_out['volume'], length=length)
        if vwma is None or vwma.isnull().all():
            raise ValueError(f"VWMA calculation failed for EVT (length={length})")
        df_out[vwma_col] = vwma

        # SuperSmoother Filter Calculation (Corrected constants and implementation)
        # Constants based on Ehlers' formula
        arg = 1.414 * np.pi / length  # Corrected sqrt(2) approx
        a = np.exp(-arg)
        b = 2 * a * np.cos(arg)
        c2 = b
        c3 = -a * a
        c1 = 1 - c2 - c3

        # Initialize smoothed series & apply filter iteratively
        smoothed = pd.Series(np.nan, index=df_out.index, dtype=float)
        # Use .values for potentially faster access if DataFrame is large
        vwma_valid = df_out[vwma_col].values

        # Prime the first two values using VWMA itself (simple initialization)
        if len(df_out) > 0 and pd.notna(vwma_valid[0]): smoothed.iloc[0] = vwma_valid[0]
        if len(df_out) > 1 and pd.notna(vwma_valid[1]): smoothed.iloc[1] = vwma_valid[1]  # Simple init

        # Iterate starting from the third element (index 2)
        for i in range(2, len(df_out)):
            # Ensure current VWMA and previous smoothed values are valid
            if pd.notna(vwma_valid[i]):
                # Use previous smoothed value if valid, otherwise fallback (careful with fallback choice)
                # Using previous VWMA as fallback might introduce lag/noise
                sm1 = smoothed.iloc[i - 1] if pd.notna(smoothed.iloc[i - 1]) else vwma_valid[i - 1]
                sm2 = smoothed.iloc[i - 2] if pd.notna(smoothed.iloc[i - 2]) else vwma_valid[i - 2]
                # Only calculate if all inputs are valid numbers
                if pd.notna(sm1) and pd.notna(sm2):
                    smoothed.iloc[i] = c1 * vwma_valid[i] + c2 * sm1 + c3 * sm2

        df_out[smooth_col] = smoothed

        # Trend Determination
        mult_h = 1.0 + float(multiplier) / 100.0
        mult_l = 1.0 - float(multiplier) / 100.0
        shifted_smooth = df_out[smooth_col].shift(1)

        # Conditions - compare only where both current and previous smoothed values are valid
        valid_comparison = pd.notna(df_out[smooth_col]) & pd.notna(shifted_smooth)
        up_trend_cond = valid_comparison & (df_out[smooth_col] > shifted_smooth * mult_h)
        down_trend_cond = valid_comparison & (df_out[smooth_col] < shifted_smooth * mult_l)

        # Vectorized trend calculation using forward fill
        trend = pd.Series(np.nan, index=df_out.index, dtype=float)  # Start with NaN
        trend[up_trend_cond] = 1.0   # Mark uptrend start
        trend[down_trend_cond] = -1.0  # Mark downtrend start

        # Forward fill the trend signal (1 or -1 persists), fill initial NaNs with 0 (neutral)
        df_out[trend_col] = trend.ffill().fillna(0).astype(int)

        # Buy/Sell Signal Generation (Trend Initiation)
        trend_shifted = df_out[trend_col].shift(1, fill_value=0)  # Previous period's trend (fill start with 0)
        df_out[buy_col] = (df_out[trend_col] == 1) & (trend_shifted != 1)  # Trend becomes 1
        df_out[sell_col] = (df_out[trend_col] == -1) & (trend_shifted != -1)  # Trend becomes -1

        logger.debug(f"Ehlers Volumetric Trend (len={length}, mult={multiplier}) calculated.")
        return df_out

    except Exception as e:
        logger.error(f"Error in EVT(len={length}, mult={multiplier}): {e}", exc_info=True)
        # Add NaN columns to original df to signal failure, maintain structure
        for col in [vwma_col, smooth_col, trend_col, buy_col, sell_col]:
            if col not in df_out.columns: df_out[col] = np.nan
        return df


# --- Master Indicator Calculation Function ---
def calculate_all_indicators(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Calculates enabled technical indicators using pandas_ta and custom functions."""
    if df is None or df.empty: logger.error("Input DataFrame empty."); return pd.DataFrame()
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]; logger.error(f"Input DataFrame missing: {missing}."); return df.copy()

    df_out = df.copy()
    settings = config.get("indicator_settings", {})
    flags = config.get("analysis_flags", {})
    min_rows = settings.get("min_data_periods", MIN_PERIODS_DEFAULT)

    # Check for sufficient valid rows AFTER ensuring columns exist
    if len(df_out.dropna(subset=required_cols)) < min_rows:
        logger.warning(f"Insufficient valid rows ({len(df_out.dropna(subset=required_cols))}) < {min_rows}. Results may be NaN/inaccurate.")

    # --- Calculate Standard Indicators (using pandas_ta if available) ---
    if PANDAS_TA_AVAILABLE:
        atr_len = settings.get('atr_period', 14)
        if flags.get("use_atr", False) and atr_len > 0:
            try:
                logger.debug(f"Calculating ATR(length={atr_len})")
                df_out.ta.atr(length=atr_len, append=True)  # Appends 'ATRr_X'
            except Exception as e: logger.error(f"Error calculating ATR({atr_len}): {e}", exc_info=False)

        # Add other pandas_ta indicators based on flags here...
        # Example: EMA
        # if flags.get("use_ema"):
        #    try:
        #         ema_s = settings.get('ema_short_period', 12); ema_l = settings.get('ema_long_period', 26)
        #         if ema_s > 0: df_out.ta.ema(length=ema_s, append=True)
        #         if ema_l > 0: df_out.ta.ema(length=ema_l, append=True)
        #    except Exception as e: logger.error(f"Error calculating EMA: {e}", exc_info=False)
    else:
        logger.warning("pandas_ta not available. Skipping standard indicators (ATR, etc.).")

    # --- Calculate Custom Strategy Indicators ---
    strategy_config = config.get('strategy_params', {}).get(config.get('strategy', {}).get('name', '').lower(), {})
    # Ehlers Volumetric Trend (Primary)
    if flags.get("use_evt"):  # Generic flag or strategy-specific check
        try:
            # Get params from strategy_config first, fallback to general indicator_settings
            evt_len = strategy_config.get('evt_length', settings.get('evt_length', 7))
            evt_mult = strategy_config.get('evt_multiplier', settings.get('evt_multiplier', 2.5))
            if evt_len > 1 and evt_mult > 0:
                logger.debug(f"Calculating EVT(len={evt_len}, mult={evt_mult})")
                df_out = ehlers_volumetric_trend(df_out, evt_len, float(evt_mult))
            else: logger.warning(f"Invalid parameters for EVT (len={evt_len}, mult={evt_mult}), skipping.")
        except Exception as e: logger.error(f"Error calculating EVT: {e}", exc_info=True)

    # Example: Dual EVT Strategy specific logic (if needed)
    # if config.get('strategy',{}).get('name','').lower() == "dual_ehlers_volumetric":
    #      # ... calculate confirmation EVT ...

    logger.debug(f"Finished calculating indicators. Final DataFrame shape: {df_out.shape}")
    # Optional: remove duplicate columns if any arose
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    return df_out


# --- Example Standalone Usage ---
if __name__ == "__main__":
    print("-" * 60); print("--- Indicator Module Demo (v1.1) ---"); print("-" * 60)
    logger.setLevel(logging.DEBUG)  # Set logger to debug for demo

    # Create dummy data (more realistic price movement)
    periods = 200
    np.random.seed(42)  # for reproducibility
    returns = np.random.normal(loc=0.0001, scale=0.01, size=periods)
    prices = 55000 * np.exp(np.cumsum(returns))  # Start price 55000

    # Ensure OHLC are consistent
    data = {'timestamp': pd.date_range(start='2023-01-01', periods=periods, freq='H', tz='UTC')}
    df_test = pd.DataFrame(data).set_index('timestamp')
    df_test['open'] = prices[:-1]
    df_test['close'] = prices[1:]
    # Simulate High/Low relative to Open/Close
    high_factor = 1 + np.random.uniform(0, 0.005, periods - 1)
    low_factor = 1 - np.random.uniform(0, 0.005, periods - 1)
    df_test = df_test.iloc[:-1].copy()  # Adjust size to match open/close
    df_test['high'] = df_test[['open', 'close']].max(axis=1) * high_factor
    df_test['low'] = df_test[['open', 'close']].min(axis=1) * low_factor
    # Ensure H >= O, H >= C and L <= O, L <= C
    df_test['high'] = df_test[['open', 'close', 'high']].max(axis=1)
    df_test['low'] = df_test[['open', 'close', 'low']].min(axis=1)
    df_test['volume'] = np.random.uniform(100, 2000, periods - 1)

    print(f"Input shape: {df_test.shape}"); print(f"Input head:\n{df_test.head()}"); print(f"Input tail:\n{df_test.tail()}")

    # Example Config
    test_config = {
        "indicator_settings": {
            "min_data_periods": 50,
            "atr_period": 14,
            "evt_length": 7,
            "evt_multiplier": 2.5
        },
        "analysis_flags": {
            "use_atr": True,
            "use_evt": True
        },
        # These mimic the structure expected by the function
        "strategy_params": {'ehlers_volumetric': {'evt_length': 7, 'evt_multiplier': 2.5}},
        "strategy": {'name': 'ehlers_volumetric'}
    }

    df_results = calculate_all_indicators(df_test, test_config)
    print("-" * 60); print(f"Output shape: {df_results.shape}"); print(f"Output tail:\n{df_results.tail()}"); print("-" * 60)
    print(f"Output columns ({len(df_results.columns)}): {df_results.columns.tolist()}"); print("-" * 60)

    # Check for NaNs in the last row of added indicators
    added_cols = df_results.columns.difference(df_test.columns)
    last_row_nans = df_results[added_cols].iloc[-1].isnull().sum()
    print(f"NaNs in last row of added indicators ({len(added_cols)} cols): {last_row_nans}")
    print(f"Last row details:\n{df_results[added_cols].iloc[-1]}")

# --- END OF FILE indicators.py ---
