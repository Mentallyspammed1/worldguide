#!/usr/bin/env python

"""Technical Indicators Module

Provides functions to calculate various technical indicators, primarily leveraging the
`pandas_ta` library for efficiency and breadth. Includes standard indicators,
pivot points, and level calculations. Designed to work with pandas DataFrames
containing OHLCV data.

Key Features:
- Wrappers around `pandas_ta` for common indicators.
- Calculation of Standard and Fibonacci Pivot Points.
- Calculation of Support/Resistance levels based on pivots and Fibonacci retracements.
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
except ImportError:
    print(
        "Error: 'pandas_ta' library not found. Please install it: pip install pandas_ta"
    )
    sys.exit(1)


# --- Setup ---
# Use logger from the main script or initialize a basic one if run standalone
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Default level

# --- Constants ---
MIN_PERIODS_DEFAULT = (
    50  # Default minimum number of data points for reliable calculations
)

# --- Helper Functions ---
# (Removed safe_decimal_conversion as we'll primarily use float64 for performance,
# unless specific high precision is absolutely required and proven necessary)


# --- Pivot Point Calculations ---


def calculate_standard_pivot_points(
    high: float, low: float, close: float
) -> dict[str, float]:
    """Calculates standard pivot points for the *next* period based on the
    *current* or *previous* period's High, Low, Close (HLC).

    Args:
        high: The high price of the period.
        low: The low price of the period.
        close: The closing price of the period.

    Returns:
        A dictionary containing the Pivot Point (PP) and standard
        Resistance (R1, R2, R3) and Support (S1, S2, S3) levels.
        Returns an empty dictionary if inputs are invalid.
    """
    if not all(
        isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]
    ):
        logger.warning("Invalid input for standard pivot points (NaN or non-numeric).")
        return {}
    if low > high:
        logger.warning(
            f"Low price ({low}) is higher than High price ({high}) for pivot calculation."
        )
        # Optionally swap or return empty, here we proceed but results might be weird
        # low, high = high, low

    pivots = {}
    try:
        pivot = (high + low + close) / 3.0
        pivots["PP"] = pivot
        pivots["S1"] = (2 * pivot) - high
        pivots["R1"] = (2 * pivot) - low
        pivots["S2"] = pivot - (high - low)
        pivots["R2"] = pivot + (high - low)
        pivots["S3"] = low - 2 * (high - pivot)
        pivots["R3"] = high + 2 * (pivot - low)
    except Exception as e:
        logger.error(f"Error calculating standard pivot points: {e}", exc_info=True)
        return {}
    return pivots


def calculate_fib_pivot_points(
    high: float, low: float, close: float
) -> dict[str, float]:
    """Calculates Fibonacci pivot points for the *next* period based on the
    *current* or *previous* period's High, Low, Close (HLC).

    Args:
        high: The high price of the period.
        low: The low price of the period.
        close: The closing price of the period.

    Returns:
        A dictionary containing the Pivot Point (PP) and Fibonacci-based
        Resistance (R1, R2, R3) and Support (S1, S2, S3) levels.
        Returns an empty dictionary if inputs are invalid or range is zero.
    """
    if not all(
        isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]
    ):
        logger.warning("Invalid input for Fibonacci pivot points (NaN or non-numeric).")
        return {}
    if low > high:
        logger.warning(
            f"Low price ({low}) is higher than High price ({high}) for Fib pivot calculation."
        )
        # low, high = high, low # Optional swap

    fib_pivots = {}
    try:
        pivot = (high + low + close) / 3.0
        fib_range = high - low

        # Avoid division by zero or issues with flat candles
        if abs(fib_range) < 1e-9:  # Use a small threshold
            logger.warning(
                "Range (High - Low) is zero or near-zero, cannot calculate Fibonacci Pivots accurately."
            )
            fib_pivots["PP"] = pivot  # Still return the pivot point
            return fib_pivots

        fib_pivots["PP"] = pivot
        fib_pivots["S1"] = pivot - (0.382 * fib_range)
        fib_pivots["R1"] = pivot + (0.382 * fib_range)
        fib_pivots["S2"] = pivot - (0.618 * fib_range)
        fib_pivots["R2"] = pivot + (0.618 * fib_range)
        fib_pivots["S3"] = pivot - (
            1.000 * fib_range
        )  # S3 = Low - (High - Pivot)? -> Equivalent to Pivot - Range
        fib_pivots["R3"] = pivot + (
            1.000 * fib_range
        )  # R3 = High + (Pivot - Low)? -> Equivalent to Pivot + Range

    except Exception as e:
        logger.error(f"Error calculating Fibonacci pivot points: {e}", exc_info=True)
        return {}
    return fib_pivots


# --- Support / Resistance Level Calculation ---


def calculate_levels(
    df_period: pd.DataFrame, current_price: float | None = None
) -> dict[str, Any]:
    """Calculates various support/resistance levels based on historical data
    for a given period (e.g., daily, weekly) and the current price.

    Includes:
    - Standard Pivot Points (based on the *previous* candle's HLC).
    - Fibonacci Pivot Points (based on the *previous* candle's HLC).
    - Fibonacci Retracement levels (based on the *entire period's* high/low range).

    Args:
        df_period: DataFrame containing OHLC data for the relevant period (e.g., last 24h for daily pivots).
                   Must contain at least 'high', 'low', 'close' columns.
                   Requires at least 2 rows to get previous candle data for pivots.
        current_price: The current market price, used to classify levels as
                       support (below current price) or resistance (above current price).
                       If None, classification is skipped.

    Returns:
        A dictionary containing:
        - 'support': {label: value} dict of levels below current_price.
        - 'resistance': {label: value} dict of levels above current_price.
        - 'pivot': The main standard pivot point value (float, or None).
        - 'fib_retracements': {label: value} dict of Fibonacci retracement levels.
        - 'standard_pivots': {label: value} dict of all standard pivot levels.
        - 'fib_pivots': {label: value} dict of all Fibonacci pivot levels.
    """
    levels: dict[str, Any] = {
        "support": {},
        "resistance": {},
        "pivot": None,
        "fib_retracements": {},
        "standard_pivots": {},
        "fib_pivots": {},
    }
    required_cols = ["high", "low", "close"]
    if (
        df_period is None
        or df_period.empty
        or not all(col in df_period.columns for col in required_cols)
    ):
        logger.warning(
            "Cannot calculate levels: DataFrame is empty or missing HLC columns."
        )
        return levels
    if len(df_period) < 2:
        logger.warning(
            "Cannot calculate pivot points: Need at least 2 data points for previous candle HLC."
        )
        # Proceed with only retracements if possible

    try:
        # --- Pivot Points (based on *previous* candle HLC) ---
        prev_high = df_period["high"].iloc[-2]
        prev_low = df_period["low"].iloc[-2]
        prev_close = df_period["close"].iloc[-2]

        standard_pivots = calculate_standard_pivot_points(
            prev_high, prev_low, prev_close
        )
        if standard_pivots:
            levels["standard_pivots"] = standard_pivots
            levels["pivot"] = standard_pivots.get("PP")  # Store main pivot separately

        fib_pivots = calculate_fib_pivot_points(prev_high, prev_low, prev_close)
        if fib_pivots:
            levels["fib_pivots"] = fib_pivots
            # Ensure PP from Fib pivots doesn't overwrite standard if both exist
            if levels["pivot"] is None:
                levels["pivot"] = fib_pivots.get("PP")

        # --- Fibonacci Retracement (based on *entire period's* high/low range) ---
        period_high = df_period["high"].max()
        period_low = df_period["low"].min()
        period_diff = period_high - period_low

        if abs(period_diff) > 1e-9:
            levels["fib_retracements"] = {
                "High": period_high,
                "Fib 78.6%": period_low
                + period_diff * 0.786,  # Common alternative/addition
                "Fib 61.8%": period_low + period_diff * 0.618,
                "Fib 50.0%": period_low + period_diff * 0.5,
                "Fib 38.2%": period_low + period_diff * 0.382,
                "Fib 23.6%": period_low + period_diff * 0.236,
                "Low": period_low,
            }
            # Alternative calculation from High down:
            # "Fib 23.6%": period_high - period_diff * 0.236,
            # "Fib 38.2%": period_high - period_diff * 0.382, ... etc.

        # --- Classify Levels relative to Current Price ---
        if current_price is not None and isinstance(current_price, (int, float)):
            all_calculated_levels = {
                **{f"Std {k}": v for k, v in standard_pivots.items()},
                **{
                    f"Fib {k}": v for k, v in fib_pivots.items() if k != "PP"
                },  # Avoid duplicate PP label
                **levels["fib_retracements"],
            }

            for label, value in all_calculated_levels.items():
                if isinstance(value, (int, float)):  # Ensure value is numeric
                    if value < current_price:
                        levels["support"][label] = value
                    elif value > current_price:
                        levels["resistance"][label] = value
                    # If value == current_price, it's neither S nor R, could be noted separately if needed
        else:
            logger.debug("Current price not provided, skipping S/R classification.")
            # Optionally populate S/R based only on pivot point if needed
            if levels["pivot"] is not None:
                pivot_val = levels["pivot"]
                all_calculated_levels = {
                    **{f"Std {k}": v for k, v in standard_pivots.items()},
                    **{f"Fib {k}": v for k, v in fib_pivots.items() if k != "PP"},
                    **levels["fib_retracements"],
                }
                for label, value in all_calculated_levels.items():
                    if isinstance(value, (int, float)):
                        if value < pivot_val:
                            levels["support"][label] = value
                        elif value > pivot_val:
                            levels["resistance"][label] = value

    except IndexError:
        logger.warning(
            "IndexError calculating levels, likely due to insufficient data (< 2 rows). Only retracements might be available."
        )
        # Recalculate only retracements if possible
        try:
            period_high = df_period["high"].max()
            period_low = df_period["low"].min()
            period_diff = period_high - period_low
            if abs(period_diff) > 1e-9:
                levels["fib_retracements"] = {
                    "High": period_high,
                    "Fib 78.6%": period_low + period_diff * 0.786,
                    "Fib 61.8%": period_low + period_diff * 0.618,
                    "Fib 50.0%": period_low + period_diff * 0.5,
                    "Fib 38.2%": period_low + period_diff * 0.382,
                    "Fib 23.6%": period_low + period_diff * 0.236,
                    "Low": period_low,
                }
        except Exception as e_retr:
            logger.error(
                f"Error calculating Fibonacci retracements even after initial failure: {e_retr}"
            )

    except Exception as e:
        logger.error(f"Unexpected error calculating S/R levels: {e}", exc_info=True)

    # Sort support descending, resistance ascending for clarity
    levels["support"] = dict(
        sorted(levels["support"].items(), key=lambda item: item[1], reverse=True)
    )
    levels["resistance"] = dict(
        sorted(levels["resistance"].items(), key=lambda item: item[1])
    )

    return levels


# --- Custom Indicator Example ---


def calculate_vwma(
    close: pd.Series, volume: pd.Series, length: int
) -> pd.Series | None:
    """Calculates Volume Weighted Moving Average (VWMA)."""
    if (
        close is None
        or volume is None
        or close.empty
        or volume.empty
        or length <= 0
        or len(close) < length
        or len(close) != len(volume)
    ):
        logger.warning(
            f"VWMA calculation skipped: Invalid inputs or insufficient length (need {length})."
        )
        return None
    try:
        # pandas_ta doesn't have a dedicated VWMA, implement manually
        pv = close * volume
        cumulative_pv = pv.rolling(window=length, min_periods=length).sum()
        cumulative_vol = volume.rolling(window=length, min_periods=length).sum()
        vwma = cumulative_pv / cumulative_vol
        # Handle potential division by zero if volume is zero over the window
        vwma.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Name the series for clarity if needed later
        vwma.name = f"VWMA_{length}"
        return vwma
    except Exception as e:
        logger.error(f"Error calculating VWMA(length={length}): {e}", exc_info=True)
        return None


def ehlers_volumetric_trend(
    df: pd.DataFrame, length: int, multiplier: float | int
) -> pd.DataFrame:
    """Calculate Ehlers Volumetric Trend indicator using VWMA and a SuperSmoother filter.
    Adds columns directly to the input DataFrame:
    - f'vwma_{length}'
    - f'smooth_vwma_{length}'
    - f'evt_trend_{length}' (1 for up, -1 for down, 0 for neutral/start)
    - f'evt_buy_{length}' (True on bullish trend initiation)
    - f'evt_sell_{length}' (True on bearish trend initiation)

    Args:
        df: DataFrame with 'close' and 'volume' columns.
        length: The period length for VWMA and smoother calculation.
        multiplier: The percentage multiplier (e.g., 2.5 for 2.5%) to define trend bands.

    Returns:
        The input DataFrame with EVT columns added, or the original DataFrame if calculation fails.
    """
    required_cols = ["close", "volume"]
    if (
        not all(col in df.columns for col in required_cols)
        or df.empty
        or length <= 1
        or multiplier <= 0
    ):
        logger.warning(
            f"EVT calculation skipped: Missing columns, empty df, or invalid params (len={length}, mult={multiplier})."
        )
        return df  # Return original df without modification

    df_out = (
        df.copy()
    )  # Work on a copy to avoid modifying original if errors occur mid-way
    vwma_col = f"vwma_{length}"
    smooth_col = f"smooth_vwma_{length}"
    trend_col = f"evt_trend_{length}"
    buy_col = f"evt_buy_{length}"
    sell_col = f"evt_sell_{length}"

    try:
        vwma = calculate_vwma(df_out["close"], df_out["volume"], length=length)
        if vwma is None or vwma.isnull().all():
            raise ValueError(f"VWMA calculation failed for EVT (length={length})")
        df_out[vwma_col] = vwma

        # SuperSmoother Filter Calculation (constants depend on length)
        a = np.exp(-1.414 * np.pi / length)
        b = 2 * a * np.cos(1.414 * np.pi / length)
        c2 = b
        c3 = -a * a
        c1 = 1 - c2 - c3

        # Initialize smoothed series with NaNs
        smoothed = pd.Series(np.nan, index=df_out.index, dtype=float)
        vwma_vals = df_out[vwma_col].values  # Use numpy array for faster access in loop

        # Apply filter iteratively - requires at least 2 previous values
        for i in range(2, len(df_out)):
            if not np.isnan(vwma_vals[i]):
                # Use previous smoothed value if available, else use VWMA value as approximation (or 0)
                sm1 = (
                    smoothed.iloc[i - 1]
                    if pd.notna(smoothed.iloc[i - 1])
                    else (vwma_vals[i - 1] if pd.notna(vwma_vals[i - 1]) else 0)
                )
                sm2 = (
                    smoothed.iloc[i - 2]
                    if pd.notna(smoothed.iloc[i - 2])
                    else (vwma_vals[i - 2] if pd.notna(vwma_vals[i - 2]) else 0)
                )
                smoothed.iloc[i] = c1 * vwma_vals[i] + c2 * sm1 + c3 * sm2
            # else: smoothed remains NaN for this index

        df_out[smooth_col] = smoothed

        # Trend Determination
        mult_h = 1.0 + float(multiplier) / 100.0
        mult_l = 1.0 - float(multiplier) / 100.0
        shifted_smooth = df_out[smooth_col].shift(1)

        # Initialize trend with 0
        trend = pd.Series(0, index=df_out.index, dtype=int)

        # Conditions for trend change (only where smooth and shifted_smooth are valid)
        up_trend_condition = (
            (df_out[smooth_col] > shifted_smooth * mult_h)
            & pd.notna(df_out[smooth_col])
            & pd.notna(shifted_smooth)
        )
        down_trend_condition = (
            (df_out[smooth_col] < shifted_smooth * mult_l)
            & pd.notna(df_out[smooth_col])
            & pd.notna(shifted_smooth)
        )

        # Apply trend logic iteratively or vectorized if possible (iterative for clarity here)
        # Note: Direct numpy where might be faster but needs careful handling of NaNs and initial state
        last_trend = 0
        for i in range(len(df_out)):
            current_trend = last_trend
            if up_trend_condition.iloc[i]:
                current_trend = 1
            elif down_trend_condition.iloc[i]:
                current_trend = -1
            # Handle initial NaNs - keep trend 0 until a signal occurs
            if pd.isna(df_out[smooth_col].iloc[i]) or pd.isna(shifted_smooth.iloc[i]):
                current_trend = (
                    0  # Or ffill? Paper implies reset. Let's use 0 for undefined.
                )

            trend.iloc[i] = current_trend
            last_trend = current_trend  # Update for next iteration

        # Alternative using ffill after initial calculation (simpler but slightly different logic)
        # trend[up_trend_condition] = 1
        # trend[down_trend_condition] = -1
        # trend = trend.replace(0, np.nan).ffill().fillna(0).astype(int) # Fill gaps

        df_out[trend_col] = trend

        # Buy/Sell Signal Generation (Trend Initiation)
        trend_shifted = (
            df_out[trend_col].shift(1).fillna(0)
        )  # Fill NaN for first comparison
        df_out[buy_col] = (df_out[trend_col] == 1) & (trend_shifted != 1)
        df_out[sell_col] = (df_out[trend_col] == -1) & (trend_shifted != -1)

        logger.debug(
            f"Ehlers Volumetric Trend (len={length}, mult={multiplier}) calculated."
        )
        return df_out

    except Exception as e:
        logger.error(
            f"Error in EVT(len={length}, mult={multiplier}): {e}", exc_info=True
        )
        # Add NaN columns to original df before returning to signal failure
        df[vwma_col] = np.nan
        df[smooth_col] = np.nan
        df[trend_col] = np.nan
        df[buy_col] = np.nan
        df[sell_col] = np.nan
        return df  # Return original df with NaN columns


# --- Master Indicator Calculation Function ---


def calculate_all_indicators(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Calculates enabled technical indicators using pandas_ta and custom functions,
    adding them as columns to the DataFrame based on the provided configuration.

    Args:
        df: Input DataFrame with 'open', 'high', 'low', 'close', 'volume'.
            Assumes DataFrame index is datetime. Modifies a copy.
        config: Configuration dictionary, expected to have:
            - 'indicator_settings': Dict with parameters for each indicator (e.g., periods).
            - 'analysis_flags': Dict with boolean flags to enable/disable indicators.
            - Optionally 'strategy_params' for strategy-specific indicators like EVT.

    Returns:
        A new DataFrame with calculated indicator columns added.
        Returns an empty DataFrame if input is invalid.
    """
    if df is None or df.empty:
        logger.error("Input DataFrame is empty. Cannot calculate indicators.")
        return pd.DataFrame()

    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(
            f"Input DataFrame missing required columns: {missing}. Cannot calculate indicators."
        )
        # Consider returning df.copy() or raising error depending on desired behavior
        return df.copy()  # Return a copy to avoid modifying original on failure

    df_out = df.copy()  # Work on a copy

    settings = config.get("indicator_settings", {})
    flags = config.get("analysis_flags", {})
    min_rows_needed = settings.get("min_data_periods", MIN_PERIODS_DEFAULT)

    # Check length after potentially dropping initial NaNs (pandas_ta handles internal NaNs well)
    if len(df_out.dropna(subset=required_cols)) < min_rows_needed:
        logger.warning(
            f"Insufficient valid rows ({len(df_out.dropna(subset=required_cols))}) "
            f"after dropping NaNs in OHLCV. Need at least {min_rows_needed} for reliable calculations. "
            f"Results may be incomplete or NaN."
        )
        # Proceed cautiously

    # --- Helper to safely get parameters ---
    def get_param(name: str, default: Any = None) -> Any:
        return settings.get(name, default)

    # --- Calculate Indicators using pandas_ta ---
    # Use try-except blocks for each indicator family or critical indicator
    # Use append=False and assign results explicitly for clarity and control

    # Moving Averages
    try:
        if flags.get("use_sma"):  # Example flag name
            sma_s = get_param("sma_short_period", 10)
            sma_l = get_param("sma_long_period", 50)
            if sma_s > 0:
                df_out.ta.sma(length=sma_s, append=True)
            if sma_l > 0:
                df_out.ta.sma(length=sma_l, append=True)
        if flags.get("use_ema"):  # Example flag name
            ema_s = get_param("ema_short_period", 12)
            ema_l = get_param("ema_long_period", 26)
            if ema_s > 0:
                df_out.ta.ema(length=ema_s, adjust=False, append=True)
            if ema_l > 0:
                df_out.ta.ema(length=ema_l, adjust=False, append=True)
        if flags.get("use_hma"):  # Example flag name
            hma_len = get_param("hma_length", 9)
            if hma_len > 0:
                df_out.ta.hma(length=hma_len, append=True)
        if flags.get("use_vwap"):  # Example flag name (Rolling VWAP)
            vwap_len = get_param(
                "vwap_length"
            )  # Can be None for standard calc, or int for rolling
            df_out.ta.vwap(
                length=vwap_len, append=True
            )  # pandas_ta handles OHLCV input
    except Exception as e:
        logger.error(f"Error calculating Moving Averages: {e}", exc_info=True)

    # Oscillators
    try:
        if flags.get("use_rsi"):
            rsi_len = get_param("rsi_period", 14)
            if rsi_len > 0:
                df_out.ta.rsi(length=rsi_len, append=True)
        if flags.get("use_stochrsi"):
            stoch_len = get_param("stoch_rsi_period", 14)
            rsi_len_stoch = get_param("rsi_period", 14)  # Often uses same RSI period
            k = get_param("stoch_k_period", 3)
            d = get_param("stoch_d_period", 3)
            if stoch_len > 0 and rsi_len_stoch > 0 and k > 0 and d > 0:
                df_out.ta.stochrsi(
                    length=stoch_len, rsi_length=rsi_len_stoch, k=k, d=d, append=True
                )
        if flags.get("use_cci"):
            cci_len = get_param("cci_period", 20)
            if cci_len > 0:
                df_out.ta.cci(length=cci_len, append=True)
        if flags.get("use_williams_r"):
            willr_len = get_param("williams_r_period", 14)
            if willr_len > 0:
                df_out.ta.willr(length=willr_len, append=True)
        if flags.get("use_mfi"):
            mfi_len = get_param("mfi_period", 14)
            if mfi_len > 0:
                df_out.ta.mfi(length=mfi_len, append=True)
        if flags.get("use_momentum"):
            mom_len = get_param("momentum_period", 10)
            if mom_len > 0:
                df_out.ta.mom(length=mom_len, append=True)
        if flags.get("use_roc"):
            roc_len = get_param("roc_length", 10)
            if roc_len > 0:
                df_out.ta.roc(length=roc_len, append=True)
        if flags.get("use_awesome_oscillator"):
            ao_fast = get_param("ao_fast", 5)
            ao_slow = get_param("ao_slow", 34)
            if ao_fast > 0 and ao_slow > 0 and ao_fast < ao_slow:
                df_out.ta.ao(fast=ao_fast, slow=ao_slow, append=True)
    except Exception as e:
        logger.error(f"Error calculating Oscillators: {e}", exc_info=True)

    # MACD
    try:
        if flags.get("use_macd"):
            fast = get_param("macd_fast", 12)
            slow = get_param("macd_slow", 26)
            signal = get_param("macd_signal", 9)
            if fast > 0 and slow > 0 and signal > 0 and fast < slow:
                df_out.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}", exc_info=True)

    # Volatility / Bands / Channels
    try:
        # ATR is often needed by other indicators/strategies, calculate if needed or always?
        # Let's calculate it if any dependent flag is true or if explicitly requested.
        atr_needed = (
            flags.get("use_atr")
            or flags.get("use_bollinger_bands")
            or flags.get("use_keltner_channels")
        )
        if atr_needed:
            atr_len = get_param("atr_period", 14)
            if atr_len > 0:
                # Calculate ATR separately first if needed by others
                atr_result = df_out.ta.atr(length=atr_len, append=False)
                if atr_result is not None:
                    df_out[atr_result.name] = atr_result  # Add ATR column

        if flags.get("use_bollinger_bands"):
            bb_len = get_param("bollinger_bands_period", 20)
            bb_std = float(get_param("bollinger_bands_std_dev", 2.0))
            if bb_len > 0 and bb_std > 0:
                df_out.ta.bbands(length=bb_len, std=bb_std, append=True)

        if flags.get("use_keltner_channels"):
            kc_len = get_param("kc_length", 20)
            kc_atr_len = get_param(
                "kc_atr_length", 10
            )  # Often uses different ATR length
            kc_mult = float(get_param("kc_multiplier", 2.0))
            # Ensure ATR for KC is calculated if needed
            if f"ATRr_{kc_atr_len}" not in df_out.columns and kc_atr_len > 0:
                kc_atr_result = df_out.ta.atr(length=kc_atr_len, append=False)
                if kc_atr_result is not None:
                    df_out[kc_atr_result.name] = kc_atr_result

            if kc_len > 0 and kc_atr_len > 0 and kc_mult > 0:
                # KC calculation might fail if the specific ATR column isn't present
                # pandas_ta kc usually recalculates ATR internally if needed, but check its behavior
                df_out.ta.kc(
                    length=kc_len,
                    atr_length=kc_atr_len,
                    scalar=kc_mult,
                    mamode="EMA",
                    append=True,
                )  # Using EMA for center line is common

    except Exception as e:
        logger.error(f"Error calculating Volatility/Bands/Channels: {e}", exc_info=True)

    # Trend
    try:
        if flags.get("use_adx"):
            adx_len = get_param("adx_period", 14)
            if adx_len > 0:
                df_out.ta.adx(length=adx_len, append=True)
        if flags.get("use_psar"):
            step = get_param("psar_step", 0.02)
            max_step = get_param("psar_max_step", 0.2)
            if step > 0 and max_step > step:
                # Need to handle PSAR output which is a dict/dataframe
                psar_results = df_out.ta.psar(af=step, max_af=max_step, append=False)
                if psar_results is not None and isinstance(psar_results, pd.DataFrame):
                    # Choose columns to append, e.g., PSAR value, trend direction, reversals
                    # Example: Append the main PSAR line
                    psar_col_name = (
                        f"PSARl_{step}_{max_step}"  # Long entry PSAR (adjust if needed)
                    )
                    psar_col_name_s = f"PSARs_{step}_{max_step}"  # Short entry PSAR
                    if psar_col_name in psar_results:
                        df_out[psar_col_name] = psar_results[psar_col_name]
                    if psar_col_name_s in psar_results:
                        df_out[psar_col_name_s] = psar_results[psar_col_name_s]
                    # Potentially add reversal indicators: PSARaf, PSARr
    except Exception as e:
        logger.error(f"Error calculating Trend indicators: {e}", exc_info=True)

    # Volume Based
    try:
        if flags.get("use_obv"):
            df_out.ta.obv(append=True)
        if flags.get("use_adosc"):  # Accumulation/Distribution Oscillator
            fast = get_param("adosc_fast", 3)  # Default ta values
            slow = get_param("adosc_slow", 10)  # Default ta values
            if fast > 0 and slow > 0 and fast < slow:
                df_out.ta.adosc(fast=fast, slow=slow, append=True)

        # Volume MA (simple rolling mean)
        if flags.get("use_volume_ma"):
            vol_ma_len = get_param("volume_ma_period", 20)
            if vol_ma_len > 0 and "volume" in df_out.columns:
                df_out[f"VOLUME_MA_{vol_ma_len}"] = (
                    df_out["volume"]
                    .rolling(window=vol_ma_len, min_periods=vol_ma_len)
                    .mean()
                )

    except Exception as e:
        logger.error(f"Error calculating Volume indicators: {e}", exc_info=True)

    # Ichimoku Cloud
    try:
        if flags.get("use_ichimoku"):
            tenkan = get_param("ichimoku_tenkan", 9)
            kijun = get_param("ichimoku_kijun", 26)
            senkou = get_param("ichimoku_senkou", 52)
            chikou_lag = get_param("ichimoku_chikou_lag", 26)  # Default lag
            senkou_lag = get_param("ichimoku_senkou_lag", 26)  # Default lag
            if all(p > 0 for p in [tenkan, kijun, senkou, chikou_lag, senkou_lag]):
                # Ichimoku returns multiple DataFrames (lines and spans)
                ichi_lines, ichi_spans = df_out.ta.ichimoku(
                    tenkan=tenkan,
                    kijun=kijun,
                    senkou=senkou,
                    chikou_lag=chikou_lag,
                    senkou_lag=senkou_lag,
                    append=False,  # Get results to assign manually
                )
                if ichi_lines is not None:
                    df_out = pd.concat([df_out, ichi_lines], axis=1)
                if ichi_spans is not None:
                    # Spans are shifted, handle concatenation carefully
                    df_out = pd.concat(
                        [df_out, ichi_spans], axis=1
                    )  # Check if index aligns correctly

    except Exception as e:
        logger.error(f"Error calculating Ichimoku Cloud: {e}", exc_info=True)

    # --- Custom Indicators ---
    # Example: Ehlers Volumetric Trend (check if needed based on strategy config)
    strategy_config = config.get("strategy_params", {}).get(
        config.get("strategy", {}).get("name", "").lower(), {}
    )
    if config.get("strategy", {}).get(
        "name", ""
    ).lower() == "dual_ehlers_volumetric" or flags.get("use_evt"):
        try:
            # Primary EVT
            evt_len = strategy_config.get("evt_length", get_param("evt_length", 7))
            evt_mult = strategy_config.get(
                "evt_multiplier", get_param("evt_multiplier", 2.5)
            )
            df_out = ehlers_volumetric_trend(df_out, evt_len, float(evt_mult))
            # Rename columns for clarity if dual strategy
            if (
                config.get("strategy", {}).get("name", "").lower()
                == "dual_ehlers_volumetric"
            ):
                df_out = df_out.rename(
                    columns={
                        f"evt_trend_{evt_len}": "primary_trend",
                        f"evt_buy_{evt_len}": "primary_evt_buy",
                        f"evt_sell_{evt_len}": "primary_evt_sell",
                        f"vwma_{evt_len}": "primary_vwma",
                        f"smooth_vwma_{evt_len}": "primary_smooth_vwma",
                    }
                )

            # Confirmation EVT (if applicable for the strategy)
            if (
                config.get("strategy", {}).get("name", "").lower()
                == "dual_ehlers_volumetric"
            ):
                conf_evt_len = strategy_config.get(
                    "confirm_evt_length", get_param("confirm_evt_length", 5)
                )
                conf_evt_mult = strategy_config.get(
                    "confirm_evt_multiplier", get_param("confirm_evt_multiplier", 2.0)
                )
                df_out = ehlers_volumetric_trend(
                    df_out, conf_evt_len, float(conf_evt_mult)
                )
                df_out = df_out.rename(
                    columns={
                        f"evt_trend_{conf_evt_len}": "confirm_trend",
                        f"evt_buy_{conf_evt_len}": "confirm_evt_buy",
                        f"evt_sell_{conf_evt_len}": "confirm_evt_sell",
                        f"vwma_{conf_evt_len}": "confirm_vwma",
                        f"smooth_vwma_{conf_evt_len}": "confirm_smooth_vwma",
                    }
                )

        except Exception as e:
            logger.error(
                f"Error calculating Ehlers Volumetric Trend: {e}", exc_info=True
            )

    logger.debug(
        f"Finished calculating indicators. Final DataFrame shape: {df_out.shape}"
    )
    # logger.debug(f"Columns: {df_out.columns.to_list()}") # Optional: Log all columns for debugging
    return df_out


# --- Example Standalone Usage ---
if __name__ == "__main__":
    print("-" * 60)
    print("--- Indicator Module Demo ---")
    print("-" * 60)

    # Ensure logger level is appropriate for demo
    logger.setLevel(logging.DEBUG)

    # Dummy config for testing
    test_config = {
        "indicator_settings": {
            "min_data_periods": 50,  # Minimum rows needed
            "ema_short_period": 9,
            "ema_long_period": 21,
            "sma_short_period": 10,
            "sma_long_period": 50,
            "hma_length": 14,
            "vwap_length": None,  # Use standard VWAP calc if length is None in pandas_ta
            "rsi_period": 14,
            "stoch_rsi_period": 14,
            "stoch_k_period": 3,
            "stoch_d_period": 3,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bollinger_bands_period": 20,
            "bollinger_bands_std_dev": 2.0,
            "kc_length": 20,
            "kc_atr_length": 10,
            "kc_multiplier": 1.5,  # Keltner
            "atr_period": 14,
            "cci_period": 20,
            "williams_r_period": 14,
            "mfi_period": 14,
            "adx_period": 14,
            "psar_step": 0.02,
            "psar_max_step": 0.2,
            "volume_ma_period": 20,
            "momentum_period": 10,
            "roc_length": 12,
            "ao_fast": 5,
            "ao_slow": 34,  # Awesome Oscillator
            "ichimoku_tenkan": 9,
            "ichimoku_kijun": 26,
            "ichimoku_senkou": 52,
            "adosc_fast": 3,
            "adosc_slow": 10,
            "evt_length": 7,
            "evt_multiplier": 2.5,  # For Ehlers Volumetric
            "confirm_evt_length": 5,
            "confirm_evt_multiplier": 2.0,
        },
        "analysis_flags": {  # Enable calculation flags
            "use_sma": True,
            "use_ema": True,
            "use_hma": True,
            "use_vwap": True,
            "use_rsi": True,
            "use_stochrsi": True,
            "use_cci": True,
            "use_williams_r": True,
            "use_mfi": True,
            "use_momentum": True,
            "use_roc": True,
            "use_awesome_oscillator": True,
            "use_macd": True,
            "use_atr": True,
            "use_bollinger_bands": True,
            "use_keltner_channels": True,
            "use_adx": True,
            "use_psar": True,
            "use_obv": True,
            "use_adosc": True,
            "use_volume_ma": True,
            "use_ichimoku": True,
            "use_evt": True,  # Enable EVT calculation via flag
        },
        # "strategy": {"name": "DUAL_EHLERS_VOLUMETRIC"} # Alternatively, enable EVT via strategy name
        # Add threshold dict if interpretation relies on it
    }

    # Create dummy data
    periods = 200
    start_price = 55.0
    volatility = 0.01  # Daily volatility
    drift = 0.0001  # Slight upward drift
    # Generate more realistic price movements (Geometric Brownian Motion)
    returns = np.random.normal(loc=drift, scale=volatility, size=periods)
    prices = start_price * np.exp(np.cumsum(returns))

    data = {
        "timestamp": pd.date_range(
            start="2023-01-01", periods=periods, freq="H", tz="UTC"
        ),
        "open": prices[:-1],
        "close": prices[1:],
    }
    # Adjust length for open/close matching
    data["timestamp"] = data["timestamp"][1:]
    data["open"] = data["open"][: periods - 1]
    data["close"] = data["close"][: periods - 1]
    df_test = pd.DataFrame(data).set_index("timestamp")

    # Simulate High, Low, Volume
    df_test["high"] = df_test[["open", "close"]].max(axis=1) * (
        1 + np.random.uniform(0, 0.01, periods - 1)
    )
    df_test["low"] = df_test[["open", "close"]].min(axis=1) * (
        1 - np.random.uniform(0, 0.01, periods - 1)
    )
    # Ensure H >= max(O,C) and L <= min(O,C)
    df_test["high"] = np.maximum.reduce(
        [
            df_test["open"],
            df_test["close"],
            df_test["high"],
        ]
    )
    df_test["low"] = np.minimum.reduce(
        [
            df_test["open"],
            df_test["close"],
            df_test["low"],
        ]
    )
    df_test["volume"] = np.random.uniform(100, 2000, periods - 1) * (
        1 + 0.5 * np.abs(df_test["close"].pct_change().fillna(0))
    )  # Volume increases with price change

    print(f"Input DataFrame shape: {df_test.shape}")
    print(f"Input DataFrame head:\n{df_test.head()}")

    # Calculate all indicators based on test_config flags
    df_results = calculate_all_indicators(df_test, test_config)

    print("-" * 60)
    print(f"Output DataFrame shape: {df_results.shape}")
    print(f"Output DataFrame head with indicators:\n{df_results.head()}")
    print("-" * 60)
    print(f"Output DataFrame tail with indicators:\n{df_results.tail()}")
    print("-" * 60)
    print(f"Output columns ({len(df_results.columns)}): {df_results.columns.tolist()}")
    print("-" * 60)

    # Check for NaNs in the last row (indicators might need warmup period)
    last_row_nans = df_results.iloc[-1].isnull().sum()
    print(f"NaNs in last row: {last_row_nans} / {len(df_results.columns)}")
    if last_row_nans > 0:
        nan_cols_last_row = df_results.iloc[-1][
            df_results.iloc[-1].isnull()
        ].index.tolist()
        print(
            f"Columns with NaNs in last row ({len(nan_cols_last_row)}):\n{nan_cols_last_row}"
        )

    # --- Test Pivot Points and Levels ---
    print("-" * 60)
    print("--- Pivot Point & Levels Demo ---")
    # Use the last few rows of data for the calculation period
    period_df_for_levels = df_results.iloc[-25:]  # Use last 25 hours for levels calc
    current_price_example = df_results["close"].iloc[-1]
    print(
        f"Calculating levels based on last {len(period_df_for_levels)} periods, current price: {current_price_example:.4f}"
    )

    levels_result = calculate_levels(period_df_for_levels, current_price_example)

    print(f"Pivot Point: {levels_result.get('pivot')}")
    print("\nStandard Pivots:")
    for k, v in levels_result.get("standard_pivots", {}).items():
        print(f"  {k}: {v:.4f}")
    print("\nFibonacci Pivots:")
    for k, v in levels_result.get("fib_pivots", {}).items():
        print(f"  {k}: {v:.4f}")
    print("\nFibonacci Retracements:")
    for k, v in levels_result.get("fib_retracements", {}).items():
        print(f"  {k}: {v:.4f}")
    print("\nSupport Levels:")
    for k, v in levels_result.get("support", {}).items():
        print(f"  {k}: {v:.4f}")
    print("\nResistance Levels:")
    for k, v in levels_result.get("resistance", {}).items():
        print(f"  {k}: {v:.4f}")
    print("-" * 60)
