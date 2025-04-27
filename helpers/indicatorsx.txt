
# indicators.py - Placeholder for Technical Indicator Functions
# Implement your indicator logic here.

import numpy as np  # Often needed for calculations
import pandas as pd


def rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    if not isinstance(close_prices, pd.Series):
        raise TypeError("close_prices must be a pandas Series")
    if close_prices.isnull().any():
        # Handle NaNs if necessary, e.g., dropna or fillna, or return NaNs
        # For simplicity, we'll proceed, but calculations might yield NaNs
        pass
    if len(close_prices) < period:
        return pd.Series([np.nan] * len(close_prices), index=close_prices.index)  # Not enough data

    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0.0).fillna(0)
    loss = -delta.where(delta < 0, 0.0).fillna(0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use Exponential Moving Average (EMA) for subsequent calculations - more common for RSI
    # For the first value, use the simple mean calculated above
    # Subsequent values use EMA formula: EMA = (Current Value * alpha) + (Previous EMA * (1 - alpha))
    # alpha = 1 / period
    alpha = 1.0 / period
    for i in range(period, len(close_prices)):
         avg_gain[i] = (gain[i] * alpha) + (avg_gain[i - 1] * (1 - alpha))
         avg_loss[i] = (loss[i] * alpha) + (avg_loss[i - 1] * (1 - alpha))

    rs = avg_gain / avg_loss
    rsi_values = 100.0 - (100.0 / (1.0 + rs))
    rsi_values.fillna(50, inplace=True)  # Fill initial NaNs, often RSI starts around 50 conceptually
    # Ensure the final series has the same index as the input
    return rsi_values.reindex(close_prices.index)


def ATR(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Average True Range (ATR)."""
    if not all(isinstance(s, pd.Series) for s in [high_prices, low_prices, close_prices]):
        raise TypeError("Inputs must be pandas Series")
    if high_prices.isnull().any() or low_prices.isnull().any() or close_prices.isnull().any():
        # Handle NaNs if necessary
        pass
    if len(high_prices) < period:
         return pd.Series([np.nan] * len(high_prices), index=high_prices.index)  # Not enough data

    high_low = high_prices - low_prices
    high_close_prev = abs(high_prices - close_prices.shift(1))
    low_close_prev = abs(low_prices - close_prices.shift(1))

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    # Use Simple Moving Average (SMA) for ATR calculation typically
    atr_values = tr.rolling(window=period, min_periods=period).mean()

    # Ensure the final series has the same index as the input
    return atr_values.reindex(high_prices.index)


def FibonacciPivotPoints(high: float, low: float, close: float) -> dict:
    """Calculates Fibonacci Pivot Points."""
    if not all(isinstance(v, (int, float)) for v in [high, low, close]):
         raise TypeError("Inputs must be numeric (int or float)")

    pivot = (high + low + close) / 3.0
    fib_range = high - low

    pivots = {
        'Pivot': pivot,
        'R1': pivot + (0.382 * fib_range),
        'R2': pivot + (0.618 * fib_range),
        'R3': pivot + (1.000 * fib_range),
        'S1': pivot - (0.382 * fib_range),
        'S2': pivot - (0.618 * fib_range),
        'S3': pivot - (1.000 * fib_range)
    }
    return pivots
# --- START OF FILE indicators_v1.1.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical Indicators Module (v1.1) for Trading Bots

Provides functions to calculate various technical indicators using the pandas_ta library
and some custom calculations where needed. Designed to work with pandas DataFrames
containing OHLCV data and return results compatible with Decimal precision if desired.

New in v1.1:
- Hull Moving Average (HMA)
- Volume Weighted Average Price (VWAP - rolling approximation)
- Fibonacci Pivot Points
- Ichimoku Cloud
- Keltner Channels
- Awesome Oscillator (AO)
- Rate of Change (ROC)
- Integrated into master calculation function based on config flags.
- Added basic interpretation logic placeholders.

Includes:
- SMA, EMA, HMA, VWAP
- RSI, Stochastic RSI, CCI, Williams %R, MFI, Momentum, ROC, AO
- MACD
- Bollinger Bands, Keltner Channels
- ATR
- ADX
- OBV, ADOSC
- PSAR
- Ehlers Volumetric Trend
- Pivot Points (Standard & Fibonacci)
- Ichimoku Cloud

Assumes input DataFrame has columns: 'timestamp', 'open', 'high', 'low', 'close', 'volume'
with a datetime index (preferably UTC).
"""

import logging
import os
import sys # Added for potential standalone use
from decimal import Decimal, InvalidOperation, getcontext
from typing import Optional, Dict, Any, Callable, Union, Tuple, List

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import]
from colorama import Fore, Style # For potential logging within functions

# --- Setup ---
# Use logger from the main script or initialize a basic one
logger = logging.getLogger(__name__)
# Set high precision for Decimal calculations if needed internally
getcontext().prec = 28

# --- Helper for Safe Decimal Conversion ---
# Assume this exists in the main script or a shared utility module
# If not, define it here:
def safe_decimal_conversion(value: Any, default: Any = Decimal("0.0")) -> Decimal | Any:
    if value is None: return default
    try:
        if isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
             if np.isnan(value): return default if isinstance(default, Decimal) else None
             value = str(value)
        elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
             value = str(value)
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        # logger.warning(f"Could not convert '{value}' to Decimal, using default {default}") # Too verbose
        return default if isinstance(default, Decimal) else None

# --- Standard Indicator Calculation Functions ---

def calculate_sma(series: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Simple Moving Average."""
    if series is None or series.empty or length <= 0 or len(series) < length: return None
    try: return ta.sma(series, length=length)
    except Exception as e: logger.error(f"Error calculating SMA(length={length}): {e}"); return None

def calculate_ema(series: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Exponential Moving Average."""
    if series is None or series.empty or length <= 0 or len(series) < length: return None
    try: return ta.ema(series, length=length, adjust=False)
    except Exception as e: logger.error(f"Error calculating EMA(length={length}): {e}"); return None

def calculate_hma(series: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Hull Moving Average."""
    if series is None or series.empty or length <= 0 or len(series) < length: return None
    try: return ta.hma(series, length=length)
    except Exception as e: logger.error(f"Error calculating HMA(length={length}): {e}"); return None

def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: Optional[int] = None) -> Optional[pd.Series]:
    """
    Calculates Volume Weighted Average Price.
    Note: Standard VWAP resets daily/session. pandas_ta provides a rolling version if length is specified.
    If length is None, it attempts typical price * volume / volume sum (less common).
    """
    if high is None or low is None or close is None or volume is None or \
       high.empty or low.empty or close.empty or volume.empty: return None
    try: return ta.vwap(high=high, low=low, close=close, volume=volume, length=length) # Use rolling if length given
    except Exception as e: logger.error(f"Error calculating VWAP(length={length}): {e}"); return None


def calculate_rsi(series: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Relative Strength Index."""
    if series is None or series.empty or length <= 0 or len(series) < length + 1: return None
    try: return ta.rsi(series, length=length)
    except Exception as e: logger.error(f"Error calculating RSI(length={length}): {e}"); return None

def calculate_stochrsi(series: pd.Series, length: int, rsi_length: int, k: int, d: int) -> Optional[pd.DataFrame]:
    """Calculates Stochastic RSI (K and D lines)."""
    min_len = rsi_length + length + max(k, d) # Rough estimate
    if series is None or series.empty or len(series) < min_len: return None
    try: return ta.stochrsi(series, length=length, rsi_length=rsi_length, k=k, d=d)
    except Exception as e: logger.error(f"Error calculating StochRSI(l={length}, rsi_l={rsi_length}, k={k}, d={d}): {e}"); return None

def calculate_macd(series: pd.Series, fast: int, slow: int, signal: int) -> Optional[pd.DataFrame]:
    """Calculates MACD, Signal line, and Histogram."""
    if series is None or series.empty or fast <= 0 or slow <= 0 or signal <= 0 or fast >= slow or len(series) < slow + signal: return None
    try: return ta.macd(series, fast=fast, slow=slow, signal=signal)
    except Exception as e: logger.error(f"Error calculating MACD(f={fast}, s={slow}, sig={signal}): {e}"); return None

def calculate_bollinger_bands(series: pd.Series, length: int, std: float | Decimal) -> Optional[pd.DataFrame]:
    """Calculates Bollinger Bands (Upper, Middle, Lower)."""
    if series is None or series.empty or length <= 0 or float(std) <= 0 or len(series) < length: return None
    try: return ta.bbands(series, length=length, std=float(std))
    except Exception as e: logger.error(f"Error calculating BBands(length={length}, std={std}): {e}"); return None

def calculate_keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, length: int, atr_length: Optional[int] = None, multiplier: float = 2.0) -> Optional[pd.DataFrame]:
    """Calculates Keltner Channels."""
    atr_len = atr_length if atr_length is not None else length # Default ATR length to KC length if not specified
    if high is None or low is None or close is None or high.empty or low.empty or close.empty or length <= 0 or atr_len <= 0 or len(close) < max(length, atr_len): return None
    try: return ta.kc(high=high, low=low, close=close, length=length, atr_length=atr_len, mamode="EMA", multiplier=multiplier) # Use EMA for center line typically
    except Exception as e: logger.error(f"Error calculating Keltner Channels(l={length}, atr_l={atr_len}): {e}"); return None

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Average True Range."""
    required_len = length + 1
    if high is None or low is None or close is None or high.empty or low.empty or close.empty or length <= 0 or \
       len(high) < required_len or len(low) < required_len or len(close) < required_len: return None
    try: return ta.atr(high=high, low=low, close=close, length=length)
    except Exception as e: logger.error(f"Error calculating ATR(length={length}): {e}"); return None

def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Commodity Channel Index."""
    if high is None or low is None or close is None or high.empty or low.empty or close.empty or length <= 0 or \
       len(high) < length or len(low) < length or len(close) < length: return None
    try: return ta.cci(high=high, low=low, close=close, length=length)
    except Exception as e: logger.error(f"Error calculating CCI(length={length}): {e}"); return None

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Williams %R."""
    if high is None or low is None or close is None or high.empty or low.empty or close.empty or length <= 0 or \
       len(high) < length or len(low) < length or len(close) < length: return None
    try: return ta.willr(high=high, low=low, close=close, length=length)
    except Exception as e: logger.error(f"Error calculating Williams %R(length={length}): {e}"); return None

def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Money Flow Index."""
    if high is None or low is None or close is None or volume is None or \
       high.empty or low.empty or close.empty or volume.empty or length <= 0 or \
       len(high) < length or len(low) < length or len(close) < length or len(volume) < length: return None
    try: return ta.mfi(high=high, low=low, close=close, volume=volume, length=length)
    except Exception as e: logger.error(f"Error calculating MFI(length={length}): {e}"); return None

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> Optional[pd.DataFrame]:
    """Calculates Average Directional Index (ADX, +DI, -DI)."""
    required_len = length * 2
    if high is None or low is None or close is None or high.empty or low.empty or close.empty or length <= 0 or \
       len(high) < required_len or len(low) < required_len or len(close) < required_len: return None
    try: return ta.adx(high=high, low=low, close=close, length=length)
    except Exception as e: logger.error(f"Error calculating ADX(length={length}): {e}"); return None

def calculate_obv(close: pd.Series, volume: pd.Series) -> Optional[pd.Series]:
    """Calculates On Balance Volume."""
    if close is None or volume is None or close.empty or volume.empty or len(close) != len(volume): return None
    try: return ta.obv(close=close, volume=volume)
    except Exception as e: logger.error(f"Error calculating OBV: {e}"); return None

def calculate_adosc(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, fast: int, slow: int) -> Optional[pd.Series]:
    """Calculates Accumulation/Distribution Oscillator."""
    required_len = max(fast, slow)
    if high is None or low is None or close is None or volume is None or \
       high.empty or low.empty or close.empty or volume.empty or fast <= 0 or slow <= 0 or \
       len(high) < required_len or len(low) < required_len or len(close) < required_len or len(volume) < required_len: return None
    try: return ta.adosc(high=high, low=low, close=close, volume=volume, fast=fast, slow=slow)
    except Exception as e: logger.error(f"Error calculating ADOSC(fast={fast}, slow={slow}): {e}"); return None

def calculate_psar(high: pd.Series, low: pd.Series, step: float, max_step: float) -> Optional[pd.DataFrame]:
    """Calculates Parabolic Stop and Reverse (PSAR)."""
    if high is None or low is None or high.empty or low.empty or step <= 0 or max_step <= step: return None
    try: return ta.psar(high=high, low=low, af=step, max_af=max_step)
    except Exception as e: logger.error(f"Error calculating PSAR(step={step}, max={max_step}): {e}"); return None

def calculate_roc(series: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Rate of Change."""
    if series is None or series.empty or length <= 0 or len(series) < length + 1: return None
    try: return ta.roc(series, length=length)
    except Exception as e: logger.error(f"Error calculating ROC(length={length}): {e}"); return None

def calculate_awesome_oscillator(high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34) -> Optional[pd.Series]:
    """Calculates Bill Williams' Awesome Oscillator (AO)."""
    required_len = slow
    if high is None or low is None or high.empty or low.empty or fast <= 0 or slow <= 0 or fast >= slow or \
       len(high) < required_len or len(low) < required_len: return None
    try: return ta.ao(high=high, low=low, fast=fast, slow=slow)
    except Exception as e: logger.error(f"Error calculating Awesome Oscillator(fast={fast}, slow={slow}): {e}"); return None

def calculate_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Calculates Ichimoku Cloud components.
    Returns two DataFrames: one for Kinko Hyo lines, one for Cloud spans.
    """
    required_len = max(tenkan, kijun, senkou) + kijun # Needs lookback for spans
    if high is None or low is None or close is None or high.empty or low.empty or close.empty or \
       len(high) < required_len or len(low) < required_len or len(close) < required_len: return None, None
    try:
        # pandas_ta returns multiple dataframes or series depending on version/params
        # We request specific components: tenkan, kijun, senkou spans, chikou
        ichi_df, span_df = ta.ichimoku(high=high, low=low, close=close, tenkan=tenkan, kijun=kijun, senkou=senkou, include_chikou=True, append=False)
        return ichi_df, span_df
    except Exception as e: logger.error(f"Error calculating Ichimoku(t={tenkan}, k={kijun}, s={senkou}): {e}"); return None, None


# --- Custom / Combined Indicator Functions ---

def ehlers_volumetric_trend(df: pd.DataFrame, length: int, multiplier: Decimal | float) -> pd.DataFrame:
    """
    Calculate Ehlers Volumetric Trend indicator using VWMA and SuperSmoother.
    Adds 'vwma', 'smoothed_vwma', 'trend' (1/-1/0), 'evt_buy', 'evt_sell' (boolean) columns.
    """
    # (Implementation remains the same as previous version)
    if not all(col in df.columns for col in ['close', 'volume']) or df.empty or length <= 0:
        logger.warning("EVT calculation skipped: Missing columns, empty df, or invalid length.")
        return df # Return original df
    try:
        vwma = calculate_vwma(df['close'], df['volume'], length=length)
        if vwma is None or vwma.isnull().all(): raise ValueError(f"VWMA(length={length}) failed")
        df[f'vwma_{length}'] = vwma
        a=np.exp(-1.414*np.pi/length); b=2*a*np.cos(1.414*np.pi/length); c2=b; c3=-a*a; c1=1-c2-c3
        smoothed=np.zeros(len(df)); vwma_vals=df[f'vwma_{length}'].values
        for i in range(2,len(df)):
            if not np.isnan(vwma_vals[i]):
                sm1=smoothed[i-1] if not np.isnan(smoothed[i-1]) else 0; sm2=smoothed[i-2] if not np.isnan(smoothed[i-2]) else 0
                smoothed[i]=c1*vwma_vals[i]+c2*sm1+c3*sm2
            else: smoothed[i]=smoothed[i-1] if i>0 and not np.isnan(smoothed[i-1]) else 0
        df[f'smooth_vwma_{length}']=smoothed
        trend=np.zeros(len(df),dtype=int); mult_h=1+float(multiplier)/100.0; mult_l=1-float(multiplier)/100.0; shifted_smooth=df[f'smooth_vwma_{length}'].shift(1).values; valid=~np.isnan(smoothed) & ~np.isnan(shifted_smooth)
        trend[valid]=np.where(smoothed[valid]>shifted_smooth[valid]*mult_h,1,trend[valid])
        trend[valid]=np.where(smoothed[valid]<shifted_smooth[valid]*mult_l,-1,trend[valid])
        trend_col_name = f'evt_trend_{length}'; buy_col = f'evt_buy_{length}'; sell_col = f'evt_sell_{length}'
        df[trend_col_name]=trend; df[trend_col_name]=df[trend_col_name].ffill().fillna(0).astype(int)
        trend_shifted=df[trend_col_name].shift(1); df[buy_col]=(df[trend_col_name]==1)&(trend_shifted!=1); df[sell_col]=(df[trend_col_name]==-1)&(trend_shifted!=-1)
        logger.debug(f"Ehlers Volumetric Trend (len={length}, mult={multiplier}) calculated.")
    except Exception as e: logger.error(f"{Fore.RED}Error in EVT(len={length}): {e}{Style.RESET_ALL}",exc_info=True); df[[f'vwma_{length}',f'smooth_vwma_{length}',f'evt_trend_{length}',f'evt_buy_{length}',f'evt_sell_{length}']]=np.nan
    return df


# --- Pivot Point Calculations ---

def calculate_standard_pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, Decimal]:
    """Calculates standard pivot points for the *next* period based on *previous* period's HLC."""
    pivots = {}
    # Ensure we have data for the previous period
    if len(high) < 1 or len(low) < 1 or len(close) < 1: return pivots
    try:
        # Use the most recent available HLC data (typically previous day's or previous candle's)
        H = Decimal(str(high.iloc[-1]))
        L = Decimal(str(low.iloc[-1]))
        C = Decimal(str(close.iloc[-1]))

        Pivot = (H + L + C) / 3
        pivots['PP'] = Pivot
        pivots['S1'] = (2 * Pivot) - H
        pivots['R1'] = (2 * Pivot) - L
        pivots['S2'] = Pivot - (H - L)
        pivots['R2'] = Pivot + (H - L)
        pivots['S3'] = L - 2 * (H - Pivot)
        pivots['R3'] = H + 2 * (Pivot - L)
    except (IndexError, InvalidOperation, TypeError) as e:
        logger.error(f"Error calculating standard pivot points: {e}")
    return pivots

def calculate_fib_pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, Decimal]:
    """Calculates Fibonacci pivot points for the *next* period."""
    fib_pivots = {}
    if len(high) < 1 or len(low) < 1 or len(close) < 1: return fib_pivots
    try:
        H = Decimal(str(high.iloc[-1]))
        L = Decimal(str(low.iloc[-1]))
        C = Decimal(str(close.iloc[-1]))
        Range = H - L
        if Range == 0: Range = Decimal('1e-8') # Avoid zero division

        Pivot = (H + L + C) / 3
        fib_pivots['PP'] = Pivot
        fib_pivots['S1'] = Pivot - (Decimal("0.382") * Range)
        fib_pivots['R1'] = Pivot + (Decimal("0.382") * Range)
        fib_pivots['S2'] = Pivot - (Decimal("0.618") * Range)
        fib_pivots['R2'] = Pivot + (Decimal("0.618") * Range)
        fib_pivots['S3'] = Pivot - (Decimal("1.000") * Range) # = Low - (H - Pivot)? Sometimes defined differently
        fib_pivots['R3'] = Pivot + (Decimal("1.000") * Range) # = High + (Pivot - Low)?

    except (IndexError, InvalidOperation, TypeError, DivisionByZero) as e:
        logger.error(f"Error calculating Fibonacci pivot points: {e}")
    return fib_pivots

def calculate_levels(df: pd.DataFrame, current_price: Decimal) -> dict:
    """Calculates various support/resistance levels."""
    levels = {"support": {}, "resistance": {}, "pivot": None, "fib_pivots": {}, "standard_pivots": {}}
    if df.empty: return levels

    try:
        # Ensure calculations use data up to the *previous* candle for pivots
        # For rolling indicators like Fib retracement, using current range is ok
        high_period = df["high"].max()
        low_period = df["low"].min()
        # For pivots, typically use previous period's HLC (e.g., previous day, previous candle)
        prev_high = df["high"].iloc[-2] if len(df) >= 2 else df["high"].iloc[-1]
        prev_low = df["low"].iloc[-2] if len(df) >= 2 else df["low"].iloc[-1]
        prev_close = df["close"].iloc[-2] if len(df) >= 2 else df["close"].iloc[-1]

        # Fibonacci Retracement (based on current period's high/low)
        diff = high_period - low_period
        if diff > 1e-9:
            fib_levels = {
                "Fib 23.6%": high_period - diff * Decimal("0.236"), "Fib 38.2%": high_period - diff * Decimal("0.382"),
                "Fib 50.0%": high_period - diff * Decimal("0.5"), "Fib 61.8%": high_period - diff * Decimal("0.618"),
                "Fib 78.6%": high_period - diff * Decimal("0.786"),
            }
            for label, value in fib_levels.items():
                if value < current_price: levels["support"][label] = value
                else: levels["resistance"][label] = value

        # Standard Pivot Points (based on *previous* candle's HLC)
        standard_pivots = calculate_standard_pivot_points(df['high'].shift(1), df['low'].shift(1), df['close'].shift(1))
        if standard_pivots:
            levels["standard_pivots"] = standard_pivots
            pivot = standard_pivots.get('PP')
            if pivot: levels["pivot"] = pivot # Store main pivot separately
            for label, value in standard_pivots.items():
                 if value < current_price: levels["support"][label] = value
                 else: levels["resistance"][label] = value

        # Fibonacci Pivot Points (based on *previous* candle's HLC)
        fib_pivots = calculate_fib_pivot_points(df['high'].shift(1), df['low'].shift(1), df['close'].shift(1))
        if fib_pivots:
             levels["fib_pivots"] = fib_pivots
             # Add Fib pivots to S/R levels too
             for label, value in fib_pivots.items():
                 if label != 'PP': # Avoid duplicating pivot point
                    if value < current_price: levels["support"][f"Fib {label}"] = value
                    else: levels["resistance"][f"Fib {label}"] = value

    except Exception as e:
        logger.exception(f"Unexpected error calculating S/R levels: {e}")

    return levels


# --- Master Indicator Calculation Function ---

def calculate_all_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculates all enabled indicators based on the provided configuration
    and adds them as columns to the DataFrame.

    Args:
        df: Input DataFrame with 'open', 'high', 'low', 'close', 'volume'.
            Assumes DataFrame index is datetime.
        config: Configuration dictionary, expected to have 'indicator_settings'
                and 'analysis_flags' keys.

    Returns:
        DataFrame with calculated indicator columns added. Original df is modified
        if indicators are calculated successfully, otherwise original df is returned.
    """
    if df is None or df.empty:
        logger.error("Cannot calculate indicators on empty DataFrame.")
        return pd.DataFrame() # Return empty DF

    # Ensure required base columns exist
    required_cols = ['high', 'low', 'close', 'volume', 'open']
    if not all(col in df.columns for col in required_cols):
         missing = [col for col in required_cols if col not in df.columns]
         logger.error(f"Input DataFrame missing required columns for indicator calculation: {missing}")
         return df # Return original df if columns missing

    df_out = df # Work directly on the DataFrame for efficiency with pandas_ta
    settings = config.get("indicator_settings", {})
    flags = config.get("analysis_flags", {})
    min_rows_needed = 50 # Default minimum rows for reliable calculation of most indicators

    # Check length after dropping potential initial NaNs from data fetching
    df_clean = df_out.dropna(subset=required_cols)
    if len(df_clean) < min_rows_needed:
        logger.warning(f"Insufficient valid rows ({len(df_clean)}) after cleaning NaNs. Need at least {min_rows_needed}. Indicator results may be unreliable or NaN.")
        # Proceed but be aware calculations might fail or be inaccurate


    # --- Define a helper to run TA-Lib functions safely ---
    def apply_ta_indicator(func: Callable, name: str, required_flag: Optional[str] = None, **kwargs):
        """Applies a pandas_ta indicator if flag is true and inputs are valid."""
        if required_flag and not flags.get(required_flag, False):
            # logger.debug(f"Indicator '{name}' disabled by flag '{required_flag}'.")
            return # Skip if flag is false

        # Prepare args, ensuring required series exist
        series_args = {k: df_out[k] for k in ['high', 'low', 'close', 'volume', 'open'] if k in kwargs.get('required_inputs', []) and k in df_out}
        if len(series_args) != len(kwargs.get('required_inputs', [])):
             logger.warning(f"Missing required columns for indicator '{name}'. Skipping.")
             return

        # Combine series args with length/param args
        final_kwargs = {**series_args, **kwargs}
        # Remove internal 'required_inputs' key before passing to ta function
        final_kwargs.pop('required_inputs', None)

        try:
             # Most ta functions append columns directly to the DataFrame
             logger.debug(f"Calculating {name} with params: {kwargs}")
             df_out.ta(kind=name.lower(), **final_kwargs, append=True)
        except AttributeError: # Handle cases where ta doesn't have the indicator or params are wrong
             logger.error(f"{Fore.RED}Pandas TA does not have indicator '{name}' or parameters are incorrect: {kwargs}{Style.RESET_ALL}")
        except Exception as e:
             logger.error(f"{Fore.RED}Error calculating indicator '{name}' with params {kwargs}: {e}{Style.RESET_ALL}")


    # --- Apply Indicators based on Flags ---
    # MAs
    apply_ta_indicator(ta.sma, "SMA", required_flag="ema_alignment", length=settings.get('sma_short_period', 10), required_inputs=['close'])
    apply_ta_indicator(ta.sma, "SMA", required_flag="ema_alignment", length=settings.get('sma_long_period', 50), required_inputs=['close'])
    apply_ta_indicator(ta.ema, "EMA", required_flag="ema_alignment", length=settings.get('ema_short_period', 12), adjust=False, required_inputs=['close'])
    apply_ta_indicator(ta.ema, "EMA", required_flag="ema_alignment", length=settings.get('ema_long_period', 26), adjust=False, required_inputs=['close'])
    apply_ta_indicator(ta.hma, "HMA", length=settings.get('hma_length', 9), required_inputs=['close']) # Add HMA flag if needed
    apply_ta_indicator(ta.vwap, "VWAP", length=settings.get('vwap_length'), required_inputs=['high', 'low', 'close', 'volume']) # Add VWAP flag if needed

    # Oscillators
    apply_ta_indicator(ta.rsi, "RSI", required_flag="rsi_threshold", length=settings.get('rsi_period', 14), required_inputs=['close'])
    apply_ta_indicator(ta.stochrsi, "StochRSI", required_flag="stoch_rsi_cross", length=settings.get('stoch_rsi_period', 14), rsi_length=settings.get('rsi_period', 14), k=settings.get('stoch_k_period', 3), d=settings.get('stoch_d_period', 3), required_inputs=['close'])
    apply_ta_indicator(ta.cci, "CCI", required_flag="cci_threshold", length=settings.get('cci_period', 20), required_inputs=['high', 'low', 'close'])
    apply_ta_indicator(ta.willr, "WILLR", required_flag="williams_r_threshold", length=settings.get('williams_r_period', 14), required_inputs=['high', 'low', 'close'])
    apply_ta_indicator(ta.mfi, "MFI", required_flag="mfi_threshold", length=settings.get('mfi_period', 14), required_inputs=['high', 'low', 'close', 'volume'])
    apply_ta_indicator(ta.mom, "Momentum", required_flag="momentum_crossover", length=settings.get('momentum_period', 10), required_inputs=['close'])
    apply_ta_indicator(ta.ao, "AwesomeOscillator", fast=5, slow=34, required_inputs=['high', 'low']) # Add AO flag if needed
    apply_ta_indicator(ta.roc, "ROC", length=settings.get('roc_length', 10), required_inputs=['close']) # Add ROC flag if needed

    # MACD
    apply_ta_indicator(ta.macd, "MACD", required_flag="macd_cross", fast=settings.get('macd_fast', 12), slow=settings.get('macd_slow', 26), signal=settings.get('macd_signal', 9), required_inputs=['close'])

    # Volatility / Bands / Channels
    apply_ta_indicator(ta.atr, "ATR", required_flag=None, length=settings.get('atr_period', 14), required_inputs=['high', 'low', 'close']) # Always calc ATR
    apply_ta_indicator(ta.bbands, "BollingerBands", required_flag="bollinger_bands_break", length=settings.get('bollinger_bands_period', 20), std=float(settings.get('bollinger_bands_std_dev', 2.0)), required_inputs=['close'])
    apply_ta_indicator(ta.kc, "KeltnerChannels", length=settings.get('kc_length', 20), atr_length=settings.get('kc_atr_length', 10), multiplier=float(settings.get('kc_multiplier', 2.0)), required_inputs=['high', 'low', 'close']) # Add KC flag if needed

    # Trend
    apply_ta_indicator(ta.adx, "ADX", required_flag="adx_trend_strength", length=settings.get('adx_period', 14), required_inputs=['high', 'low', 'close'])
    apply_ta_indicator(ta.psar, "PSAR", required_flag="psar_flip", step=settings.get('psar_step', 0.02), max_step=settings.get('psar_max_step', 0.2), required_inputs=['high', 'low'])

    # Volume Based
    apply_ta_indicator(ta.obv, "OBV", required_flag="obv_trend", required_inputs=['close', 'volume'])
    apply_ta_indicator(ta.adosc, "ADOSC", required_flag="adi_trend", fast=settings.get('macd_fast', 12), slow=settings.get('macd_slow', 26), required_inputs=['high', 'low', 'close', 'volume'])
    # Volume MA (simple rolling mean)
    vol_ma_len = settings.get('volume_ma_period', 20)
    if flags.get("volume_confirmation") and vol_ma_len > 0:
        sma_vol = get_indicator(calculate_sma, df_out['volume'], length=vol_ma_len)
        if sma_vol is not None: df_out[f'volume_ma_{vol_ma_len}'] = sma_vol

    # Custom: Ehlers Volumetric Trend
    if config.get('strategy',{}).get('name','').upper() == "DUAL_EHLERS_VOLUMETRIC":
         evt_len = config.get('strategy_params',{}).get('dual_ehlers_volumetric',{}).get('evt_length', 7)
         evt_mult = config.get('strategy_params',{}).get('dual_ehlers_volumetric',{}).get('evt_multiplier', Decimal('2.5'))
         conf_evt_len = config.get('strategy_params',{}).get('dual_ehlers_volumetric',{}).get('confirm_evt_length', 5)
         conf_evt_mult = config.get('strategy_params',{}).get('dual_ehlers_volumetric',{}).get('confirm_evt_multiplier', Decimal('2.0'))
         df_out = ehlers_volumetric_trend(df_out, evt_len, evt_mult)
         df_out = ehlers_volumetric_trend(df_out, conf_evt_len, conf_evt_mult)
         # Rename columns to avoid conflicts and be clear
         df_out = df_out.rename(columns={
             f'evt_trend_{evt_len}': 'primary_trend', f'evt_buy_{evt_len}': 'primary_evt_buy', f'evt_sell_{evt_len}': 'primary_evt_sell',
             f'evt_trend_{conf_evt_len}': 'confirm_trend', f'evt_buy_{conf_evt_len}': 'confirm_evt_buy', f'evt_sell_{conf_evt_len}': 'confirm_evt_sell'
         })


    # Convert final results to Decimal if desired (optional, adds overhead)
    # for col in df_out.columns:
    #     if df_out[col].dtype == 'float64':
    #         df_out[col] = df_out[col].apply(lambda x: safe_decimal_conversion(x, default=Decimal('NaN')))

    logger.debug(f"Finished calculating indicators. Final DataFrame shape: {df_out.shape}")
    return df_out


# --- Example Standalone Usage ---
if __name__ == "__main__":
    print("-" * 60)
    print("--- Indicator Module v1.1 Demo ---")
    print("-" * 60)

    # Basic logger for testing this module
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    # Dummy config for testing
    test_config = {
        "indicator_settings": {
            "ema_short_period": 9, "ema_long_period": 21,
            "sma_short_period": 10, "sma_long_period": 50,
            "hma_length": 14, "vwap_length": 20, # Rolling VWAP
            "rsi_period": 14, "stoch_rsi_period": 14, "stoch_k_period": 3, "stoch_d_period": 3,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "bollinger_bands_period": 20, "bollinger_bands_std_dev": 2.0,
            "kc_length": 20, "kc_atr_length": 10, "kc_multiplier": 1.5, # Keltner
            "atr_period": 14, "cci_period": 20, "williams_r_period": 14, "mfi_period": 14,
            "adx_period": 14, "psar_step": 0.02, "psar_max_step": 0.2,
            "volume_ma_period": 20, "momentum_period": 10, "roc_length": 12,
            "ao_fast": 5, "ao_slow": 34, # Awesome Oscillator
            "ichimoku_tenkan": 9, "ichimoku_kijun": 26, "ichimoku_senkou": 52, # Ichimoku
            "evt_length": 7, "evt_multiplier": 2.5, # For Ehlers Volumetric
            "confirm_evt_length": 5, "confirm_evt_multiplier": 2.0,
        },
        "analysis_flags": { # Enable all calculations for testing
            "ema_alignment": True, "momentum_crossover": True, "volume_confirmation": True,
            "rsi_divergence": True, "macd_divergence": True, "stoch_rsi_cross": True,
            "rsi_threshold": True, "mfi_threshold": True, "cci_threshold": True,
            "williams_r_threshold": True, "macd_cross": True, "bollinger_bands_break": True,
            "keltner_channels": True, "adx_trend_strength": True, "obv_trend": True,
            "adi_trend": True, "psar_flip": True, "roc": True, "awesome_oscillator": True,
            "ichimoku": True
        },
        "strategy": {"name": "DUAL_EHLERS_VOLUMETRIC"} # Example strategy to test EVT calc
        # Add threshold dict if interpretation relies on it
    }

    # Create dummy data
    periods = 200
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=periods, freq='H', tz='UTC'),
        'open': np.random.uniform(50, 60, periods),
        'high': np.random.uniform(55, 65, periods),
        'low': np.random.uniform(45, 55, periods),
        'close': np.random.uniform(50, 60, periods),
        'volume': np.random.uniform(100, 1100, periods)
    }
    # Ensure HLOC consistency
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    df_test = pd.DataFrame(data).set_index('timestamp')

    print(f"Input DataFrame head:\n{df_test.head()}")

    # Calculate all indicators based on test_config flags
    df_results = calculate_all_indicators(df_test, test_config)

    print("-" * 60)
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
        print(f"Columns with NaNs in last row:\n{df_results.iloc[-1][df_results.iloc[-1].isnull()].index.tolist()}")

# --- END OF FILE indicators_v1.1.py ---