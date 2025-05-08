#!/usr/bin/env python

"""
Technical Indicators Module (v3.0 - Full Trend Analysis Suite)

Comprehensive module for calculating technical indicators, with a focus on trend analysis tools.
Designed for integration with Bybit Trading Enchanced in a Termux environment. Leverages pandas_ta
for efficiency and includes custom implementations for specialized indicators.

Key Features:
- Trend Analysis: SMA, EMA, WMA, HMA, DEMA, TEMA, ADX, PSAR, Supertrend, Ichimoku Cloud, MACD, AO, KST, Keltner Channels, Donchian Channels.
- Pivot Points: Standard and Fibonacci-based, with support/resistance levels.
- Custom Indicators: VWMA, Ehlers Volumetric Trend (EVT).
- Incremental updates for WebSocket-driven real-time data.
- Configuration compatible with AppConfig (bybit_trading_enchanced.py).
- Robust error handling and logging aligned with BybitHelper.

Assumes input DataFrame has columns: 'open', 'high', 'low', 'close', 'volume' and a datetime index (UTC).
"""

import logging
import sys
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    print("Error: pandas_ta library not found. Install: pip install pandas_ta", file=sys.stderr)
    sys.exit(1)

# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Constants ---
MIN_PERIODS_DEFAULT = 50


# --- Pivot Point Calculations ---
def calculate_standard_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculate standard pivot points for the next period."""
    if not all(isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]):
        logger.warning("Invalid input for standard pivot points.")
        return {}
    if low > high:
        logger.warning(f"Low ({low}) > High ({high}) for pivot calculation.")
    try:
        pivot = (high + low + close) / 3.0
        return {
            "PP": pivot,
            "S1": (2 * pivot) - high,
            "R1": (2 * pivot) - low,
            "S2": pivot - (high - low),
            "R2": pivot + (high - low),
            "S3": low - 2 * (high - pivot),
            "R3": high + 2 * (pivot - low),
        }
    except Exception as e:
        logger.error(f"Error calculating standard pivot points: {e}", exc_info=True)
        return {}


def calculate_fib_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculate Fibonacci pivot points for the next period."""
    if not all(isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]):
        logger.warning("Invalid input for Fibonacci pivot points.")
        return {}
    if low > high:
        logger.warning(f"Low ({low}) > High ({high}) for Fib pivot calculation.")
    try:
        pivot = (high + low + close) / 3.0
        fib_range = high - low
        if abs(fib_range) < 1e-9:
            logger.warning("Zero or near-zero range for Fibonacci pivots.")
            return {"PP": pivot}
        return {
            "PP": pivot,
            "S1": pivot - (0.382 * fib_range),
            "R1": pivot + (0.382 * fib_range),
            "S2": pivot - (0.618 * fib_range),
            "R2": pivot + (0.618 * fib_range),
            "S3": pivot - (1.000 * fib_range),
            "R3": pivot + (1.000 * fib_range),
        }
    except Exception as e:
        logger.error(f"Error calculating Fibonacci pivot points: {e}", exc_info=True)
        return {}


# --- Support/Resistance Levels ---
def calculate_levels(df_period: pd.DataFrame, current_price: Optional[float] = None) -> Dict[str, Any]:
    """Calculate support/resistance levels including pivots and Fibonacci retracements."""
    levels = {
        "support": {},
        "resistance": {},
        "pivot": None,
        "fib_retracements": {},
        "standard_pivots": {},
        "fib_pivots": {},
    }
    required_cols = ["high", "low", "close"]
    if not all(col in df_period.columns for col in required_cols) or df_period.empty:
        logger.warning("Missing HLC columns or empty DataFrame for levels calculation.")
        return levels
    try:
        if len(df_period) >= 2:
            prev_high = df_period["high"].iloc[-2]
            prev_low = df_period["low"].iloc[-2]
            prev_close = df_period["close"].iloc[-2]
            levels["standard_pivots"] = calculate_standard_pivot_points(prev_high, prev_low, prev_close)
            levels["fib_pivots"] = calculate_fib_pivot_points(prev_high, prev_low, prev_close)
            levels["pivot"] = levels["standard_pivots"].get("PP", levels["fib_pivots"].get("PP"))

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

        if current_price is not None:
            all_levels = {
                **{f"Std {k}": v for k, v in levels["standard_pivots"].items()},
                **{f"Fib {k}": v for k, v in levels["fib_pivots"].items() if k != "PP"},
                **levels["fib_retracements"],
            }
            for label, value in all_levels.items():
                if isinstance(value, (int, float)):
                    if value < current_price:
                        levels["support"][label] = value
                    elif value > current_price:
                        levels["resistance"][label] = value

        levels["support"] = dict(sorted(levels["support"].items(), key=lambda x: x[1], reverse=True))
        levels["resistance"] = dict(sorted(levels["resistance"].items(), key=lambda x: x[1]))
        return levels
    except Exception as e:
        logger.error(f"Error calculating levels: {e}", exc_info=True)
        return levels


# --- Custom Indicators ---
def calculate_vwma(close: pd.Series, volume: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculate Volume Weighted Moving Average (VWMA)."""
    if close.empty or volume.empty or length <= 0 or len(close) < length or len(close) != len(volume):
        logger.warning(f"Invalid inputs for VWMA (length={length}).")
        return None
    try:
        pv = close * volume
        cumulative_pv = pv.rolling(window=length, min_periods=length).sum()
        cumulative_vol = volume.rolling(window=length, min_periods=length).sum()
        vwma = cumulative_pv / cumulative_vol
        vwma.replace([np.inf, -np.inf], np.nan, inplace=True)
        vwma.name = f"VWMA_{length}"
        return vwma
    except Exception as e:
        logger.error(f"Error calculating VWMA(length={length}): {e}", exc_info=True)
        return None


def ehlers_volumetric_trend(df: pd.DataFrame, length: int, multiplier: float) -> pd.DataFrame:
    """Calculate Ehlers Volumetric Trend with VWMA and SuperSmoother."""
    required_cols = ["close", "volume"]
    if not all(col in df.columns for col in required_cols) or df.empty or length <= 1 or multiplier <= 0:
        logger.warning(f"EVT skipped: Invalid params (len={length}, mult={multiplier}).")
        return df
    df_out = df.copy()
    vwma_col = f"vwma_{length}"
    smooth_col = f"smooth_vwma_{length}"
    trend_col = f"evt_trend_{length}"
    buy_col = f"evt_buy_{length}"
    sell_col = f"evt_sell_{length}"
    try:
        vwma = calculate_vwma(df_out["close"], df_out["volume"], length)
        if vwma is None:
            raise ValueError(f"VWMA calculation failed for EVT (length={length})")
        df_out[vwma_col] = vwma

        # SuperSmoother Filter
        a = np.exp(-1.414 * np.pi / length)
        b = 2 * a * np.cos(1.414 * np.pi / length)
        c2, c3, c1 = b, -a * a, 1 - b + a * a
        smoothed = pd.Series(np.nan, index=df_out.index, dtype=float)
        vwma_vals = df_out[vwma_col].values
        for i in range(2, len(df_out)):
            if not np.isnan(vwma_vals[i]):
                sm1 = smoothed.iloc[i - 1] if pd.notna(smoothed.iloc[i - 1]) else vwma_vals[i - 1]
                sm2 = smoothed.iloc[i - 2] if pd.notna(smoothed.iloc[i - 2]) else vwma_vals[i - 2]
                smoothed.iloc[i] = c1 * vwma_vals[i] + c2 * sm1 + c3 * sm2
        df_out[smooth_col] = smoothed

        # Trend Determination
        mult_h, mult_l = 1.0 + multiplier / 100.0, 1.0 - multiplier / 100.0
        shifted_smooth = df_out[smooth_col].shift(1)
        trend = pd.Series(0, index=df_out.index, dtype=int)
        up_trend = (
            (df_out[smooth_col] > shifted_smooth * mult_h) & pd.notna(df_out[smooth_col]) & pd.notna(shifted_smooth)
        )
        down_trend = (
            (df_out[smooth_col] < shifted_smooth * mult_l) & pd.notna(df_out[smooth_col]) & pd.notna(shifted_smooth)
        )
        for i in range(len(df_out)):
            trend.iloc[i] = 1 if up_trend.iloc[i] else -1 if down_trend.iloc[i] else trend.iloc[i - 1] if i > 0 else 0
        df_out[trend_col] = trend
        df_out[buy_col] = (df_out[trend_col] == 1) & (df_out[trend_col].shift(1).fillna(0) != 1)
        df_out[sell_col] = (df_out[trend_col] == -1) & (df_out[trend_col].shift(1).fillna(0) != -1)

        logger.debug(f"EVT calculated (len={length}, mult={multiplier}).")
        return df_out
    except Exception as e:
        logger.error(f"Error in EVT(len={length}, mult={multiplier}): {e}", exc_info=True)
        df_out[vwma_col] = df_out[smooth_col] = df_out[trend_col] = df_out[buy_col] = df_out[sell_col] = np.nan
        return df_out


def calculate_supertrend(df: pd.DataFrame, period: int, multiplier: float) -> pd.DataFrame:
    """Calculate Supertrend indicator."""
    required_cols = ["high", "low", "close"]
    if not all(col in df.columns for col in required_cols) or df.empty or period <= 0 or multiplier <= 0:
        logger.warning(f"Supertrend skipped: Invalid params (period={period}, mult={multiplier}).")
        return df
    df_out = df.copy()
    try:
        supertrend = ta.supertrend(
            high=df_out["high"], low=df_out["low"], close=df_out["close"], length=period, multiplier=multiplier
        )
        df_out[f"SUPERT_{period}_{multiplier}"] = supertrend[f"SUPERT_{period}_{multiplier}"]
        df_out[f"SUPERTd_{period}_{multiplier}"] = supertrend[f"SUPERTd_{period}_{multiplier}"]
        logger.debug(f"Supertrend calculated (period={period}, mult={multiplier}).")
        return df_out
    except Exception as e:
        logger.error(f"Error calculating Supertrend: {e}", exc_info=True)
        df_out[f"SUPERT_{period}_{multiplier}"] = np.nan
        df_out[f"SUPERTd_{period}_{multiplier}"] = np.nan
        return df_out


# --- Incremental Indicator Updates ---
def update_indicators_incrementally(
    df: pd.DataFrame, config: Dict[str, Any], prev_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Update indicators incrementally for the latest data point."""
    if df.empty or not all(col in df.columns for col in ["open", "high", "low", "close", "volume"]):
        logger.warning("Invalid DataFrame for incremental update.")
        return df
    df_out = df.copy()
    settings = config.get("indicator_settings", {})
    flags = config.get("analysis_flags", {})
    strategy_params = config.get("strategy_params", {}).get("ehlers_volumetric", {})

    try:
        # EVT
        if flags.get("use_evt"):
            evt_len = strategy_params.get("evt_length", settings.get("evt_length", 7))
            evt_mult = strategy_params.get("evt_multiplier", settings.get("evt_multiplier", 2.5))
            if prev_df is not None and len(prev_df) >= evt_len:
                df_combined = pd.concat([prev_df.iloc[-(evt_len - 1) :], df_out], ignore_index=True)
                df_combined = ehlers_volumetric_trend(df_combined, evt_len, float(evt_mult))
                df_out = df_combined.iloc[-len(df_out) :].copy()
            else:
                df_out = ehlers_volumetric_trend(df_out, evt_len, float(evt_mult))

        # ATR
        if flags.get("use_atr"):
            atr_len = settings.get("atr_period", 14)
            if prev_df is not None and len(prev_df) >= atr_len:
                df_combined = pd.concat([prev_df.iloc[-(atr_len - 1) :], df_out], ignore_index=True)
                atr_result = ta.atr(df_combined["high"], df_combined["low"], df_combined["close"], length=atr_len)
                df_out[f"ATRr_{atr_len}"] = atr_result.iloc[-len(df_out) :]
            else:
                df_out[f"ATRr_{atr_len}"] = ta.atr(df_out["high"], df_out["low"], df_out["close"], length=atr_len)

        # Moving Averages
        if flags.get("use_sma"):
            for period in [settings.get("sma_short_period", 10), settings.get("sma_long_period", 50)]:
                if period > 0 and prev_df is not None and len(prev_df) >= period:
                    df_combined = pd.concat([prev_df.iloc[-(period - 1) :], df_out], ignore_index=True)
                    df_out[f"SMA_{period}"] = ta.sma(df_combined["close"], length=period).iloc[-len(df_out) :]
                elif period > 0:
                    df_out[f"SMA_{period}"] = ta.sma(df_out["close"], length=period)

        if flags.get("use_ema"):
            for period in [settings.get("ema_short_period", 12), settings.get("ema_long_period", 26)]:
                if period > 0 and prev_df is not None and len(prev_df) >= period:
                    df_combined = pd.concat([prev_df.iloc[-(period - 1) :], df_out], ignore_index=True)
                    df_out[f"EMA_{period}"] = ta.ema(df_combined["close"], length=period).iloc[-len(df_out) :]
                elif period > 0:
                    df_out[f"EMA_{period}"] = ta.ema(df_combined["close"], length=period)

        # ADX
        if flags.get("use_adx"):
            adx_len = settings.get("adx_period", 14)
            if prev_df is not None and len(prev_df) >= adx_len:
                df_combined = pd.concat([prev_df.iloc[-(adx_len - 1) :], df_out], ignore_index=True)
                adx_result = ta.adx(df_combined["high"], df_combined["low"], df_combined["close"], length=adx_len)
                df_out[f"ADX_{adx_len}"] = adx_result[f"ADX_{adx_len}"].iloc[-len(df_out) :]
            elif adx_len > 0:
                df_out[f"ADX_{adx_len}"] = ta.adx(df_out["high"], df_out["low"], df_out["close"], length=adx_len)[
                    f"ADX_{adx_len}"
                ]

        # Supertrend
        if flags.get("use_supertrend"):
            st_period = settings.get("supertrend_period", 10)
            st_mult = settings.get("supertrend_multiplier", 3.0)
            if prev_df is not None and len(prev_df) >= st_period:
                df_combined = pd.concat([prev_df.iloc[-(st_period - 1) :], df_out], ignore_index=True)
                st_result = ta.supertrend(
                    df_combined["high"], df_combined["low"], df_combined["close"], length=st_period, multiplier=st_mult
                )
                df_out[f"SUPERT_{st_period}_{st_mult}"] = st_result[f"SUPERT_{st_period}_{st_mult}"].iloc[
                    -len(df_out) :
                ]
                df_out[f"SUPERTd_{st_period}_{st_mult}"] = st_result[f"SUPERTd_{st_period}_{st_mult}"].iloc[
                    -len(df_out) :
                ]
            elif st_period > 0:
                df_out = calculate_supertrend(df_out, st_period, st_mult)

        return df_out
    except Exception as e:
        logger.error(f"Error in incremental indicator update: {e}", exc_info=True)
        return df_out


# --- Master Indicator Calculation ---
def calculate_all_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Calculate all enabled trend analysis indicators based on config."""
    if df.empty or not all(col in df.columns for col in ["open", "high", "low", "close", "volume"]):
        logger.error("Invalid DataFrame for indicator calculation.")
        return pd.DataFrame()
    df_out = df.copy()
    settings = config.get("indicator_settings", {})
    flags = config.get("analysis_flags", {})
    min_rows = settings.get("min_data_periods", MIN_PERIODS_DEFAULT)
    strategy_params = config.get("strategy_params", {}).get("ehlers_volumetric", {})

    if len(df_out.dropna(subset=["open", "high", "low", "close", "volume"])) < min_rows:
        logger.warning(f"Insufficient valid rows ({len(df_out)}) for reliable calculations. Need {min_rows}.")

    try:
        # Moving Averages
        if flags.get("use_sma"):
            for period in [settings.get("sma_short_period", 10), settings.get("sma_long_period", 50)]:
                if period > 0:
                    df_out[f"SMA_{period}"] = ta.sma(df_out["close"], length=period)

        if flags.get("use_ema"):
            for period in [settings.get("ema_short_period", 12), settings.get("ema_long_period", 26)]:
                if period > 0:
                    df_out[f"EMA_{period}"] = ta.ema(df_out["close"], length=period)

        if flags.get("use_wma"):
            wma_len = settings.get("wma_period", 9)
            if wma_len > 0:
                df_out[f"WMA_{wma_len}"] = ta.wma(df_out["close"], length=wma_len)

        if flags.get("use_hma"):
            hma_len = settings.get("hma_period", 9)
            if hma_len > 0:
                df_out[f"HMA_{hma_len}"] = ta.hma(df_out["close"], length=hma_len)

        if flags.get("use_dema"):
            dema_len = settings.get("dema_period", 9)
            if dema_len > 0:
                df_out[f"DEMA_{dema_len}"] = ta.dema(df_out["close"], length=dema_len)

        if flags.get("use_tema"):
            tema_len = settings.get("tema_period", 9)
            if tema_len > 0:
                df_out[f"TEMA_{tema_len}"] = ta.tema(df_out["close"], length=tema_len)

        # Trend Indicators
        if flags.get("use_adx"):
            adx_len = settings.get("adx_period", 14)
            if adx_len > 0:
                adx_result = ta.adx(df_out["high"], df_out["low"], df_out["close"], length=adx_len)
                df_out[f"ADX_{adx_len}"] = adx_result[f"ADX_{adx_len}"]
                df_out[f"DMP_{adx_len}"] = adx_result[f"DMP_{adx_len}"]
                df_out[f"DMN_{adx_len}"] = adx_result[f"DMN_{adx_len}"]

        if flags.get("use_psar"):
            step = settings.get("psar_step", 0.02)
            max_step = settings.get("psar_max_step", 0.2)
            if step > 0 and max_step > step:
                psar_result = ta.psar(df_out["high"], df_out["low"], df_out["close"], af=step, max_af=max_step)
                df_out[f"PSARl_{step}_{max_step}"] = psar_result[f"PSARl_{step}_{max_step}"]
                df_out[f"PSARs_{step}_{max_step}"] = psar_result[f"PSARs_{step}_{max_step}"]

        if flags.get("use_supertrend"):
            st_period = settings.get("supertrend_period", 10)
            st_mult = settings.get("supertrend_multiplier", 3.0)
            if st_period > 0 and st_mult > 0:
                df_out = calculate_supertrend(df_out, st_period, st_mult)

        if flags.get("use_ichimoku"):
            tenkan = settings.get("ichimoku_tenkan", 9)
            kijun = settings.get("ichimoku_kijun", 26)
            senkou = settings.get("ichimoku_senkou", 52)
            if all(p > 0 for p in [tenkan, kijun, senkou]):
                ichi_lines, ichi_spans = ta.ichimoku(
                    high=df_out["high"],
                    low=df_out["low"],
                    close=df_out["close"],
                    tenkan=tenkan,
                    kijun=kijun,
                    senkou=senkou,
                )
                if ichi_lines is not None:
                    df_out = pd.concat([df_out, ichi_lines], axis=1)
                if ichi_spans is not None:
                    df_out = pd.concat([df_out, ichi_spans], axis=1)

        # Momentum-Based Trend Indicators
        if flags.get("use_macd"):
            fast = settings.get("macd_fast", 12)
            slow = settings.get("macd_slow", 26)
            signal = settings.get("macd_signal", 9)
            if fast > 0 and slow > 0 and signal > 0 and fast < slow:
                macd_result = ta.macd(df_out["close"], fast=fast, slow=slow, signal=signal)
                df_out[f"MACD_{fast}_{slow}_{signal}"] = macd_result[f"MACD_{fast}_{slow}_{signal}"]
                df_out[f"MACDh_{fast}_{slow}_{signal}"] = macd_result[f"MACDh_{fast}_{slow}_{signal}"]
                df_out[f"MACDs_{fast}_{slow}_{signal}"] = macd_result[f"MACDs_{fast}_{slow}_{signal}"]

        if flags.get("use_ao"):
            ao_fast = settings.get("ao_fast", 5)
            ao_slow = settings.get("ao_slow", 34)
            if ao_fast > 0 and ao_slow > 0 and ao_fast < ao_slow:
                df_out[f"AO_{ao_fast}_{ao_slow}"] = ta.ao(df_out["high"], df_out["low"], fast=ao_fast, slow=ao_slow)

        if flags.get("use_kst"):
            roc1 = settings.get("kst_roc1", 10)
            roc2 = settings.get("kst_roc2", 15)
            roc3 = settings.get("kst_roc3", 20)
            roc4 = settings.get("kst_roc4", 30)
            sma1 = settings.get("kst_sma1", 10)
            sma2 = settings.get("kst_sma2", 10)
            sma3 = settings.get("kst_sma3", 10)
            sma4 = settings.get("kst_sma4", 15)
            signal = settings.get("kst_signal", 9)
            if all(p > 0 for p in [roc1, roc2, roc3, roc4, sma1, sma2, sma3, sma4, signal]):
                kst_result = ta.kst(
                    df_out["close"],
                    roc1=roc1,
                    roc2=roc2,
                    roc3=roc3,
                    roc4=roc4,
                    sma1=sma1,
                    sma2=sma2,
                    sma3=sma3,
                    sma4=sma4,
                    signal=signal,
                )
                df_out[f"KST_{roc1}_{roc2}_{roc3}_{roc4}"] = kst_result[f"KST_{roc1}_{roc2}_{roc3}_{roc4}"]
                df_out[f"KSTs_{roc1}_{roc2}_{roc3}_{roc4}"] = kst_result[f"KSTs_{roc1}_{roc2}_{roc3}_{roc4}"]

        # Volatility-Based Trend Channels
        if flags.get("use_keltner"):
            kc_len = settings.get("kc_length", 20)
            kc_atr_len = settings.get("kc_atr_length", 10)
            kc_mult = settings.get("kc_multiplier", 2.0)
            if kc_len > 0 and kc_atr_len > 0 and kc_mult > 0:
                kc_result = ta.kc(
                    high=df_out["high"],
                    low=df_out["low"],
                    close=df_out["close"],
                    length=kc_len,
                    atr_length=kc_atr_len,
                    scalar=kc_mult,
                    mamode="ema",
                )
                df_out[f"KCUe_{kc_len}_{kc_mult}"] = kc_result[f"KCUe_{kc_len}_{kc_mult}"]
                df_out[f"KCMe_{kc_len}_{kc_mult}"] = kc_result[f"KCMe_{kc_len}_{kc_mult}"]
                df_out[f"KCle_{kc_len}_{kc_mult}"] = kc_result[f"KCle_{kc_len}_{kc_mult}"]

        if flags.get("use_donchian"):
            dc_len = settings.get("donchian_length", 20)
            if dc_len > 0:
                dc_result = ta.donchian(df_out["high"], df_out["low"], length=dc_len)
                df_out[f"DCU_{dc_len}"] = dc_result[f"DCU_{dc_len}"]
                df_out[f"DCM_{dc_len}"] = dc_result[f"DCM_{dc_len}"]
                df_out[f"DCL_{dc_len}"] = dc_result[f"DCL_{dc_len}"]

        # Volume-Based
        if flags.get("use_vwma"):
            vwma_len = settings.get("vwma_period", 14)
            if vwma_len > 0:
                vwma = calculate_vwma(df_out["close"], df_out["volume"], vwma_len)
                if vwma is not None:
                    df_out[f"VWMA_{vwma_len}"] = vwma

        # Custom EVT
        if flags.get("use_evt"):
            evt_len = strategy_params.get("evt_length", settings.get("evt_length", 7))
            evt_mult = strategy_params.get("evt_multiplier", settings.get("evt_multiplier", 2.5))
            df_out = ehlers_volumetric_trend(df_out, evt_len, float(evt_mult))

        # ATR (used by multiple indicators)
        if flags.get("use_atr") or any(flags.get(k) for k in ["use_supertrend", "use_keltner"]):
            atr_len = settings.get("atr_period", 14)
            if atr_len > 0:
                df_out[f"ATRr_{atr_len}"] = ta.atr(df_out["high"], df_out["low"], df_out["close"], length=atr_len)

        logger.debug(f"Calculated indicators. DataFrame shape: {df_out.shape}")
        return df_out
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True)
        return df_out


# --- Standalone Demo ---
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    test_config = {
        "indicator_settings": {
            "min_data_periods": 50,
            "sma_short_period": 10,
            "sma_long_period": 50,
            "ema_short_period": 12,
            "ema_long_period": 26,
            "wma_period": 9,
            "hma_period": 9,
            "dema_period": 9,
            "tema_period": 9,
            "adx_period": 14,
            "psar_step": 0.02,
            "psar_max_step": 0.2,
            "supertrend_period": 10,
            "supertrend_multiplier": 3.0,
            "ichimoku_tenkan": 9,
            "ichimoku_kijun": 26,
            "ichimoku_senkou": 52,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "ao_fast": 5,
            "ao_slow": 34,
            "kst_roc1": 10,
            "kst_roc2": 15,
            "kst_roc3": 20,
            "kst_roc4": 30,
            "kst_sma1": 10,
            "kst_sma2": 10,
            "kst_sma3": 10,
            "kst_sma4": 15,
            "kst_signal": 9,
            "kc_length": 20,
            "kc_atr_length": 10,
            "kc_multiplier": 2.0,
            "donchian_length": 20,
            "vwma_period": 14,
            "evt_length": 7,
            "evt_multiplier": 2.5,
            "atr_period": 14,
        },
        "analysis_flags": {
            "use_sma": True,
            "use_ema": True,
            "use_wma": True,
            "use_hma": True,
            "use_dema": True,
            "use_tema": True,
            "use_adx": True,
            "use_psar": True,
            "use_supertrend": True,
            "use_ichimoku": True,
            "use_macd": True,
            "use_ao": True,
            "use_kst": True,
            "use_keltner": True,
            "use_donchian": True,
            "use_vwma": True,
            "use_evt": True,
            "use_atr": True,
        },
        "strategy_params": {"ehlers_volumetric": {"evt_length": 7, "evt_multiplier": 2.5}},
    }
    periods = 200
    prices = 1000 * np.exp(np.cumsum(np.random.normal(loc=0.0001, scale=0.01, size=periods)))
    df_test = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2025-01-01", periods=periods, freq="5min"),
            "open": prices[:-1],
            "close": prices[1:],
            "high": prices[1:] * (1 + np.random.uniform(0, 0.01, periods - 1)),
            "low": prices[1:] * (1 - np.random.uniform(0, 0.01, periods - 1)),
            "volume": np.random.uniform(100, 2000, periods - 1),
        }
    ).set_index("timestamp")
    df_test["high"] = np.maximum.reduce([df_test["open"], df_test["close"], df_test["high"]])
    df_test["low"] = np.minimum.reduce([df_test["open"], df_test["close"], df_test["low"]])
    df_results = calculate_all_indicators(df_test, test_config)
    print(f"Output shape: {df_results.shape}")
    print(f"Columns: {df_results.columns.tolist()}")
    print(f"Last row:\n{df_results.iloc[-1]}")
