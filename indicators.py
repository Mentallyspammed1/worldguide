"""
Technical Indicators Module

This module contains functions for calculating technical indicators
and trading signals based on the configured strategy.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import pandas_ta as ta

# Configure logger
logger = logging.getLogger("indicators")

# Default indicator parameters
DEFAULT_RSI_WINDOW = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BB_WINDOW = 20
DEFAULT_BB_STD = 2.0
DEFAULT_EMA_FAST = 8
DEFAULT_EMA_SLOW = 21
DEFAULT_ATR_WINDOW = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_STOCH_WINDOW = 14
DEFAULT_STOCH_K = 3
DEFAULT_STOCH_D = 3


def calculate_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Calculate technical indicators based on configuration.
    
    Args:
        df: DataFrame with OHLCV data
        config: Dictionary with indicator configuration
        
    Returns:
        pd.DataFrame: DataFrame with added indicator columns
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate RSI if enabled
    if "rsi" in config and config["rsi"].get("enabled", True):
        window = config["rsi"].get("window", DEFAULT_RSI_WINDOW)
        try:
            df_copy["rsi"] = ta.rsi(df_copy["close"], length=window)
            logger.debug(f"Calculated RSI with window={window}")
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
    
    # Calculate MACD if enabled
    if "macd" in config and config["macd"].get("enabled", True):
        fast_period = config["macd"].get("fast_period", DEFAULT_MACD_FAST)
        slow_period = config["macd"].get("slow_period", DEFAULT_MACD_SLOW)
        signal_period = config["macd"].get("signal_period", DEFAULT_MACD_SIGNAL)
        try:
            macd = ta.macd(
                df_copy["close"], 
                fast=fast_period, 
                slow=slow_period, 
                signal=signal_period
            )
            # Add MACD components to dataframe
            df_copy["macd"] = macd["MACD_" + str(fast_period) + "_" + str(slow_period) + "_" + str(signal_period)]
            df_copy["macd_signal"] = macd["MACDs_" + str(fast_period) + "_" + str(slow_period) + "_" + str(signal_period)]
            df_copy["macd_hist"] = macd["MACDh_" + str(fast_period) + "_" + str(slow_period) + "_" + str(signal_period)]
            logger.debug(f"Calculated MACD with fast={fast_period}, slow={slow_period}, signal={signal_period}")
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
    
    # Calculate Bollinger Bands if enabled
    if "bollinger_bands" in config and config["bollinger_bands"].get("enabled", True):
        window = config["bollinger_bands"].get("window", DEFAULT_BB_WINDOW)
        std_dev = config["bollinger_bands"].get("std_dev", DEFAULT_BB_STD)
        try:
            bbands = ta.bbands(df_copy["close"], length=window, std=std_dev)
            df_copy["bb_upper"] = bbands["BBU_" + str(window) + "_" + str(std_dev)]
            df_copy["bb_middle"] = bbands["BBM_" + str(window) + "_" + str(std_dev)]
            df_copy["bb_lower"] = bbands["BBL_" + str(window) + "_" + str(std_dev)]
            df_copy["bb_width"] = bbands["BBB_" + str(window) + "_" + str(std_dev)]
            logger.debug(f"Calculated Bollinger Bands with window={window}, std_dev={std_dev}")
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
    
    # Calculate EMA Cross if enabled
    if "ema_cross" in config and config["ema_cross"].get("enabled", True):
        fast_period = config["ema_cross"].get("fast_period", DEFAULT_EMA_FAST)
        slow_period = config["ema_cross"].get("slow_period", DEFAULT_EMA_SLOW)
        try:
            df_copy["ema_fast"] = ta.ema(df_copy["close"], length=fast_period)
            df_copy["ema_slow"] = ta.ema(df_copy["close"], length=slow_period)
            # Calculate crossover signals
            df_copy["ema_cross"] = np.where(
                df_copy["ema_fast"] > df_copy["ema_slow"], 
                1, 
                np.where(df_copy["ema_fast"] < df_copy["ema_slow"], -1, 0)
            )
            logger.debug(f"Calculated EMA Cross with fast={fast_period}, slow={slow_period}")
        except Exception as e:
            logger.error(f"Error calculating EMA Cross: {e}")
    
    # Calculate ATR if enabled
    if "atr" in config and config["atr"].get("enabled", True):
        window = config["atr"].get("window", DEFAULT_ATR_WINDOW)
        try:
            df_copy["atr"] = ta.atr(
                df_copy["high"], 
                df_copy["low"], 
                df_copy["close"], 
                length=window
            )
            logger.debug(f"Calculated ATR with window={window}")
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
    
    # Calculate CCI if enabled
    if "cci" in config and config["cci"].get("enabled", True):
        window = config["cci"].get("window", DEFAULT_CCI_WINDOW)
        try:
            df_copy["cci"] = ta.cci(
                df_copy["high"], 
                df_copy["low"], 
                df_copy["close"], 
                length=window
            )
            logger.debug(f"Calculated CCI with window={window}")
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
    
    # Calculate Stochastic if enabled
    if "stochastic" in config and config["stochastic"].get("enabled", True):
        window = config["stochastic"].get("window", DEFAULT_STOCH_WINDOW)
        k = config["stochastic"].get("k", DEFAULT_STOCH_K)
        d = config["stochastic"].get("d", DEFAULT_STOCH_D)
        try:
            stoch = ta.stoch(
                df_copy["high"], 
                df_copy["low"], 
                df_copy["close"], 
                k=window, 
                d=k, 
                smooth_d=d
            )
            df_copy["stoch_k"] = stoch["STOCHk_" + str(window) + "_" + str(k) + "_" + str(d)]
            df_copy["stoch_d"] = stoch["STOCHd_" + str(window) + "_" + str(k) + "_" + str(d)]
            logger.debug(f"Calculated Stochastic with window={window}, k={k}, d={d}")
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
    
    # Calculate Stochastic RSI if enabled
    if "stoch_rsi" in config and config["stoch_rsi"].get("enabled", True):
        window = config["stoch_rsi"].get("window", DEFAULT_STOCH_WINDOW)
        k = config["stoch_rsi"].get("k", DEFAULT_STOCH_K)
        d = config["stoch_rsi"].get("d", DEFAULT_STOCH_D)
        try:
            stoch_rsi = ta.stochrsi(
                df_copy["close"], 
                length=window, 
                rsi_length=window, 
                k=k, 
                d=d
            )
            df_copy["stoch_rsi_k"] = stoch_rsi["STOCHRSIk_" + str(window) + "_" + str(k) + "_" + str(d)]
            df_copy["stoch_rsi_d"] = stoch_rsi["STOCHRSId_" + str(window) + "_" + str(k) + "_" + str(d)]
            logger.debug(f"Calculated Stochastic RSI with window={window}, k={k}, d={d}")
        except Exception as e:
            logger.error(f"Error calculating Stochastic RSI: {e}")
    
    # Calculate PSAR if enabled
    if "psar" in config and config["psar"].get("enabled", True):
        af = config["psar"].get("af", 0.02)
        max_af = config["psar"].get("max_af", 0.2)
        try:
            psar = ta.psar(
                df_copy["high"], 
                df_copy["low"], 
                af=af, 
                max_af=max_af
            )
            df_copy["psar"] = psar["PSARl_" + str(af).replace(".", "_") + "_" + str(max_af).replace(".", "_")]
            df_copy["psar_direction"] = psar["PSARaf_" + str(af).replace(".", "_") + "_" + str(max_af).replace(".", "_")]
            df_copy["psar_signal"] = np.where(
                df_copy["close"] > df_copy["psar"], 
                1, 
                np.where(df_copy["close"] < df_copy["psar"], -1, 0)
            )
            logger.debug(f"Calculated PSAR with af={af}, max_af={max_af}")
        except Exception as e:
            logger.error(f"Error calculating PSAR: {e}")
    
    return df_copy


def calculate_rsi_signal(df: pd.DataFrame, config: Dict) -> Tuple[float, str]:
    """
    Calculate trading signal based on RSI indicator.
    
    Args:
        df: DataFrame with indicator data
        config: RSI configuration
        
    Returns:
        Tuple[float, str]: Signal strength (-1.0 to 1.0) and direction ("long" or "short")
    """
    if "rsi" not in df.columns:
        return 0.0, ""
    
    # Get parameters
    rsi = df["rsi"].iloc[-1]
    overbought = config.get("overbought", 70)
    oversold = config.get("oversold", 30)
    
    # Calculate signal
    if rsi <= oversold:
        # Oversold - buy signal
        signal_strength = 1.0 - (rsi / oversold)
        return signal_strength, "long"
    elif rsi >= overbought:
        # Overbought - sell signal
        signal_strength = (rsi - overbought) / (100 - overbought)
        return signal_strength, "short"
    else:
        # Neutral zone - weak signal
        midpoint = (overbought + oversold) / 2
        if rsi < midpoint:
            signal_strength = (midpoint - rsi) / (midpoint - oversold) * 0.5
            return signal_strength, "long"
        else:
            signal_strength = (rsi - midpoint) / (overbought - midpoint) * 0.5
            return signal_strength, "short"


def calculate_macd_signal(df: pd.DataFrame, config: Dict) -> Tuple[float, str]:
    """
    Calculate trading signal based on MACD indicator.
    
    Args:
        df: DataFrame with indicator data
        config: MACD configuration
        
    Returns:
        Tuple[float, str]: Signal strength (-1.0 to 1.0) and direction ("long" or "short")
    """
    if "macd" not in df.columns or "macd_signal" not in df.columns or "macd_hist" not in df.columns:
        return 0.0, ""
    
    # Get latest values
    macd = df["macd"].iloc[-1]
    signal = df["macd_signal"].iloc[-1]
    hist = df["macd_hist"].iloc[-1]
    prev_hist = df["macd_hist"].iloc[-2] if len(df) > 2 else 0
    
    # Calculate signal
    if hist > 0 and prev_hist <= 0:
        # Bullish crossover
        signal_strength = min(1.0, abs(hist) / 0.5)  # Normalize with typical value
        return signal_strength, "long"
    elif hist < 0 and prev_hist >= 0:
        # Bearish crossover
        signal_strength = min(1.0, abs(hist) / 0.5)  # Normalize with typical value
        return signal_strength, "short"
    elif hist > 0:
        # Bullish trend
        signal_strength = min(0.5, abs(hist) / 1.0)  # Weaker signal during trend
        return signal_strength, "long"
    elif hist < 0:
        # Bearish trend
        signal_strength = min(0.5, abs(hist) / 1.0)  # Weaker signal during trend
        return signal_strength, "short"
    else:
        return 0.0, ""


def calculate_bollinger_signal(df: pd.DataFrame, config: Dict) -> Tuple[float, str]:
    """
    Calculate trading signal based on Bollinger Bands indicator.
    
    Args:
        df: DataFrame with indicator data
        config: Bollinger Bands configuration
        
    Returns:
        Tuple[float, str]: Signal strength (-1.0 to 1.0) and direction ("long" or "short")
    """
    if "bb_upper" not in df.columns or "bb_lower" not in df.columns or "bb_middle" not in df.columns:
        return 0.0, ""
    
    # Get latest values
    close = df["close"].iloc[-1]
    upper = df["bb_upper"].iloc[-1]
    lower = df["bb_lower"].iloc[-1]
    middle = df["bb_middle"].iloc[-1]
    width = df["bb_width"].iloc[-1] if "bb_width" in df.columns else (upper - lower) / middle
    
    # Percentage distance from bands
    upper_dist = (upper - close) / (upper - lower) if upper != lower else 0
    lower_dist = (close - lower) / (upper - lower) if upper != lower else 0
    
    # Calculate signal
    if close <= lower:
        # Price at or below lower band - strong buy signal
        signal_strength = min(1.0, 1.0 + lower_dist)  # Can exceed 1.0 for breakouts
        return signal_strength, "long"
    elif close >= upper:
        # Price at or above upper band - strong sell signal
        signal_strength = min(1.0, 1.0 + upper_dist)  # Can exceed 1.0 for breakouts
        return signal_strength, "short"
    elif close < middle:
        # Price between middle and lower band - weak buy signal
        signal_strength = lower_dist * 0.5  # Scale down for less confident signal
        return signal_strength, "long"
    elif close > middle:
        # Price between middle and upper band - weak sell signal
        signal_strength = upper_dist * 0.5  # Scale down for less confident signal
        return signal_strength, "short"
    else:
        return 0.0, ""


def calculate_ema_cross_signal(df: pd.DataFrame, config: Dict) -> Tuple[float, str]:
    """
    Calculate trading signal based on EMA cross indicator.
    
    Args:
        df: DataFrame with indicator data
        config: EMA cross configuration
        
    Returns:
        Tuple[float, str]: Signal strength (-1.0 to 1.0) and direction ("long" or "short")
    """
    if "ema_fast" not in df.columns or "ema_slow" not in df.columns:
        return 0.0, ""
    
    # Get latest values
    fast = df["ema_fast"].iloc[-1]
    slow = df["ema_slow"].iloc[-1]
    
    # Get previous values if available
    prev_fast = df["ema_fast"].iloc[-2] if len(df) > 2 else fast
    prev_slow = df["ema_slow"].iloc[-2] if len(df) > 2 else slow
    
    # Calculate percentage difference between fast and slow EMAs
    diff_pct = (fast - slow) / slow * 100 if slow != 0 else 0
    
    # Calculate signal
    if fast > slow and prev_fast <= prev_slow:
        # Bullish crossover - strong buy signal
        signal_strength = 1.0
        return signal_strength, "long"
    elif fast < slow and prev_fast >= prev_slow:
        # Bearish crossover - strong sell signal
        signal_strength = 1.0
        return signal_strength, "short"
    elif fast > slow:
        # Fast above slow - existing bullish trend
        signal_strength = min(0.5, abs(diff_pct) / 2.0)  # Normalize to reasonable range
        return signal_strength, "long"
    elif fast < slow:
        # Fast below slow - existing bearish trend
        signal_strength = min(0.5, abs(diff_pct) / 2.0)  # Normalize to reasonable range
        return signal_strength, "short"
    else:
        return 0.0, ""


def calculate_atr_signal(df: pd.DataFrame, config: Dict) -> Tuple[float, str]:
    """
    Calculate volatility-adjusted signal based on ATR.
    This function doesn't provide direction, only volatility assessment.
    
    Args:
        df: DataFrame with indicator data
        config: ATR configuration
        
    Returns:
        Tuple[float, str]: Signal strength (0.0 to 1.0) and empty direction
    """
    if "atr" not in df.columns:
        return 0.0, ""
    
    # Get latest values
    atr = df["atr"].iloc[-1]
    close = df["close"].iloc[-1]
    
    # Calculate ATR as percentage of price
    atr_pct = (atr / close) * 100 if close != 0 else 0
    
    # Typical ATR percentages range from 0.5% to 5% for most assets
    # Higher ATR percentage indicates higher volatility
    # Scale to 0-1 range assuming 5% is maximum expected volatility
    max_expected_atr_pct = 5.0
    volatility_score = min(1.0, atr_pct / max_expected_atr_pct)
    
    # ATR doesn't provide direction, only volatility assessment
    return volatility_score, ""


def calculate_signal(df: pd.DataFrame, indicators_config: Dict, 
                     threshold: float = 0.5) -> Tuple[float, str]:
    """
    Calculate overall trading signal based on multiple indicators.
    
    Args:
        df: DataFrame with indicator data
        indicators_config: Configuration for all indicators
        threshold: Signal threshold for entry
        
    Returns:
        Tuple[float, str]: Overall signal strength (-1.0 to 1.0) and direction ("long" or "short")
    """
    if df.empty:
        return 0.0, ""
    
    # Initialize variables for weighted signal calculation
    total_weight = 0.0
    long_signal = 0.0
    short_signal = 0.0
    
    # RSI signal
    if "rsi" in indicators_config and indicators_config["rsi"].get("enabled", True):
        weight = float(indicators_config["rsi"].get("weight", 1.0))
        signal_strength, direction = calculate_rsi_signal(df, indicators_config["rsi"])
        total_weight += weight
        if direction == "long":
            long_signal += signal_strength * weight
        elif direction == "short":
            short_signal += signal_strength * weight
    
    # MACD signal
    if "macd" in indicators_config and indicators_config["macd"].get("enabled", True):
        weight = float(indicators_config["macd"].get("weight", 1.0))
        signal_strength, direction = calculate_macd_signal(df, indicators_config["macd"])
        total_weight += weight
        if direction == "long":
            long_signal += signal_strength * weight
        elif direction == "short":
            short_signal += signal_strength * weight
    
    # Bollinger Bands signal
    if "bollinger_bands" in indicators_config and indicators_config["bollinger_bands"].get("enabled", True):
        weight = float(indicators_config["bollinger_bands"].get("weight", 1.0))
        signal_strength, direction = calculate_bollinger_signal(df, indicators_config["bollinger_bands"])
        total_weight += weight
        if direction == "long":
            long_signal += signal_strength * weight
        elif direction == "short":
            short_signal += signal_strength * weight
    
    # EMA Cross signal
    if "ema_cross" in indicators_config and indicators_config["ema_cross"].get("enabled", True):
        weight = float(indicators_config["ema_cross"].get("weight", 1.0))
        signal_strength, direction = calculate_ema_cross_signal(df, indicators_config["ema_cross"])
        total_weight += weight
        if direction == "long":
            long_signal += signal_strength * weight
        elif direction == "short":
            short_signal += signal_strength * weight
    
    # If no weights/signals, return neutral
    if total_weight == 0:
        return 0.0, ""
    
    # Calculate final signals normalized by total weight
    final_long_signal = long_signal / total_weight
    final_short_signal = short_signal / total_weight
    
    # Determine direction based on strongest signal
    if final_long_signal > final_short_signal:
        if final_long_signal >= threshold:
            return final_long_signal, "long"
        else:
            return final_long_signal, ""  # Below threshold
    elif final_short_signal > final_long_signal:
        if final_short_signal >= threshold:
            return final_short_signal, "short"
        else:
            return -final_short_signal, ""  # Below threshold
    else:
        return 0.0, ""  # Neutral