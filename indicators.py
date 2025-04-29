"""
Technical Indicators Module

This module provides functions for calculating various technical indicators
used in trading strategies, including:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- EMA/SMA (Exponential/Simple Moving Averages)
- Supertrend
- And more
"""

import logging
import numpy as np
import pandas as pd
import pandas_ta as ta  # pandas_ta provides many technical indicators

# Configure logger
logger = logging.getLogger("indicators")


def calculate_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculate standard technical indicators on OHLCV data
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary with indicator parameters
        
    Returns:
        pd.DataFrame: DataFrame with added indicator columns
    """
    if df is None or len(df) == 0:
        logger.warning("Empty DataFrame provided to calculate_indicators")
        return df
    
    try:
        # Extract indicator config with defaults
        indicator_config = config.get("strategy", {}).get("indicators", {})
        
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Add basic indicators based on config
        # RSI
        if indicator_config.get("rsi", {}).get("enabled", True):
            rsi_period = indicator_config.get("rsi", {}).get("window", 14)
            df_copy["rsi"] = ta.rsi(df_copy["close"], length=rsi_period)
        
        # MACD
        if indicator_config.get("macd", {}).get("enabled", True):
            fast_period = indicator_config.get("macd", {}).get("fast_period", 12)
            slow_period = indicator_config.get("macd", {}).get("slow_period", 26)
            signal_period = indicator_config.get("macd", {}).get("signal_period", 9)
            
            macd = ta.macd(
                df_copy["close"],
                fast=fast_period,
                slow=slow_period,
                signal=signal_period
            )
            
            df_copy["macd"] = macd["MACD_" + str(fast_period) + "_" + str(slow_period) + "_" + str(signal_period)]
            df_copy["macd_signal"] = macd["MACDs_" + str(fast_period) + "_" + str(slow_period) + "_" + str(signal_period)]
            df_copy["macd_hist"] = macd["MACDh_" + str(fast_period) + "_" + str(slow_period) + "_" + str(signal_period)]
        
        # Bollinger Bands
        if indicator_config.get("bollinger_bands", {}).get("enabled", True):
            bb_period = indicator_config.get("bollinger_bands", {}).get("window", 20)
            bb_std = indicator_config.get("bollinger_bands", {}).get("std_dev", 2.0)
            
            bbands = ta.bbands(df_copy["close"], length=bb_period, std=bb_std)
            df_copy["bb_upper"] = bbands["BBU_" + str(bb_period) + "_" + str(bb_std)]
            df_copy["bb_middle"] = bbands["BBM_" + str(bb_period) + "_" + str(bb_std)]
            df_copy["bb_lower"] = bbands["BBL_" + str(bb_period) + "_" + str(bb_std)]
        
        # ATR
        if indicator_config.get("atr", {}).get("enabled", True):
            atr_period = indicator_config.get("atr", {}).get("window", 14)
            df_copy["atr"] = ta.atr(df_copy["high"], df_copy["low"], df_copy["close"], length=atr_period)
        
        # Moving Averages (EMA/SMA)
        if indicator_config.get("ema_cross", {}).get("enabled", True):
            ema_fast_period = indicator_config.get("ema_cross", {}).get("fast_period", 8)
            ema_slow_period = indicator_config.get("ema_cross", {}).get("slow_period", 21)
            
            df_copy["ema_fast"] = ta.ema(df_copy["close"], length=ema_fast_period)
            df_copy["ema_slow"] = ta.ema(df_copy["close"], length=ema_slow_period)
        
        if indicator_config.get("sma", {}).get("enabled", True):
            sma_fast_period = indicator_config.get("sma", {}).get("fast_period", 10)
            sma_slow_period = indicator_config.get("sma", {}).get("slow_period", 50)
            
            df_copy["sma_fast"] = ta.sma(df_copy["close"], length=sma_fast_period)
            df_copy["sma_slow"] = ta.sma(df_copy["close"], length=sma_slow_period)
        
        # Stochastic RSI
        if indicator_config.get("stoch_rsi", {}).get("enabled", True):
            try:
                stoch_period = indicator_config.get("stoch_rsi", {}).get("period", 14)
                stoch_k = indicator_config.get("stoch_rsi", {}).get("k", 3)
                stoch_d = indicator_config.get("stoch_rsi", {}).get("d", 3)
                
                stoch = ta.stoch(df_copy["high"], df_copy["low"], df_copy["close"], k=stoch_k, d=stoch_d, length=stoch_period)
                k_col = f"STOCHk_{stoch_period}_{stoch_k}_{stoch_d}"
                d_col = f"STOCHd_{stoch_period}_{stoch_k}_{stoch_d}"
                
                if k_col in stoch.columns:
                    df_copy["stoch_k"] = stoch[k_col]
                else:
                    logger.warning(f"Column {k_col} not found in stoch result")
                    
                if d_col in stoch.columns:    
                    df_copy["stoch_d"] = stoch[d_col]
                else:
                    logger.warning(f"Column {d_col} not found in stoch result")
            except Exception as e:
                logger.warning(f"Error calculating Stochastic: {e}")
        
        # ADX
        if indicator_config.get("adx", {}).get("enabled", True):
            adx_period = indicator_config.get("adx", {}).get("period", 14)
            adx = ta.adx(df_copy["high"], df_copy["low"], df_copy["close"], length=adx_period)
            
            df_copy["adx"] = adx["ADX_" + str(adx_period)]
            df_copy["plus_di"] = adx["DMP_" + str(adx_period)]
            df_copy["minus_di"] = adx["DMN_" + str(adx_period)]
        
        # Volume indicators
        if indicator_config.get("volume", {}).get("enabled", True):
            vol_sma_period = indicator_config.get("volume", {}).get("sma_period", 20)
            df_copy["volume_sma"] = ta.sma(df_copy["volume"], length=vol_sma_period)
            
            # On-Balance Volume
            df_copy["obv"] = ta.obv(df_copy["close"], df_copy["volume"])
            
            # Accumulation/Distribution Index
            df_copy["adi"] = ta.ad(df_copy["high"], df_copy["low"], df_copy["close"], df_copy["volume"])
        
        # Supertrend Indicator
        if indicator_config.get("supertrend", {}).get("enabled", True):
            atr_period = indicator_config.get("supertrend", {}).get("atr_period", 10)
            atr_multiplier = indicator_config.get("supertrend", {}).get("multiplier", 3.0)
            
            supertrend = calculate_supertrend(df_copy, atr_period, atr_multiplier)
            
            df_copy["supertrend"] = supertrend["supertrend"]
            df_copy["supertrend_direction"] = supertrend["direction"]
            df_copy["supertrend_upper"] = supertrend["upper_band"]
            df_copy["supertrend_lower"] = supertrend["lower_band"]
        
        # Ehlers' Indicators for use in Ehlers' strategies
        # if needed, implement Ehlers indicators from the provided file

        return df_copy
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        # Return original DataFrame if calculation fails
        return df


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> dict:
    """
    Calculate Supertrend indicator
    
    Args:
        df: DataFrame with OHLCV data
        period: ATR period
        multiplier: ATR multiplier
        
    Returns:
        dict: Dictionary with supertrend, direction, and band values
    """
    # Calculate ATR
    atr = ta.atr(df["high"], df["low"], df["close"], length=period)
    
    # Calculate basic upper and lower bands
    hl2 = (df["high"] + df["low"]) / 2
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend direction as 1 (uptrend)
    supertrend = pd.Series([np.nan] * len(df), index=df.index)
    direction = pd.Series([1] * len(df), index=df.index)
    
    # Calculate Supertrend using iteration (can't be fully vectorized due to dependencies)
    for i in range(1, len(df)):
        # Determine direction
        if df["close"].iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1  # Uptrend
        elif df["close"].iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1  # Downtrend
        else:
            direction.iloc[i] = direction.iloc[i-1]  # Continue previous trend
            
            # Adjust bands based on previous direction
            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i-1]
        
        # Calculate Supertrend value
        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
    
    return {
        "supertrend": supertrend,
        "direction": direction,
        "upper_band": upper_band,
        "lower_band": lower_band
    }


def calculate_signal(df: pd.DataFrame, strategy: str, config: dict) -> tuple:
    """
    Calculate trading signal based on indicators
    
    Args:
        df: DataFrame with indicator data
        strategy: Strategy name
        config: Strategy configuration
        
    Returns:
        tuple: (signal_strength, direction, parameters)
    """
    if df is None or len(df) < 2:
        return 0.0, "none", {}
    
    # Get the last row for current values
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Default return values
    signal_strength = 0.0
    direction = "none"
    parameters = {}
    
    # Dispatch to appropriate strategy
    if strategy == "simple_crossover":
        return calculate_simple_crossover_signal(df, config)
    elif strategy == "macd_crossover":
        return calculate_macd_crossover_signal(df, config)
    elif strategy == "rsi_divergence":
        return calculate_rsi_divergence_signal(df, config)
    elif strategy == "supertrend":
        return calculate_supertrend_signal(df, config)
    elif strategy == "ehlers_supertrend":
        return calculate_ehlers_supertrend_signal(df, config)
    else:
        logger.warning(f"Unknown strategy: {strategy}")
        return signal_strength, direction, parameters


def calculate_simple_crossover_signal(df: pd.DataFrame, config: dict) -> tuple:
    """
    Calculate signal based on moving average crossover
    
    Args:
        df: DataFrame with indicator data
        config: Strategy configuration
        
    Returns:
        tuple: (signal_strength, direction, parameters)
    """
    if "ema_fast" not in df.columns or "ema_slow" not in df.columns:
        return 0.0, "none", {}
    
    # Get the last two rows
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Check for crossover
    current_cross = current["ema_fast"] > current["ema_slow"]
    previous_cross = previous["ema_fast"] > previous["ema_slow"]
    
    signal_strength = 0.0
    direction = "none"
    parameters = {
        "atr": current.get("atr", 0.0),
        "current_price": current["close"]
    }
    
    # Bullish crossover (fast crosses above slow)
    if current_cross and not previous_cross:
        signal_strength = 1.0
        direction = "buy"
    
    # Bearish crossover (fast crosses below slow)
    elif not current_cross and previous_cross:
        signal_strength = 1.0
        direction = "sell"
    
    return signal_strength, direction, parameters


def calculate_macd_crossover_signal(df: pd.DataFrame, config: dict) -> tuple:
    """
    Calculate signal based on MACD crossover
    
    Args:
        df: DataFrame with indicator data
        config: Strategy configuration
        
    Returns:
        tuple: (signal_strength, direction, parameters)
    """
    if "macd" not in df.columns or "macd_signal" not in df.columns:
        return 0.0, "none", {}
    
    # Get the last two rows
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Check for MACD crossover
    current_cross = current["macd"] > current["macd_signal"]
    previous_cross = previous["macd"] > previous["macd_signal"]
    
    signal_strength = 0.0
    direction = "none"
    parameters = {
        "atr": current.get("atr", 0.0),
        "current_price": current["close"],
        "macd": current["macd"],
        "macd_signal": current["macd_signal"],
        "macd_hist": current.get("macd_hist", 0.0)
    }
    
    # Bullish crossover (MACD crosses above signal)
    if current_cross and not previous_cross:
        signal_strength = 1.0
        direction = "buy"
        
        # Adjust strength based on histogram
        if current.get("macd_hist", 0) > 0:
            signal_strength = min(1.5, abs(current["macd_hist"]) * 10)
    
    # Bearish crossover (MACD crosses below signal)
    elif not current_cross and previous_cross:
        signal_strength = 1.0
        direction = "sell"
        
        # Adjust strength based on histogram
        if current.get("macd_hist", 0) < 0:
            signal_strength = min(1.5, abs(current["macd_hist"]) * 10)
    
    return signal_strength, direction, parameters


def calculate_rsi_divergence_signal(df: pd.DataFrame, config: dict) -> tuple:
    """
    Calculate signal based on RSI divergence
    
    Args:
        df: DataFrame with indicator data
        config: Strategy configuration
        
    Returns:
        tuple: (signal_strength, direction, parameters)
    """
    if "rsi" not in df.columns or len(df) < 20:
        return 0.0, "none", {}
    
    # Config values
    lookback = config.get("lookback", 14)
    rsi_overbought = config.get("rsi_overbought", 70)
    rsi_oversold = config.get("rsi_oversold", 30)
    
    # Get the last value
    current = df.iloc[-1]
    
    # Initialize
    signal_strength = 0.0
    direction = "none"
    parameters = {
        "atr": current.get("atr", 0.0),
        "current_price": current["close"],
        "rsi": current["rsi"]
    }
    
    # Check for divergence within lookback period
    subset = df.iloc[-lookback:]
    
    # Find price highs/lows
    price_high_idx = subset["close"].idxmax()
    price_low_idx = subset["close"].idxmin()
    
    # Find RSI highs/lows
    rsi_high_idx = subset["rsi"].idxmax()
    rsi_low_idx = subset["rsi"].idxmin()
    
    # Bullish divergence (price makes lower low but RSI makes higher low)
    if price_low_idx > rsi_low_idx and subset.loc[price_low_idx, "rsi"] > subset.loc[rsi_low_idx, "rsi"] and current["rsi"] < rsi_oversold:
        signal_strength = 1.0
        direction = "buy"
        
        # Adjust strength based on RSI
        signal_strength = min(1.5, (rsi_oversold - current["rsi"]) / 10)
    
    # Bearish divergence (price makes higher high but RSI makes lower high)
    elif price_high_idx > rsi_high_idx and subset.loc[price_high_idx, "rsi"] < subset.loc[rsi_high_idx, "rsi"] and current["rsi"] > rsi_overbought:
        signal_strength = 1.0
        direction = "sell"
        
        # Adjust strength based on RSI
        signal_strength = min(1.5, (current["rsi"] - rsi_overbought) / 10)
    
    return signal_strength, direction, parameters


def calculate_supertrend_signal(df: pd.DataFrame, config: dict) -> tuple:
    """
    Calculate signal based on Supertrend indicator
    
    Args:
        df: DataFrame with indicator data
        config: Strategy configuration
        
    Returns:
        tuple: (signal_strength, direction, parameters)
    """
    if "supertrend_direction" not in df.columns:
        return 0.0, "none", {}
    
    # Get the last two rows
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Check for trend change
    current_direction = current["supertrend_direction"]
    previous_direction = previous["supertrend_direction"]
    
    signal_strength = 0.0
    direction = "none"
    parameters = {
        "atr": current.get("atr", 0.0),
        "current_price": current["close"],
        "supertrend": current["supertrend"],
        "supertrend_upper": current["supertrend_upper"],
        "supertrend_lower": current["supertrend_lower"]
    }
    
    # Trend change detection
    if current_direction != previous_direction:
        # Bullish trend change
        if current_direction == 1:
            signal_strength = 1.0
            direction = "buy"
        # Bearish trend change
        elif current_direction == -1:
            signal_strength = 1.0
            direction = "sell"
    
    return signal_strength, direction, parameters


def calculate_ehlers_supertrend_signal(df: pd.DataFrame, config: dict) -> tuple:
    """
    Calculate signal based on Ehlers Supertrend strategy
    
    Args:
        df: DataFrame with indicator data
        config: Strategy configuration
        
    Returns:
        tuple: (signal_strength, direction, parameters)
    """
    # For now, use regular supertrend logic until Ehlers' implementation is complete
    if "supertrend_direction" not in df.columns:
        return 0.0, "none", {}
    
    # Get the last two rows
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Check for trend change
    current_direction = current["supertrend_direction"]
    previous_direction = previous["supertrend_direction"]
    
    signal_strength = 0.0
    direction = "none"
    parameters = {
        "atr": current.get("atr", 0.0),
        "current_price": current["close"],
        "supertrend": current["supertrend"],
        "supertrend_upper": current.get("supertrend_upper", 0),
        "supertrend_lower": current.get("supertrend_lower", 0)
    }
    
    # Trend change from down to up (buy signal)
    if current_direction == 1 and previous_direction == -1:
        signal_strength = 1.0
        direction = "long"
        
        # Adjust strength based on ATR
        if "atr" in current:
            volatility_factor = current["atr"] / current["close"] * 100  # ATR as % of price
            signal_strength = min(1.5, volatility_factor / 2)  # Scale up to 1.5x based on volatility
    
    # Trend change from up to down (sell signal)
    elif current_direction == -1 and previous_direction == 1:
        signal_strength = 1.0
        direction = "short"
        
        # Adjust strength based on ATR
        if "atr" in current:
            volatility_factor = current["atr"] / current["close"] * 100  # ATR as % of price
            signal_strength = min(1.5, volatility_factor / 2)  # Scale up to 1.5x based on volatility
    
    return signal_strength, direction, parameters