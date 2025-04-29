"""
Ehlers Indicators Module

This module implements various John Ehlers indicators:
- Supertrend
- Center of Gravity Oscillator
- Sine Wave Indicator
- Roofing Filter
- Fisher Transform
- Instantaneous Trendline
- Cyber Cycle
- Adaptive RSI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


def calculate_ehlers_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculate Ehlers Supertrend indicator.
    
    Args:
        df: DataFrame with OHLCV data
        period: Lookback period for ATR calculation
        multiplier: Multiplier for the ATR value
        
    Returns:
        pd.DataFrame: DataFrame with added Supertrend columns
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Calculate ATR
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = np.abs(df["high"] - df["close"].shift(1))
    df["low_close"] = np.abs(df["low"] - df["close"].shift(1))
    df["true_range"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["atr"] = df["true_range"].rolling(period).mean()
    
    # Calculate Smoothed ATR using Ehlers' smoothing technique
    alpha = 2.0 / (period + 1)
    df["smoothed_atr"] = df["true_range"].copy()
    
    for i in range(1, len(df)):
        df.loc[df.index[i], "smoothed_atr"] = alpha * df.loc[df.index[i], "true_range"] + \
                                              (1 - alpha) * df.loc[df.index[i-1], "smoothed_atr"]
    
    # Calculate Supertrend
    df["basic_upper_band"] = (df["high"] + df["low"]) / 2 + multiplier * df["smoothed_atr"]
    df["basic_lower_band"] = (df["high"] + df["low"]) / 2 - multiplier * df["smoothed_atr"]
    
    # Calculate Ehlers Supertrend
    df["ehlers_supertrend_upper"] = df["basic_upper_band"].copy()
    df["ehlers_supertrend_lower"] = df["basic_lower_band"].copy()
    df["ehlers_supertrend"] = np.nan
    df["ehlers_supertrend_direction"] = np.nan
    
    # Initialize supertrend direction
    if len(df) > period:
        df.loc[df.index[period], "ehlers_supertrend_direction"] = 1 if df.loc[df.index[period], "close"] > \
            df.loc[df.index[period], "basic_upper_band"] else -1
    
    # Calculate Supertrend for the rest of the data
    for i in range(period + 1, len(df)):
        curr_idx = df.index[i]
        prev_idx = df.index[i-1]
        
        # Adjust upper band
        if (df.loc[prev_idx, "ehlers_supertrend_upper"] > df.loc[prev_idx, "basic_upper_band"]) and \
           (df.loc[curr_idx, "basic_upper_band"] > df.loc[prev_idx, "ehlers_supertrend_upper"]):
            df.loc[curr_idx, "ehlers_supertrend_upper"] = df.loc[curr_idx, "basic_upper_band"]
        elif df.loc[prev_idx, "ehlers_supertrend_upper"] > df.loc[prev_idx, "basic_upper_band"]:
            df.loc[curr_idx, "ehlers_supertrend_upper"] = df.loc[prev_idx, "ehlers_supertrend_upper"]
        else:
            df.loc[curr_idx, "ehlers_supertrend_upper"] = df.loc[curr_idx, "basic_upper_band"]
        
        # Adjust lower band
        if (df.loc[prev_idx, "ehlers_supertrend_lower"] < df.loc[prev_idx, "basic_lower_band"]) and \
           (df.loc[curr_idx, "basic_lower_band"] < df.loc[prev_idx, "ehlers_supertrend_lower"]):
            df.loc[curr_idx, "ehlers_supertrend_lower"] = df.loc[curr_idx, "basic_lower_band"]
        elif df.loc[prev_idx, "ehlers_supertrend_lower"] < df.loc[prev_idx, "basic_lower_band"]:
            df.loc[curr_idx, "ehlers_supertrend_lower"] = df.loc[prev_idx, "ehlers_supertrend_lower"]
        else:
            df.loc[curr_idx, "ehlers_supertrend_lower"] = df.loc[curr_idx, "basic_lower_band"]
        
        # Determine trend direction
        if df.loc[prev_idx, "ehlers_supertrend_direction"] == 1:
            if df.loc[curr_idx, "close"] < df.loc[curr_idx, "ehlers_supertrend_lower"]:
                df.loc[curr_idx, "ehlers_supertrend_direction"] = -1
                df.loc[curr_idx, "ehlers_supertrend"] = df.loc[curr_idx, "ehlers_supertrend_upper"]
            else:
                df.loc[curr_idx, "ehlers_supertrend_direction"] = 1
                df.loc[curr_idx, "ehlers_supertrend"] = df.loc[curr_idx, "ehlers_supertrend_lower"]
        else:
            if df.loc[curr_idx, "close"] > df.loc[curr_idx, "ehlers_supertrend_upper"]:
                df.loc[curr_idx, "ehlers_supertrend_direction"] = 1
                df.loc[curr_idx, "ehlers_supertrend"] = df.loc[curr_idx, "ehlers_supertrend_lower"]
            else:
                df.loc[curr_idx, "ehlers_supertrend_direction"] = -1
                df.loc[curr_idx, "ehlers_supertrend"] = df.loc[curr_idx, "ehlers_supertrend_upper"]
    
    # Drop intermediate columns
    df.drop(["high_low", "high_close", "low_close", "true_range", "basic_upper_band", "basic_lower_band"], 
            axis=1, inplace=True)
    
    return df


def calculate_center_of_gravity(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    Calculate Ehlers Center of Gravity (COG) Oscillator.
    
    Args:
        df: DataFrame with OHLCV data
        period: Lookback period for calculation
        
    Returns:
        pd.DataFrame: DataFrame with added COG oscillator
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Use typical price as input
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    
    # Calculate numerator and denominator of COG
    df["cog_num"] = 0.0
    df["cog_den"] = 0.0
    
    for i in range(period, len(df)):
        num_sum = 0.0
        den_sum = 0.0
        
        for j in range(period):
            price = df["typical_price"].iloc[i-j]
            weight = j + 1
            num_sum += price * weight
            den_sum += price
        
        df["cog_num"].iloc[i] = num_sum
        df["cog_den"].iloc[i] = den_sum
    
    # Calculate COG
    df["cog"] = -df["cog_num"] / df["cog_den"] + (period + 1) / 2
    
    # Calculate trigger line (signal)
    df["cog_signal"] = df["cog"].rolling(3).mean()
    
    # Drop intermediate columns
    df.drop(["cog_num", "cog_den", "typical_price"], axis=1, inplace=True)
    
    return df


def calculate_roofing_filter(df: pd.DataFrame, period: int = 80, hp_period: int = 48) -> pd.DataFrame:
    """
    Calculate Ehlers Roofing Filter.
    
    Args:
        df: DataFrame with OHLCV data
        period: Lookback period for calculation
        hp_period: High pass filter period
        
    Returns:
        pd.DataFrame: DataFrame with added Roofing Filter
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Use typical price as input
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    
    # Calculate alpha1 and alpha2
    alpha1 = (np.cos(2 * np.pi / period) + np.sin(2 * np.pi / period) - 1) / np.cos(2 * np.pi / period)
    alpha2 = (np.cos(2 * np.pi / hp_period) + np.sin(2 * np.pi / hp_period) - 1) / np.cos(2 * np.pi / hp_period)
    
    # Initialize columns
    df["hp_filter"] = 0.0
    df["bp_filter"] = 0.0
    df["roofing_filter"] = 0.0
    
    # Apply high pass filter
    for i in range(2, len(df)):
        df.loc[df.index[i], "hp_filter"] = (1 - alpha1/2) * (1 - alpha1/2) * \
                                          (df.loc[df.index[i], "typical_price"] - 2*df.loc[df.index[i-1], "typical_price"] + \
                                           df.loc[df.index[i-2], "typical_price"]) + \
                                          2 * (1 - alpha1) * df.loc[df.index[i-1], "hp_filter"] - \
                                          (1 - alpha1) * (1 - alpha1) * df.loc[df.index[i-2], "hp_filter"]
    
    # Apply band pass filter (roofing filter)
    for i in range(2, len(df)):
        df.loc[df.index[i], "bp_filter"] = (1 - alpha2/2) * (1 - alpha2/2) * \
                                          (df.loc[df.index[i], "hp_filter"] - 2*df.loc[df.index[i-1], "hp_filter"] + \
                                           df.loc[df.index[i-2], "hp_filter"]) + \
                                          2 * (1 - alpha2) * df.loc[df.index[i-1], "bp_filter"] - \
                                          (1 - alpha2) * (1 - alpha2) * df.loc[df.index[i-2], "bp_filter"]
    
    # Normalize roofing filter
    df["roofing_filter"] = df["bp_filter"] / df["bp_filter"].rolling(period).std()
    
    # Drop intermediate columns
    df.drop(["typical_price", "hp_filter", "bp_filter"], axis=1, inplace=True)
    
    return df


def calculate_instantaneous_trendline(df: pd.DataFrame, alpha: float = 0.07) -> pd.DataFrame:
    """
    Calculate Ehlers Instantaneous Trendline.
    
    Args:
        df: DataFrame with OHLCV data
        alpha: Smoothing factor (0 to 1)
        
    Returns:
        pd.DataFrame: DataFrame with added Instantaneous Trendline
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Use median price as input
    df["median_price"] = (df["high"] + df["low"]) / 2
    
    # Initialize IT column
    df["it_value"] = df["median_price"].copy()
    
    # Calculate Instantaneous Trendline
    for i in range(2, len(df)):
        df.loc[df.index[i], "it_value"] = (alpha - alpha*alpha/4) * df.loc[df.index[i], "median_price"] + \
                                      0.5 * alpha*alpha * df.loc[df.index[i-1], "median_price"] - \
                                      (alpha - 0.75*alpha*alpha) * df.loc[df.index[i-2], "median_price"] + \
                                      2 * (1 - alpha) * df.loc[df.index[i-1], "it_value"] - \
                                      (1 - alpha) * (1 - alpha) * df.loc[df.index[i-2], "it_value"]
    
    # Calculate trigger line
    df["it_trigger"] = df["it_value"].rolling(3).mean()
    
    # Calculate trend direction
    df["it_direction"] = np.where(df["it_value"] > df["it_trigger"], 1, 
                                 np.where(df["it_value"] < df["it_trigger"], -1, 0))
    
    # Drop intermediate columns
    df.drop(["median_price"], axis=1, inplace=True)
    
    return df


def identify_market_ranges(df: pd.DataFrame, volatility_period: int = 20, 
                          threshold: float = 1.5) -> pd.DataFrame:
    """
    Identify market ranges using volatility metrics.
    
    Args:
        df: DataFrame with OHLCV data
        volatility_period: Period for volatility calculation
        threshold: Threshold for range identification
        
    Returns:
        pd.DataFrame: DataFrame with added range identification
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Calculate price movement volatility
    df["price_range"] = df["high"] - df["low"]
    df["price_range_ma"] = df["price_range"].rolling(volatility_period).mean()
    df["price_range_std"] = df["price_range"].rolling(volatility_period).std()
    
    # Calculate directional movement
    df["directional_move"] = abs(df["close"] - df["close"].shift(1))
    df["directional_move_ma"] = df["directional_move"].rolling(volatility_period).mean()
    
    # Calculate efficiency ratio (directional movement / total movement)
    df["efficiency_ratio"] = df["directional_move_ma"] / df["price_range_ma"]
    
    # Identify ranging market when efficiency ratio is low
    df["is_ranging"] = np.where(df["efficiency_ratio"] < 1/threshold, 1, 0)
    
    # Identify volatile ranging market
    df["range_volatility"] = df["price_range"] / df["price_range_ma"]
    df["is_volatile_range"] = np.where((df["is_ranging"] == 1) & (df["range_volatility"] > threshold), 1, 0)
    
    # Drop intermediate columns
    df.drop(["price_range", "directional_move"], axis=1, inplace=True)
    
    return df


def calculate_dual_supertrend(df: pd.DataFrame, period1: int = 10, multiplier1: float = 2.0, 
                             period2: int = 20, multiplier2: float = 4.0) -> pd.DataFrame:
    """
    Calculate Dual Ehlers Supertrend indicator.
    
    Args:
        df: DataFrame with OHLCV data
        period1: Period for the first Supertrend
        multiplier1: Multiplier for the first Supertrend
        period2: Period for the second Supertrend
        multiplier2: Multiplier for the second Supertrend
        
    Returns:
        pd.DataFrame: DataFrame with added Dual Supertrend columns
    """
    # Calculate the first Supertrend
    df = calculate_ehlers_supertrend(df, period=period1, multiplier=multiplier1)
    
    # Rename first Supertrend columns
    df.rename(columns={
        "ehlers_supertrend": "supertrend1",
        "ehlers_supertrend_upper": "supertrend1_upper",
        "ehlers_supertrend_lower": "supertrend1_lower",
        "ehlers_supertrend_direction": "supertrend1_direction"
    }, inplace=True)
    
    # Calculate the second Supertrend
    df = calculate_ehlers_supertrend(df, period=period2, multiplier=multiplier2)
    
    # Rename second Supertrend columns
    df.rename(columns={
        "ehlers_supertrend": "supertrend2",
        "ehlers_supertrend_upper": "supertrend2_upper",
        "ehlers_supertrend_lower": "supertrend2_lower",
        "ehlers_supertrend_direction": "supertrend2_direction"
    }, inplace=True)
    
    # Create Dual Supertrend Signal
    df["dual_supertrend_signal"] = np.where(
        (df["supertrend1_direction"] == 1) & (df["supertrend2_direction"] == 1), 1,
        np.where((df["supertrend1_direction"] == -1) & (df["supertrend2_direction"] == -1), -1, 0)
    )
    
    return df