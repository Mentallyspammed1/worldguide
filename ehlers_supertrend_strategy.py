"""
Dual Ehlers Supertrend with Ranges Strategy

This strategy combines:
1. Dual Ehlers Supertrend for trend identification
2. Market range detection for adaptive trading
3. Limit orders for better entry prices
4. Trailing stop-loss in profit for risk management

The strategy employs different approaches in trending vs ranging markets:
- In trending markets: follows the trend with Dual Supertrend
- In ranging markets: trades range reversals with limit orders
"""

import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Union, Any

from ehlers_indicators import (
    calculate_dual_supertrend,
    identify_market_ranges,
    calculate_instantaneous_trendline,
    calculate_center_of_gravity
)


def prepare_ehlers_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Prepare all required Ehlers indicators for the strategy.
    
    Args:
        df: DataFrame with OHLCV data
        config: Strategy configuration
        
    Returns:
        pd.DataFrame: DataFrame with added indicators
    """
    if df.empty or len(df) < 50:
        return df
    
    # Get configuration parameters
    st_period1 = config.get("supertrend_period1", 10)
    st_mult1 = config.get("supertrend_multiplier1", 2.0)
    st_period2 = config.get("supertrend_period2", 20)
    st_mult2 = config.get("supertrend_multiplier2", 4.0)
    
    volatility_period = config.get("volatility_period", 20)
    volatility_threshold = config.get("volatility_threshold", 1.5)
    
    # Calculate Dual Supertrend
    df = calculate_dual_supertrend(
        df, 
        period1=st_period1, 
        multiplier1=st_mult1,
        period2=st_period2, 
        multiplier2=st_mult2
    )
    
    # Identify market ranges
    df = identify_market_ranges(
        df,
        volatility_period=volatility_period,
        threshold=volatility_threshold
    )
    
    # Calculate Instantaneous Trendline
    df = calculate_instantaneous_trendline(df, alpha=config.get("it_alpha", 0.07))
    
    # Calculate Center of Gravity Oscillator
    df = calculate_center_of_gravity(df, period=config.get("cog_period", 10))
    
    return df


def calculate_ehlers_supertrend_signal(
    df: pd.DataFrame, 
    config: Dict
) -> Tuple[float, str, Dict]:
    """
    Calculate trading signal based on Dual Ehlers Supertrend with Ranges.
    
    Args:
        df: DataFrame with indicator data
        config: Strategy configuration
        
    Returns:
        Tuple[float, str, Dict]: Signal strength, direction, and additional trade parameters
    """
    if df.empty or len(df) < 50:
        return 0.0, "", {}
    
    # Prepare indicators if not already present
    required_indicators = ["dual_supertrend_signal", "is_ranging", "it_direction", "cog"]
    missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
    
    if missing_indicators:
        df = prepare_ehlers_indicators(df, config)
    
    # Get the current market state
    current_idx = df.index[-1]
    is_ranging = df.loc[current_idx, "is_ranging"] == 1
    is_volatile_range = df.loc[current_idx, "is_volatile_range"] == 1
    dual_signal = df.loc[current_idx, "dual_supertrend_signal"]
    it_direction = df.loc[current_idx, "it_direction"]
    cog = df.loc[current_idx, "cog"]
    cog_signal = df.loc[current_idx, "cog_signal"]
    
    # Current price and ATR
    current_price = df.loc[current_idx, "close"]
    current_atr = df.loc[current_idx, "atr"]
    
    # Get configuration values
    range_reversal_threshold = config.get("range_reversal_threshold", 0.8)
    trend_strength_threshold = config.get("trend_strength_threshold", 0.5)
    limit_order_offset = config.get("limit_order_offset", 0.5)  # As multiple of ATR
    
    # Default result values
    signal_strength = 0.0
    direction = ""
    trade_params = {}
    
    # Trading logic based on market conditions
    if is_ranging:
        # Ranging market strategy: Reversal trading with limit orders
        if is_volatile_range:
            # More volatile range requires stronger signals
            # Use COG oscillator for reversal signals in ranges
            cog_threshold = range_reversal_threshold
            
            if cog < -cog_threshold and cog < cog_signal:
                # Oversold condition, potential long signal
                signal_strength = min(1.0, abs(cog) / 2.0)
                direction = "long"
                
                # Calculate limit order price below current price
                limit_price = current_price - (current_atr * limit_order_offset)
                
                # Calculate stop loss and take profit levels
                stop_loss = limit_price - (current_atr * 1.5)
                take_profit = limit_price + (current_atr * 3.0)
                
                # Determine range boundaries
                upper_boundary = df["high"].rolling(config.get("volatility_period", 20)).max().iloc[-1]
                lower_boundary = df["low"].rolling(config.get("volatility_period", 20)).min().iloc[-1]
                
                take_profit = min(take_profit, upper_boundary)
                
                # Trailing stop parameters
                trailing_stop = {
                    "enabled": True,
                    "activation_pct": 1.0,  # Activate at 1% profit
                    "trail_pct": 0.5        # Trail by 0.5%
                }
                
                # Set trade parameters for limit order
                trade_params = {
                    "type": "limit",
                    "entry_price": limit_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward_ratio": 2.0,
                    "trailing_stop": trailing_stop,
                    "strategy": "ehlers_supertrend_range",
                    "time_limit": 12,  # hours
                    "range_boundaries": {
                        "upper": upper_boundary,
                        "lower": lower_boundary
                    }
                }
                
            elif cog > cog_threshold and cog > cog_signal:
                # Overbought condition, potential short signal
                signal_strength = min(1.0, abs(cog) / 2.0)
                direction = "short"
                
                # Calculate limit order price above current price
                limit_price = current_price + (current_atr * limit_order_offset)
                
                # Calculate stop loss and take profit levels
                stop_loss = limit_price + (current_atr * 1.5)
                take_profit = limit_price - (current_atr * 3.0)
                
                # Determine range boundaries
                upper_boundary = df["high"].rolling(config.get("volatility_period", 20)).max().iloc[-1]
                lower_boundary = df["low"].rolling(config.get("volatility_period", 20)).min().iloc[-1]
                
                take_profit = max(take_profit, lower_boundary)
                
                # Trailing stop parameters
                trailing_stop = {
                    "enabled": True,
                    "activation_pct": 1.0,  # Activate at 1% profit
                    "trail_pct": 0.5        # Trail by 0.5%
                }
                
                # Set trade parameters for limit order
                trade_params = {
                    "type": "limit",
                    "entry_price": limit_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward_ratio": 2.0,
                    "trailing_stop": trailing_stop,
                    "strategy": "ehlers_supertrend_range",
                    "time_limit": 12,  # hours
                    "range_boundaries": {
                        "upper": upper_boundary,
                        "lower": lower_boundary
                    }
                }
            
        else:
            # Regular range trading with narrower risk parameters
            # Use more conservative limit orders with tighter stops
            if cog < -range_reversal_threshold and cog < cog_signal:
                # Oversold condition, potential long signal
                signal_strength = min(0.8, abs(cog) / 2.0)  # Less strength in regular ranges
                direction = "long"
                
                # Calculate limit order price below current price
                limit_price = current_price - (current_atr * (limit_order_offset * 0.7))  # Smaller offset
                
                # Calculate stop loss and take profit levels
                stop_loss = limit_price - (current_atr * 1.0)  # Tighter stop loss
                take_profit = limit_price + (current_atr * 2.0)  # Smaller target
                
                # Determine range boundaries
                upper_boundary = df["high"].rolling(config.get("volatility_period", 20)).max().iloc[-1]
                lower_boundary = df["low"].rolling(config.get("volatility_period", 20)).min().iloc[-1]
                
                take_profit = min(take_profit, upper_boundary)
                
                # Trailing stop parameters
                trailing_stop = {
                    "enabled": True,
                    "activation_pct": 0.7,  # Activate earlier
                    "trail_pct": 0.3        # Tighter trail
                }
                
                # Set trade parameters for limit order
                trade_params = {
                    "type": "limit",
                    "entry_price": limit_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward_ratio": 2.0,
                    "trailing_stop": trailing_stop,
                    "strategy": "ehlers_supertrend_range",
                    "time_limit": 8,  # hours - shorter time limit in regular ranges
                    "range_boundaries": {
                        "upper": upper_boundary,
                        "lower": lower_boundary
                    }
                }
                
            elif cog > range_reversal_threshold and cog > cog_signal:
                # Overbought condition, potential short signal
                signal_strength = min(0.8, abs(cog) / 2.0)  # Less strength in regular ranges
                direction = "short"
                
                # Calculate limit order price above current price
                limit_price = current_price + (current_atr * (limit_order_offset * 0.7))  # Smaller offset
                
                # Calculate stop loss and take profit levels
                stop_loss = limit_price + (current_atr * 1.0)  # Tighter stop loss
                take_profit = limit_price - (current_atr * 2.0)  # Smaller target
                
                # Determine range boundaries
                upper_boundary = df["high"].rolling(config.get("volatility_period", 20)).max().iloc[-1]
                lower_boundary = df["low"].rolling(config.get("volatility_period", 20)).min().iloc[-1]
                
                take_profit = max(take_profit, lower_boundary)
                
                # Trailing stop parameters
                trailing_stop = {
                    "enabled": True,
                    "activation_pct": 0.7,  # Activate earlier
                    "trail_pct": 0.3        # Tighter trail
                }
                
                # Set trade parameters for limit order
                trade_params = {
                    "type": "limit",
                    "entry_price": limit_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward_ratio": 2.0,
                    "trailing_stop": trailing_stop,
                    "strategy": "ehlers_supertrend_range",
                    "time_limit": 8,  # hours - shorter time limit in regular ranges
                    "range_boundaries": {
                        "upper": upper_boundary,
                        "lower": lower_boundary
                    }
                }
    
    else:
        # Trending market strategy: Follow the trend with market orders
        # Check for agreement between Dual Supertrend and Instantaneous Trendline
        trend_alignment = (dual_signal == 1 and it_direction == 1) or (dual_signal == -1 and it_direction == -1)
        
        if trend_alignment:
            if dual_signal == 1:
                # Strong long trend signal
                signal_strength = trend_strength_threshold + 0.3  # Stronger signal in trends
                direction = "long"
                
                # Calculate stop loss and take profit levels
                stop_loss = current_price - (current_atr * 2.5)
                take_profit = current_price + (current_atr * 5.0)
                
                # Trailing stop parameters
                trailing_stop = {
                    "enabled": True,
                    "activation_pct": 1.5,  # Activate at higher profit in trends
                    "trail_pct": 1.0        # Wider trail in trends
                }
                
                # Additional settings for trend trades
                trade_params = {
                    "type": "market",
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward_ratio": 2.0,
                    "trailing_stop": trailing_stop,
                    "strategy": "ehlers_supertrend_trend",
                    "supertrend1_level": df.loc[current_idx, "supertrend1_lower"],
                    "supertrend2_level": df.loc[current_idx, "supertrend2_lower"],
                    "add_on_levels": [
                        # Scale-in levels for partial position additions
                        current_price * 1.01,  # Add 25% more position at 1% above entry
                        current_price * 1.02   # Add 25% more position at 2% above entry
                    ],
                    "add_on_sizes": [0.25, 0.25]  # Add 25% position size at each level
                }
                
            elif dual_signal == -1:
                # Strong short trend signal
                signal_strength = trend_strength_threshold + 0.3  # Stronger signal in trends
                direction = "short"
                
                # Calculate stop loss and take profit levels
                stop_loss = current_price + (current_atr * 2.5)
                take_profit = current_price - (current_atr * 5.0)
                
                # Trailing stop parameters
                trailing_stop = {
                    "enabled": True,
                    "activation_pct": 1.5,  # Activate at higher profit in trends
                    "trail_pct": 1.0        # Wider trail in trends
                }
                
                # Additional settings for trend trades
                trade_params = {
                    "type": "market",
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward_ratio": 2.0,
                    "trailing_stop": trailing_stop,
                    "strategy": "ehlers_supertrend_trend",
                    "supertrend1_level": df.loc[current_idx, "supertrend1_upper"],
                    "supertrend2_level": df.loc[current_idx, "supertrend2_upper"],
                    "add_on_levels": [
                        # Scale-in levels for partial position additions
                        current_price * 0.99,  # Add 25% more position at 1% below entry
                        current_price * 0.98   # Add 25% more position at 2% below entry
                    ],
                    "add_on_sizes": [0.25, 0.25]  # Add 25% position size at each level
                }
    
    # Add risk management parameters
    if trade_params:
        trade_params.update({
            "risk_per_trade_pct": config.get("risk_per_trade_pct", 1.0),
            "entry_time": int(time.time() * 1000),  # Current time in ms
            "partial_take_profits": [
                # Take profit ladder for scaling out of positions
                {
                    "percentage": 33,  # Take 33% of position off at first target
                    "level": current_price * (1.02 if direction == "long" else 0.98)  # 2% movement
                },
                {
                    "percentage": 33,  # Take another 33% off at second target
                    "level": current_price * (1.04 if direction == "long" else 0.96)  # 4% movement
                }
                # Remaining 34% will exit at final take profit or trailing stop
            ]
        })
    
    return signal_strength, direction, trade_params


def calculate_ehlers_supertrend_exit_signal(
    df: pd.DataFrame,
    position: Dict,
    config: Dict
) -> bool:
    """
    Calculate exit signal for Dual Ehlers Supertrend with Ranges strategy.
    
    Args:
        df: DataFrame with indicator data
        position: Current position information
        config: Strategy configuration
        
    Returns:
        bool: True if should exit, False otherwise
    """
    if df.empty or len(df) < 50:
        return False
    
    # Prepare indicators if not already present
    required_indicators = ["dual_supertrend_signal", "is_ranging", "it_direction", "cog"]
    missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
    
    if missing_indicators:
        df = prepare_ehlers_indicators(df, config)
    
    # Get current position details
    position_side = position.get("side")
    entry_price = position.get("entry_price", 0)
    position_type = position.get("parameters", {}).get("type", "market")
    strategy_type = position.get("parameters", {}).get("strategy", "")
    
    # Get current market state
    current_idx = df.index[-1]
    current_price = df.loc[current_idx, "close"]
    is_ranging = df.loc[current_idx, "is_ranging"] == 1
    dual_signal = df.loc[current_idx, "dual_supertrend_signal"]
    it_direction = df.loc[current_idx, "it_direction"]
    
    # Exit criteria depending on strategy and market condition
    if "range" in strategy_type:
        # Range strategy exit rules
        if position_side == "long":
            # Exit long range trade if:
            # 1. Market becomes trending downward
            # 2. COG becomes extremely overbought
            # 3. Price exceeds range upper boundary
            
            if not is_ranging and (dual_signal == -1 or it_direction == -1):
                # Market changed from range to trending downward
                return True
            
            if df.loc[current_idx, "cog"] > config.get("range_exit_threshold", 1.5):
                # Extreme overbought condition in COG
                return True
            
            # Check if price exceeds range boundary
            range_boundaries = position.get("parameters", {}).get("range_boundaries", {})
            upper_boundary = range_boundaries.get("upper")
            
            if upper_boundary and current_price > upper_boundary:
                # Price exceeded upper boundary
                return True
            
        elif position_side == "short":
            # Exit short range trade if:
            # 1. Market becomes trending upward
            # 2. COG becomes extremely oversold
            # 3. Price falls below range lower boundary
            
            if not is_ranging and (dual_signal == 1 or it_direction == 1):
                # Market changed from range to trending upward
                return True
            
            if df.loc[current_idx, "cog"] < -config.get("range_exit_threshold", 1.5):
                # Extreme oversold condition in COG
                return True
            
            # Check if price exceeds range boundary
            range_boundaries = position.get("parameters", {}).get("range_boundaries", {})
            lower_boundary = range_boundaries.get("lower")
            
            if lower_boundary and current_price < lower_boundary:
                # Price exceeded lower boundary
                return True
    
    else:
        # Trend strategy exit rules
        if position_side == "long":
            # Exit long trend trade if:
            # 1. Dual Supertrend signal turns bearish
            # 2. Instantaneous Trendline direction turns negative
            
            if dual_signal == -1:
                # Supertrend turned bearish
                return True
            
            if it_direction == -1 and df.loc[current_idx-1, "it_direction"] == 1:
                # Instantaneous Trendline changed direction
                return True
            
        elif position_side == "short":
            # Exit short trend trade if:
            # 1. Dual Supertrend signal turns bullish
            # 2. Instantaneous Trendline direction turns positive
            
            if dual_signal == 1:
                # Supertrend turned bullish
                return True
            
            if it_direction == 1 and df.loc[current_idx-1, "it_direction"] == -1:
                # Instantaneous Trendline changed direction
                return True
    
    # Time-based exit for limit orders
    if position_type == "limit" and "entry_time" in position:
        entry_time = position.get("entry_time", 0)
        time_limit_hours = position.get("parameters", {}).get("time_limit", 12)
        current_time = time.time() * 1000  # Current time in ms
        
        if (current_time - entry_time) / (60 * 60 * 1000) > time_limit_hours:
            # Exceeded time limit for limit order
            return True
    
    return False