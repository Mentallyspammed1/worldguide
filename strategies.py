"""
Advanced Trading Strategies Module

This module implements several advanced trading strategies with entry and exit rules:
1. Ehlers Supertrend Strategy - Based on John Ehlers' Supertrend with market adaptation
2. Momentum Divergence Strategy - Uses RSI/MACD divergence with volatility filters 
3. Multi-Timeframe Trend Strategy - Combines multiple timeframes for trend confirmation
4. Support/Resistance Breakout Strategy - Detects breakouts with volume confirmation

Each strategy includes comprehensive risk management parameters.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

from indicators import calculate_supertrend_signal, calculate_signal

# Configure logger
logger = logging.getLogger("strategies")


def evaluate_strategies(
    df: pd.DataFrame,
    higher_tf_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None
) -> Dict:
    """
    Evaluate all enabled strategies and return a consolidated signal
    
    Args:
        df: DataFrame with indicator data
        higher_tf_df: Optional higher timeframe DataFrame
        config: Strategy configuration
        
    Returns:
        Dict: Consolidated results with signals and weights
    """
    if config is None:
        config = {}
    
    strategies_config = config.get("strategies", {})
    enabled_strategies = [
        s for s, conf in strategies_config.items() 
        if conf.get("enabled", False)
    ]
    
    if not enabled_strategies:
        # Fall back to the active strategy if no explicit strategy configurations
        active_strategy = config.get("strategy", {}).get("active", "ehlers_supertrend")
        enabled_strategies = [active_strategy]
    
    results = {}
    weights = {}
    total_weight = 0
    
    # Evaluate each enabled strategy
    for strategy_name in enabled_strategies:
        strategy_config = strategies_config.get(strategy_name, {})
        weight = strategy_config.get("weight", 1.0)
        
        # Get the signal from the appropriate strategy function
        if strategy_name == "ehlers_supertrend":
            signal_strength, direction, params = calculate_ehlers_supertrend_strategy(
                df, strategy_config
            )
        elif strategy_name == "momentum_divergence":
            signal_strength, direction, params = calculate_momentum_divergence_strategy(
                df, higher_tf_df, strategy_config
            )
        elif strategy_name == "multi_timeframe_trend":
            signal_strength, direction, params = calculate_multi_timeframe_trend_strategy(
                df, higher_tf_df, strategy_config
            )
        elif strategy_name == "support_resistance_breakout":
            signal_strength, direction, params = calculate_support_resistance_breakout_strategy(
                df, higher_tf_df, strategy_config
            )
        else:
            logger.warning(f"Unknown strategy: {strategy_name}, skipping")
            continue
        
        # Store results with weight
        results[strategy_name] = {
            "signal_strength": signal_strength,
            "direction": direction,
            "params": params,
            "weight": weight
        }
        weights[strategy_name] = weight
        total_weight += weight
    
    # Calculate weighted signals
    buy_signal = 0.0
    sell_signal = 0.0
    
    for strategy_name, result in results.items():
        if result["direction"] == "buy":
            buy_signal += result["signal_strength"] * result["weight"]
        elif result["direction"] == "sell":
            sell_signal += result["signal_strength"] * result["weight"]
    
    # Normalize by total weight
    if total_weight > 0:
        buy_signal /= total_weight
        sell_signal /= total_weight
    
    # Determine final direction and strength
    final_direction = "none"
    final_strength = 0.0
    
    if buy_signal > sell_signal and buy_signal >= config.get("entry_threshold", 0.5):
        final_direction = "buy"
        final_strength = buy_signal
    elif sell_signal > buy_signal and sell_signal >= config.get("entry_threshold", 0.5):
        final_direction = "sell"
        final_strength = sell_signal
    
    # Collect all parameters from all strategies
    all_params = {}
    for result in results.values():
        all_params.update(result["params"])
    
    return {
        "strategies": results,
        "consolidated_signal": {
            "direction": final_direction,
            "strength": final_strength,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "params": all_params
        }
    }


def calculate_ehlers_supertrend_strategy(
    df: pd.DataFrame, 
    config: Optional[Dict] = None
) -> Tuple[float, str, Dict]:
    """
    Calculate signal based on Ehlers Supertrend strategy
    
    This is a trend-following strategy that uses John Ehlers' Supertrend indicator
    with dynamic adaptation to market conditions.
    
    Args:
        df: DataFrame with indicator data
        config: Strategy configuration
        
    Returns:
        Tuple[float, str, Dict]: Signal strength, direction, and parameters
    """
    if df is None or len(df) < 2:
        return 0.0, "none", {}
    
    if config is None:
        config = {}
    
    # For now, we'll use the standard Supertrend indicator as a simpler implementation
    # In a real implementation, could add Ehlers' specific enhancements
    return calculate_supertrend_signal(df, config)


def calculate_momentum_divergence_strategy(
    df: pd.DataFrame, 
    higher_tf_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None
) -> Tuple[float, str, Dict]:
    """
    Calculate signal based on momentum divergence
    
    This strategy looks for divergences between price and momentum indicators
    (like RSI or MACD) to identify potential trend reversals.
    
    Args:
        df: DataFrame with indicator data
        higher_tf_df: Optional higher timeframe DataFrame
        config: Strategy configuration
        
    Returns:
        Tuple[float, str, Dict]: Signal strength, direction, and parameters
    """
    if df is None or len(df) < 20:  # Need sufficient data for divergence detection
        return 0.0, "none", {}
    
    if config is None:
        config = {}
    
    # Get configuration parameters with defaults
    rsi_period = config.get("rsi_period", 14)
    rsi_overbought = config.get("rsi_overbought", 70)
    rsi_oversold = config.get("rsi_oversold", 30)
    lookback_period = config.get("lookback_period", 20)
    use_macd = config.get("use_macd_divergence", True)
    use_rsi = config.get("use_rsi_divergence", True)
    
    # Default values
    signal_strength = 0.0
    direction = "none"
    params = {"current_price": df["close"].iloc[-1]}
    
    if "atr" in df.columns:
        params["atr"] = df["atr"].iloc[-1]
    
    # Check if we have necessary indicators
    if use_rsi and "rsi" not in df.columns:
        logger.warning("RSI not found in DataFrame but required for momentum divergence strategy")
        use_rsi = False
    
    if use_macd and ("macd" not in df.columns or "macd_signal" not in df.columns):
        logger.warning("MACD not found in DataFrame but required for momentum divergence strategy")
        use_macd = False
    
    if not use_rsi and not use_macd:
        logger.warning("Neither RSI nor MACD available for momentum divergence strategy")
        return 0.0, "none", params
    
    # Get the recent data subset for divergence analysis
    recent_df = df.iloc[-lookback_period:]
    
    # RSI Divergence
    if use_rsi:
        # Find peak/trough for price and RSI
        price_high_idx = recent_df["close"].idxmax()
        price_low_idx = recent_df["close"].idxmin()
        rsi_high_idx = recent_df["rsi"].idxmax()
        rsi_low_idx = recent_df["rsi"].idxmin()
        
        current_rsi = recent_df["rsi"].iloc[-1]
        
        # Bullish divergence (price makes lower low, RSI makes higher low)
        if price_low_idx > rsi_low_idx and recent_df.loc[price_low_idx, "rsi"] > recent_df.loc[rsi_low_idx, "rsi"]:
            if current_rsi < rsi_oversold:  # Confirm with oversold condition
                signal_strength = max(signal_strength, 0.8)
                direction = "buy"
                params["divergence_type"] = "bullish_rsi"
        
        # Bearish divergence (price makes higher high, RSI makes lower high)
        elif price_high_idx > rsi_high_idx and recent_df.loc[price_high_idx, "rsi"] < recent_df.loc[rsi_high_idx, "rsi"]:
            if current_rsi > rsi_overbought:  # Confirm with overbought condition
                signal_strength = max(signal_strength, 0.8)
                direction = "sell"
                params["divergence_type"] = "bearish_rsi"
    
    # MACD Divergence
    if use_macd:
        # For MACD, use histogram for divergence
        if "macd_hist" not in df.columns:
            logger.warning("MACD histogram not found in DataFrame")
        else:
            recent_hist = recent_df["macd_hist"]
            
            # Find histogram peaks/troughs
            macd_high_idx = recent_hist.idxmax()
            macd_low_idx = recent_hist.idxmin()
            
            current_macd = recent_df["macd"].iloc[-1]
            current_macd_signal = recent_df["macd_signal"].iloc[-1]
            
            # Bullish divergence (price makes lower low, MACD hist makes higher low)
            if price_low_idx > macd_low_idx and recent_hist.loc[price_low_idx] > recent_hist.loc[macd_low_idx]:
                if current_macd < 0 and current_macd > current_macd_signal:  # Confirm with MACD crossing up while negative
                    signal_strength = max(signal_strength, 0.9)
                    direction = "buy"
                    params["divergence_type"] = "bullish_macd"
            
            # Bearish divergence (price makes higher high, MACD hist makes lower high)
            elif price_high_idx > macd_high_idx and recent_hist.loc[price_high_idx] < recent_hist.loc[macd_high_idx]:
                if current_macd > 0 and current_macd < current_macd_signal:  # Confirm with MACD crossing down while positive
                    signal_strength = max(signal_strength, 0.9)
                    direction = "sell"
                    params["divergence_type"] = "bearish_macd"
    
    return signal_strength, direction, params


def calculate_multi_timeframe_trend_strategy(
    df: pd.DataFrame, 
    higher_tf_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None
) -> Tuple[float, str, Dict]:
    """
    Calculate signal based on multi-timeframe trend alignment
    
    This strategy aligns trends across multiple timeframes for stronger signals.
    
    Args:
        df: DataFrame with indicator data
        higher_tf_df: Optional higher timeframe DataFrame
        config: Strategy configuration
        
    Returns:
        Tuple[float, str, Dict]: Signal strength, direction, and parameters
    """
    if df is None or len(df) < 2:
        return 0.0, "none", {}
    
    if config is None:
        config = {}
    
    # Default values
    signal_strength = 0.0
    direction = "none"
    params = {"current_price": df["close"].iloc[-1]}
    
    if "atr" in df.columns:
        params["atr"] = df["atr"].iloc[-1]
    
    # Check if we have higher timeframe data
    if higher_tf_df is None or len(higher_tf_df) < 2:
        logger.warning("Higher timeframe data not available for multi-timeframe trend strategy")
        return 0.0, "none", params
    
    # Get configuration parameters
    use_ema = config.get("use_ema", True)
    use_supertrend = config.get("use_supertrend", True)
    use_adx = config.get("use_adx", True)
    adx_threshold = config.get("adx_threshold", 25)
    
    # Check current timeframe trend
    current_trend = "none"
    current_strength = 0.0
    
    # Check via EMA
    if use_ema and "ema_fast" in df.columns and "ema_slow" in df.columns:
        if df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]:
            current_trend = "up"
            current_strength += 0.5
        elif df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1]:
            current_trend = "down"
            current_strength += 0.5
    
    # Check via Supertrend
    if use_supertrend and "supertrend_direction" in df.columns:
        supertrend_direction = df["supertrend_direction"].iloc[-1]
        if supertrend_direction == 1:  # Uptrend
            if current_trend == "up":
                current_strength += 0.5
            elif current_trend == "none":
                current_trend = "up"
                current_strength += 0.5
        elif supertrend_direction == -1:  # Downtrend
            if current_trend == "down":
                current_strength += 0.5
            elif current_trend == "none":
                current_trend = "down"
                current_strength += 0.5
    
    # Check via ADX
    if use_adx and "adx" in df.columns:
        if df["adx"].iloc[-1] > adx_threshold:
            current_strength *= 1.5  # Boost strength if trend is strong
    
    # Check higher timeframe trend
    higher_trend = "none"
    higher_strength = 0.0
    
    # Check via higher timeframe EMA
    if use_ema and "ema_fast" in higher_tf_df.columns and "ema_slow" in higher_tf_df.columns:
        if higher_tf_df["ema_fast"].iloc[-1] > higher_tf_df["ema_slow"].iloc[-1]:
            higher_trend = "up"
            higher_strength += 0.5
        elif higher_tf_df["ema_fast"].iloc[-1] < higher_tf_df["ema_slow"].iloc[-1]:
            higher_trend = "down"
            higher_strength += 0.5
    
    # Check via higher timeframe Supertrend
    if use_supertrend and "supertrend_direction" in higher_tf_df.columns:
        higher_supertrend_direction = higher_tf_df["supertrend_direction"].iloc[-1]
        if higher_supertrend_direction == 1:  # Uptrend
            if higher_trend == "up":
                higher_strength += 0.5
            elif higher_trend == "none":
                higher_trend = "up"
                higher_strength += 0.5
        elif higher_supertrend_direction == -1:  # Downtrend
            if higher_trend == "down":
                higher_strength += 0.5
            elif higher_trend == "none":
                higher_trend = "down"
                higher_strength += 0.5
    
    # Check for alignment between timeframes
    if current_trend == higher_trend and current_trend != "none":
        # Calculate final signal strength
        signal_strength = (current_strength + higher_strength) / 3.0  # Normalize to 0-1 range
        
        # Set direction
        if current_trend == "up":
            direction = "buy"
        else:
            direction = "sell"
        
        # Add to parameters
        params["current_trend"] = current_trend
        params["higher_trend"] = higher_trend
        params["current_strength"] = current_strength
        params["higher_strength"] = higher_strength
    
    return signal_strength, direction, params


def calculate_support_resistance_breakout_strategy(
    df: pd.DataFrame, 
    higher_tf_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None
) -> Tuple[float, str, Dict]:
    """
    Calculate signal based on support/resistance breakouts with volume confirmation
    
    This strategy identifies support and resistance levels and generates signals
    when price breaks these levels with volume confirmation.
    
    Args:
        df: DataFrame with indicator data
        higher_tf_df: Optional higher timeframe DataFrame for S/R levels
        config: Strategy configuration
        
    Returns:
        Tuple[float, str, Dict]: Signal strength, direction, and parameters
    """
    if df is None or len(df) < 30:  # Need sufficient data for S/R detection
        return 0.0, "none", {}
    
    if config is None:
        config = {}
    
    # Default values
    signal_strength = 0.0
    direction = "none"
    params = {"current_price": df["close"].iloc[-1]}
    
    if "atr" in df.columns:
        params["atr"] = df["atr"].iloc[-1]
    
    # Get configuration parameters
    lookback_period = config.get("lookback_period", 100)
    min_touches = config.get("min_touches", 2)
    volume_increase_threshold = config.get("volume_increase_threshold", 1.5)
    
    # Calculate S/R levels
    levels = find_support_resistance_levels(
        df.iloc[-min(len(df), lookback_period):],
        min_touches=min_touches
    )
    
    # Get recent price and volume data
    current_price = df["close"].iloc[-1]
    previous_price = df["close"].iloc[-2]
    current_volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].iloc[-21:-1].mean()  # 20-period volume average
    
    # Check for breakout with volume confirmation
    volume_condition = current_volume > (avg_volume * volume_increase_threshold)
    
    # Check if we have any levels
    if levels and len(levels) > 0:
        # Find closest level
        closest_level = min(levels, key=lambda x: abs(x - current_price))
        
        # Calculate breakout threshold (0.5% of price or ATR)
        breakout_threshold = 0.005 * current_price
        if "atr" in df.columns:
            breakout_threshold = max(breakout_threshold, df["atr"].iloc[-1] * 0.5)
        
        # Check for bullish breakout (break above resistance)
        if previous_price < closest_level and current_price > closest_level and volume_condition:
            # Calculate distance to next resistance (if any)
            higher_levels = [l for l in levels if l > current_price]
            if higher_levels:
                next_resistance = min(higher_levels)
                distance_to_resistance = next_resistance - current_price
                
                # Adjust signal strength based on room to move
                signal_strength = 0.8 + min(0.2, distance_to_resistance / (current_price * 0.1))
            else:
                signal_strength = 1.0
            
            direction = "buy"
            params["breakout_level"] = closest_level
            params["breakout_type"] = "resistance"
        
        # Check for bearish breakout (break below support)
        elif previous_price > closest_level and current_price < closest_level and volume_condition:
            # Calculate distance to next support (if any)
            lower_levels = [l for l in levels if l < current_price]
            if lower_levels:
                next_support = max(lower_levels)
                distance_to_support = current_price - next_support
                
                # Adjust signal strength based on room to move
                signal_strength = 0.8 + min(0.2, distance_to_support / (current_price * 0.1))
            else:
                signal_strength = 1.0
            
            direction = "sell"
            params["breakout_level"] = closest_level
            params["breakout_type"] = "support"
    
    return signal_strength, direction, params


def find_support_resistance_levels(df: pd.DataFrame, min_touches: int = 2) -> List[float]:
    """
    Find support and resistance levels in price data
    
    Args:
        df: DataFrame with OHLCV data
        min_touches: Minimum number of times price must touch a level
        
    Returns:
        List[float]: List of S/R levels
    """
    # Find local maxima/minima
    highs = df["high"].values
    lows = df["low"].values
    
    potential_levels = []
    
    # Extract significant highs
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            potential_levels.append(highs[i])
    
    # Extract significant lows
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            potential_levels.append(lows[i])
    
    # Find clusters of levels (combine levels that are close to each other)
    if not potential_levels:
        return []
    
    # Sort levels
    potential_levels.sort()
    
    # Group levels that are close (within 0.5% of each other)
    grouped_levels = []
    current_group = [potential_levels[0]]
    
    for level in potential_levels[1:]:
        if level - current_group[-1] < current_group[-1] * 0.005:  # Within 0.5%
            current_group.append(level)
        else:
            grouped_levels.append(sum(current_group) / len(current_group))  # Average of group
            current_group = [level]
    
    # Add the last group
    if current_group:
        grouped_levels.append(sum(current_group) / len(current_group))
    
    # Count touches for each level
    filtered_levels = []
    for level in grouped_levels:
        # Count how many times price comes within 0.5% of level
        touches = 0
        for i in range(len(df)):
            if (df["high"].iloc[i] >= level >= df["low"].iloc[i] or  # Level inside candle
                abs(df["high"].iloc[i] - level) / level < 0.005 or    # High near level
                abs(df["low"].iloc[i] - level) / level < 0.005):      # Low near level
                touches += 1
        
        if touches >= min_touches:
            filtered_levels.append(level)
    
    return filtered_levels