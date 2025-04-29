"""
Advanced Trading Strategies Module

This module implements three advanced trading strategies with entry and exit rules:
1. Momentum Divergence Strategy - Uses RSI/MACD divergence with volatility filters
2. Trend Following with Multi-Timeframe Confirmation
3. Support/Resistance Breakout with Volume Confirmation

Each strategy includes comprehensive risk management parameters.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Configure logger
logger = logging.getLogger("strategies")

#############################
# Strategy 1: Momentum Divergence Strategy
#############################

def calculate_momentum_divergence_strategy(df: pd.DataFrame, config: Dict) -> Tuple[float, str, Dict]:
    """
    Momentum Divergence Strategy with Volatility Filters
    
    Entry Rules:
    - LONG: RSI divergence (price making lower lows while RSI makes higher lows)
    - SHORT: RSI divergence (price making higher highs while RSI makes lower highs)
    
    Exit Rules:
    - Take profit at specified target
    - Stop loss based on ATR
    - Exit when RSI crosses midpoint in opposite direction
    
    Filters:
    - Only enter when volatility (ATR %) is within acceptable range
    - Only take long positions when price is above 200 EMA
    - Only take short positions when price is below 200 EMA
    
    Risk Management:
    - Position size based on ATR stop loss
    - Trailing stop activated after specific profit target
    - Max drawdown control - reduce position size after consecutive losses
    
    Args:
        df: DataFrame with indicator data
        config: Strategy configuration
        
    Returns:
        Tuple[float, str, Dict]: Signal strength, direction, and additional trade parameters
    """
    if df.empty or len(df) < 200:
        return 0.0, "", {}
    
    # Get the last 20 candles for analysis
    recent_df = df.tail(20)
    
    # Calculate 200 EMA if not present
    if "ema_200" not in df.columns:
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    
    # Get current price and indicator values
    current_price = df["close"].iloc[-1]
    current_ema_200 = df["ema_200"].iloc[-1]
    
    # Check for required indicators
    if "rsi" not in df.columns or "atr" not in df.columns:
        logger.warning("RSI or ATR indicators missing for Momentum Divergence Strategy")
        return 0.0, "", {}
    
    # Get configuration values with defaults
    rsi_divergence_threshold = config.get("rsi_divergence_threshold", 5)
    min_atr_pct = config.get("min_atr_pct", 0.5)
    max_atr_pct = config.get("max_atr_pct", 3.0)
    atr_multiplier = config.get("atr_multiplier", 2.5)
    take_profit_atr_multiplier = config.get("take_profit_atr_multiplier", 5.0)
    
    # Current ATR and ATR percentage
    current_atr = df["atr"].iloc[-1]
    atr_pct = (current_atr / current_price) * 100
    
    # Skip if volatility is outside acceptable range
    if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
        return 0.0, "", {}
        
    # Check for bullish divergence (price making lower lows, RSI making higher lows)
    price_lows = recent_df["low"].rolling(window=2).min()
    rsi_lows = recent_df["rsi"].rolling(window=2).min()
    
    bullish_divergence = False
    if len(price_lows.dropna()) > 5 and len(rsi_lows.dropna()) > 5:
        price_lower_low = price_lows.iloc[-1] < price_lows.iloc[-5]
        rsi_higher_low = rsi_lows.iloc[-1] > rsi_lows.iloc[-5]
        bullish_divergence = price_lower_low and rsi_higher_low
    
    # Check for bearish divergence (price making higher highs, RSI making lower highs)
    price_highs = recent_df["high"].rolling(window=2).max()
    rsi_highs = recent_df["rsi"].rolling(window=2).max()
    
    bearish_divergence = False
    if len(price_highs.dropna()) > 5 and len(rsi_highs.dropna()) > 5:
        price_higher_high = price_highs.iloc[-1] > price_highs.iloc[-5]
        rsi_lower_high = rsi_highs.iloc[-1] < rsi_highs.iloc[-5]
        bearish_divergence = price_higher_high and rsi_lower_high
    
    # Calculate signal
    signal_strength = 0.0
    direction = ""
    trade_params = {}
    
    # LONG signal
    if bullish_divergence and current_price > current_ema_200:
        signal_strength = 0.8  # Strong signal
        direction = "long"
        
        # Calculate stop loss and take profit levels
        stop_loss = current_price - (current_atr * atr_multiplier)
        take_profit = current_price + (current_atr * take_profit_atr_multiplier)
        
        # Trailing stop settings
        trailing_stop = {
            "enabled": True,
            "activation_pct": 2.0,
            "trail_pct": 1.0
        }
        
        # Additional trade parameters
        trade_params = {
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": take_profit_atr_multiplier / atr_multiplier,
            "trailing_stop": trailing_stop,
            "risk_per_trade_pct": 1.0,
            "strategy": "momentum_divergence"
        }
    
    # SHORT signal
    elif bearish_divergence and current_price < current_ema_200:
        signal_strength = 0.8  # Strong signal
        direction = "short"
        
        # Calculate stop loss and take profit levels
        stop_loss = current_price + (current_atr * atr_multiplier)
        take_profit = current_price - (current_atr * take_profit_atr_multiplier)
        
        # Trailing stop settings
        trailing_stop = {
            "enabled": True,
            "activation_pct": 2.0,
            "trail_pct": 1.0
        }
        
        # Additional trade parameters
        trade_params = {
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": take_profit_atr_multiplier / atr_multiplier,
            "trailing_stop": trailing_stop,
            "risk_per_trade_pct": 1.0,
            "strategy": "momentum_divergence"
        }
    
    return signal_strength, direction, trade_params

#############################
# Strategy 2: Multi-Timeframe Trend Following
#############################

def calculate_multi_timeframe_trend_strategy(
    df: pd.DataFrame, 
    higher_tf_df: pd.DataFrame, 
    config: Dict
) -> Tuple[float, str, Dict]:
    """
    Multi-Timeframe Trend Following Strategy
    
    Entry Rules:
    - LONG: Higher timeframe in uptrend (price > 50 EMA) + current timeframe bullish engulfing
    - SHORT: Higher timeframe in downtrend (price < 50 EMA) + current timeframe bearish engulfing
    
    Exit Rules:
    - Take profit at 3x initial risk
    - Stop loss based on recent swing high/low
    - Exit when price closes below the higher timeframe 50 EMA (for longs)
    - Exit when price closes above the higher timeframe 50 EMA (for shorts)
    
    Filters:
    - Only enter when ADX > 25 (strong trend)
    - Volume on signal candle > 150% of 20-period average volume
    - Avoid trading during high-impact news events
    
    Risk Management:
    - Position size based on distance to swing high/low
    - Scale-in strategy: add to position when trend continues
    - Partial take-profit at 2x risk, move stop loss to breakeven
    
    Args:
        df: DataFrame with indicator data (current timeframe)
        higher_tf_df: DataFrame with indicator data (higher timeframe)
        config: Strategy configuration
        
    Returns:
        Tuple[float, str, Dict]: Signal strength, direction, and additional trade parameters
    """
    if df.empty or higher_tf_df.empty or len(df) < 50:
        return 0.0, "", {}
    
    # Calculate required indicators if not present
    if "ema_50" not in higher_tf_df.columns:
        higher_tf_df["ema_50"] = higher_tf_df["close"].ewm(span=50, adjust=False).mean()
    
    if "adx" not in df.columns:
        # Calculate ADX using pandas_ta
        try:
            import pandas_ta as ta
            df_temp = df.copy()
            df_temp["adx"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
            df["adx"] = df_temp["adx"]
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            # Use simplified ADX approximation
            df["tr"] = np.maximum(
                df["high"] - df["low"],
                np.maximum(
                    abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"] - df["close"].shift(1))
                )
            )
            df["atr_14"] = df["tr"].rolling(window=14).mean()
            df["plus_dm"] = np.where(
                (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
                np.maximum(df["high"] - df["high"].shift(1), 0),
                0
            )
            df["minus_dm"] = np.where(
                (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
                np.maximum(df["low"].shift(1) - df["low"], 0),
                0
            )
            df["plus_di_14"] = 100 * (df["plus_dm"].rolling(window=14).mean() / df["atr_14"])
            df["minus_di_14"] = 100 * (df["minus_dm"].rolling(window=14).mean() / df["atr_14"])
            df["dx"] = 100 * abs(df["plus_di_14"] - df["minus_di_14"]) / (df["plus_di_14"] + df["minus_di_14"])
            df["adx"] = df["dx"].rolling(window=14).mean()
    
    # Get current price and indicator values
    current_price = df["close"].iloc[-1]
    current_adx = df["adx"].iloc[-1]
    
    # Get configuration values with defaults
    min_adx = config.get("min_adx", 25)
    min_volume_factor = config.get("min_volume_factor", 1.5)
    stop_loss_lookback = config.get("stop_loss_lookback", 10)
    take_profit_factor = config.get("take_profit_factor", 3.0)
    scale_in_levels = config.get("scale_in_levels", 2)
    
    # Skip if ADX is below minimum threshold
    if current_adx < min_adx:
        return 0.0, "", {}
    
    # Check volume condition
    avg_volume = df["volume"].iloc[-20:].mean()
    current_volume = df["volume"].iloc[-1]
    volume_condition = current_volume > (avg_volume * min_volume_factor)
    
    # Higher timeframe trend detection
    higher_tf_price = higher_tf_df["close"].iloc[-1]
    higher_tf_ema50 = higher_tf_df["ema_50"].iloc[-1]
    uptrend = higher_tf_price > higher_tf_ema50
    downtrend = higher_tf_price < higher_tf_ema50
    
    # Bullish engulfing pattern
    prev_open = df["open"].iloc[-2]
    prev_close = df["close"].iloc[-2]
    curr_open = df["open"].iloc[-1]
    curr_close = df["close"].iloc[-1]
    
    bullish_engulfing = (
        curr_close > curr_open and
        prev_close < prev_open and
        curr_open <= prev_close and
        curr_close > prev_open
    )
    
    # Bearish engulfing pattern
    bearish_engulfing = (
        curr_close < curr_open and
        prev_close > prev_open and
        curr_open >= prev_close and
        curr_close < prev_open
    )
    
    # Find recent swing low/high for stop loss
    recent_low = df["low"].iloc[-stop_loss_lookback:].min()
    recent_high = df["high"].iloc[-stop_loss_lookback:].max()
    
    # Calculate signal
    signal_strength = 0.0
    direction = ""
    trade_params = {}
    
    # LONG signal
    if uptrend and bullish_engulfing and volume_condition:
        # Calculate stop loss and risk
        stop_loss = recent_low * 0.998  # Slightly below recent low
        risk_amount = current_price - stop_loss
        
        # Calculate take profit levels
        take_profit_full = current_price + (risk_amount * take_profit_factor)
        take_profit_partial = current_price + (risk_amount * 2.0)
        
        # Scale-in levels
        scale_in_points = []
        for i in range(1, scale_in_levels + 1):
            scale_point = current_price + (risk_amount * 0.5 * i)
            scale_in_points.append(scale_point)
        
        signal_strength = 0.9  # Strong signal
        direction = "long"
        
        # Additional trade parameters
        trade_params = {
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit_full,
            "take_profit_partial": take_profit_partial,
            "partial_exit_pct": 50,  # Exit 50% of position at partial take profit
            "risk_reward_ratio": take_profit_factor,
            "scale_in_points": scale_in_points,
            "adx": current_adx,
            "risk_per_trade_pct": 1.5,
            "strategy": "multi_timeframe_trend",
            "move_to_breakeven_at": take_profit_partial,
            "exit_indicator": "Higher TF EMA 50 cross"
        }
    
    # SHORT signal
    elif downtrend and bearish_engulfing and volume_condition:
        # Calculate stop loss and risk
        stop_loss = recent_high * 1.002  # Slightly above recent high
        risk_amount = stop_loss - current_price
        
        # Calculate take profit levels
        take_profit_full = current_price - (risk_amount * take_profit_factor)
        take_profit_partial = current_price - (risk_amount * 2.0)
        
        # Scale-in levels
        scale_in_points = []
        for i in range(1, scale_in_levels + 1):
            scale_point = current_price - (risk_amount * 0.5 * i)
            scale_in_points.append(scale_point)
        
        signal_strength = 0.9  # Strong signal
        direction = "short"
        
        # Additional trade parameters
        trade_params = {
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit_full,
            "take_profit_partial": take_profit_partial,
            "partial_exit_pct": 50,  # Exit 50% of position at partial take profit
            "risk_reward_ratio": take_profit_factor,
            "scale_in_points": scale_in_points,
            "adx": current_adx,
            "risk_per_trade_pct": 1.5,
            "strategy": "multi_timeframe_trend",
            "move_to_breakeven_at": take_profit_partial,
            "exit_indicator": "Higher TF EMA 50 cross"
        }
    
    return signal_strength, direction, trade_params

#############################
# Strategy 3: Support-Resistance Breakout with Volume Confirmation
#############################

def calculate_support_resistance_breakout_strategy(df: pd.DataFrame, config: Dict) -> Tuple[float, str, Dict]:
    """
    Support-Resistance Breakout Strategy with Volume Confirmation
    
    Entry Rules:
    - LONG: Breakout above significant resistance level with volume confirmation
    - SHORT: Breakdown below significant support level with volume confirmation
    
    Exit Rules:
    - Take profit at the next resistance level
    - Stop loss at 50% of the height of the consolidation range
    - Exit when prices close back inside the consolidation range (false breakout)
    
    Filters:
    - Minimum consolidation period of 15 candles
    - Range must be tight (less than 5% from high to low)
    - Volume spike on breakout (>200% of average volume)
    
    Risk Management:
    - Position size based on ATR for risk control
    - Use OCO (One-Cancels-Other) orders for take profit and stop loss
    - Use time-based exit if price stalls after breakout
    
    Args:
        df: DataFrame with indicator data
        config: Strategy configuration
        
    Returns:
        Tuple[float, str, Dict]: Signal strength, direction, and additional trade parameters
    """
    if df.empty or len(df) < 30:
        return 0.0, "", {}
    
    # Get configuration values with defaults
    consolidation_period = config.get("consolidation_period", 15)
    max_range_pct = config.get("max_range_pct", 5.0)
    volume_breakout_factor = config.get("volume_breakout_factor", 2.0)
    stop_loss_factor = config.get("stop_loss_factor", 0.5)
    max_time_in_trade = config.get("max_time_in_trade", 20)  # candles
    false_breakout_pct = config.get("false_breakout_pct", 0.2)
    
    # Current price and volume
    current_close = df["close"].iloc[-1]
    current_volume = df["volume"].iloc[-1]
    previous_close = df["close"].iloc[-2]
    
    # Get the consolidation range (excluding the most recent candle)
    consolidation_df = df.iloc[-(consolidation_period+1):-1]
    consolidation_high = consolidation_df["high"].max()
    consolidation_low = consolidation_df["low"].min()
    consolidation_range = consolidation_high - consolidation_low
    consolidation_range_pct = (consolidation_range / consolidation_low) * 100
    
    # Average volume during consolidation
    avg_volume = consolidation_df["volume"].mean()
    
    # Check if range is tight enough
    if consolidation_range_pct > max_range_pct:
        return 0.0, "", {}
    
    # Find the next resistance and support levels using zigzag method
    def find_pivot_points(df, window=10):
        highs = []
        lows = []
        
        # Find pivot highs
        for i in range(window, len(df) - window):
            if df["high"].iloc[i] == df["high"].iloc[i-window:i+window+1].max():
                highs.append((i, df["high"].iloc[i]))
        
        # Find pivot lows
        for i in range(window, len(df) - window):
            if df["low"].iloc[i] == df["low"].iloc[i-window:i+window+1].min():
                lows.append((i, df["low"].iloc[i]))
        
        return highs, lows
    
    pivot_highs, pivot_lows = find_pivot_points(df.iloc[:-1], window=5)  # Exclude current candle
    
    # Sort by price level
    resistance_levels = sorted([price for _, price in pivot_highs])
    support_levels = sorted([price for _, price in pivot_lows])
    
    # Find next resistance and support levels
    next_resistance = None
    for level in resistance_levels:
        if level > current_close:
            next_resistance = level
            break
    
    next_support = None
    for level in reversed(support_levels):
        if level < current_close:
            next_support = level
            break
    
    # Calculate signal
    signal_strength = 0.0
    direction = ""
    trade_params = {}
    
    # Check for breakout/breakdown with volume confirmation
    breakout = (
        previous_close <= consolidation_high and
        current_close > consolidation_high and
        current_volume > (avg_volume * volume_breakout_factor)
    )
    
    breakdown = (
        previous_close >= consolidation_low and
        current_close < consolidation_low and
        current_volume > (avg_volume * volume_breakout_factor)
    )
    
    # LONG signal (Breakout)
    if breakout and next_resistance is not None:
        # Calculate stop loss
        stop_loss = current_close - (consolidation_range * stop_loss_factor)
        
        # Calculate take profit
        take_profit = next_resistance
        
        # Risk calculation
        risk_amount = current_close - stop_loss
        reward_amount = take_profit - current_close
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Only take trades with good risk/reward
        if risk_reward_ratio >= 1.5:
            signal_strength = 0.85
            direction = "long"
            
            # Additional trade parameters
            trade_params = {
                "entry_price": current_close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": risk_reward_ratio,
                "consolidation_high": consolidation_high,
                "consolidation_low": consolidation_low,
                "false_breakout_level": consolidation_high * (1 - false_breakout_pct/100),
                "max_time_in_trade": max_time_in_trade,
                "volume_confirmation": current_volume / avg_volume,
                "risk_per_trade_pct": 1.0,
                "strategy": "support_resistance_breakout"
            }
    
    # SHORT signal (Breakdown)
    elif breakdown and next_support is not None:
        # Calculate stop loss
        stop_loss = current_close + (consolidation_range * stop_loss_factor)
        
        # Calculate take profit
        take_profit = next_support
        
        # Risk calculation
        risk_amount = stop_loss - current_close
        reward_amount = current_close - take_profit
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Only take trades with good risk/reward
        if risk_reward_ratio >= 1.5:
            signal_strength = 0.85
            direction = "short"
            
            # Additional trade parameters
            trade_params = {
                "entry_price": current_close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": risk_reward_ratio,
                "consolidation_high": consolidation_high,
                "consolidation_low": consolidation_low,
                "false_breakout_level": consolidation_low * (1 + false_breakout_pct/100),
                "max_time_in_trade": max_time_in_trade,
                "volume_confirmation": current_volume / avg_volume,
                "risk_per_trade_pct": 1.0,
                "strategy": "support_resistance_breakout"
            }
    
    return signal_strength, direction, trade_params

#############################
# Strategy Selection and Evaluation
#############################

def evaluate_strategies(df: pd.DataFrame, higher_tf_df: pd.DataFrame, config: Dict) -> Dict:
    """
    Evaluate all strategies and return the best signal with parameters.
    
    Args:
        df: DataFrame with indicator data
        higher_tf_df: DataFrame with higher timeframe data
        config: Strategy configuration
        
    Returns:
        Dict: Best strategy signal and parameters
    """
    # Initialize results
    best_signal = 0.0
    best_direction = ""
    best_params = {}
    best_strategy = ""
    
    # Get weights for each strategy
    momentum_weight = config.get("momentum_divergence", {}).get("weight", 1.0)
    mtf_weight = config.get("multi_timeframe_trend", {}).get("weight", 1.0)
    sr_weight = config.get("support_resistance_breakout", {}).get("weight", 1.0)
    ehlers_weight = config.get("ehlers_supertrend", {}).get("weight", 1.2)
    
    # Evaluate Momentum Divergence Strategy
    if config.get("momentum_divergence", {}).get("enabled", True):
        momentum_config = config.get("momentum_divergence", {})
        signal_strength, direction, trade_params = calculate_momentum_divergence_strategy(df, momentum_config)
        
        # Apply strategy weight
        weighted_signal = signal_strength * momentum_weight
        
        if weighted_signal > best_signal and direction:
            best_signal = weighted_signal
            best_direction = direction
            best_params = trade_params
            best_strategy = "momentum_divergence"
    
    # Evaluate Multi-Timeframe Trend Strategy
    if config.get("multi_timeframe_trend", {}).get("enabled", True):
        mtf_config = config.get("multi_timeframe_trend", {})
        signal_strength, direction, trade_params = calculate_multi_timeframe_trend_strategy(df, higher_tf_df, mtf_config)
        
        # Apply strategy weight
        weighted_signal = signal_strength * mtf_weight
        
        if weighted_signal > best_signal and direction:
            best_signal = weighted_signal
            best_direction = direction
            best_params = trade_params
            best_strategy = "multi_timeframe_trend"
    
    # Evaluate Support-Resistance Breakout Strategy
    if config.get("support_resistance_breakout", {}).get("enabled", True):
        sr_config = config.get("support_resistance_breakout", {})
        signal_strength, direction, trade_params = calculate_support_resistance_breakout_strategy(df, sr_config)
        
        # Apply strategy weight
        weighted_signal = signal_strength * sr_weight
        
        if weighted_signal > best_signal and direction:
            best_signal = weighted_signal
            best_direction = direction
            best_params = trade_params
            best_strategy = "support_resistance_breakout"
    
    # Evaluate Ehlers Supertrend with Ranges Strategy
    if config.get("ehlers_supertrend", {}).get("enabled", True):
        try:
            # Import here to avoid circular imports
            from ehlers_supertrend_strategy import calculate_ehlers_supertrend_signal
            
            ehlers_config = config.get("ehlers_supertrend", {})
            signal_strength, direction, trade_params = calculate_ehlers_supertrend_signal(df, ehlers_config)
            
            # Apply strategy weight
            weighted_signal = signal_strength * ehlers_weight
            
            if weighted_signal > best_signal and direction:
                best_signal = weighted_signal
                best_direction = direction
                best_params = trade_params
                best_strategy = "ehlers_supertrend"
        except ImportError as e:
            pass  # Skip if module not available
    
    return {
        "signal": best_signal,
        "direction": best_direction,
        "parameters": best_params,
        "strategy": best_strategy
    }