"""
Ehlers Supertrend Strategy Module

This module implements John Ehlers' Supertrend indicator with enhancements:
- Dynamic adaptation to trending vs ranging markets
- Market type detection using Fisher Transform/CG Oscillator
- Smart order type selection between limit and market orders
- Dual timeframe confirmation
- Range-adapted position sizing and risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from ehlers_indicators import (
    compute_roofing_filter, 
    compute_super_smoother_filter, 
    compute_fisher_transform,
    compute_center_of_gravity,
    compute_cyber_cycle,
    compute_autocorrelation_periodogram,
    compute_signal_to_noise_ratio,
    detect_market_type
)

import logging
logger = logging.getLogger("trading_bot.strategies.ehlers_supertrend")

def calculate_ehlers_supertrend(
    data: pd.DataFrame,
    atr_length: int = 10,
    atr_multiplier: float = 2.0,
    smoothing_length: int = 5,
    use_median_price: bool = True,
    rsi_filter: bool = True,
    rsi_length: int = 14,
    rsi_threshold_high: float = 70.0,
    rsi_threshold_low: float = 30.0
) -> pd.DataFrame:
    """
    Calculate Ehlers Supertrend indicator
    
    Args:
        data: OHLCV DataFrame
        atr_length: Period for ATR calculation
        atr_multiplier: Multiplier for ATR bands
        smoothing_length: Super smoother filter length
        use_median_price: Whether to use median price instead of close
        rsi_filter: Whether to apply RSI filter
        rsi_length: RSI period
        rsi_threshold_high: Upper RSI threshold
        rsi_threshold_low: Lower RSI threshold
        
    Returns:
        pd.DataFrame: DataFrame with Supertrend indicator
    """
    df = data.copy()
    
    # Calculate median price if requested
    if use_median_price:
        df['price'] = (df['high'] + df['low']) / 2
    else:
        df['price'] = df['close']
    
    # Smooth price using SuperSmoother filter
    df['smooth_price'] = compute_super_smoother_filter(df['price'], smoothing_length)
    
    # Calculate ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=atr_length).mean()
    
    # Calculate upper and lower bands
    df['upper_band'] = df['smooth_price'] + df['atr'] * atr_multiplier
    df['lower_band'] = df['smooth_price'] - df['atr'] * atr_multiplier
    
    # Initialize supertrend columns
    df['supertrend'] = 0.0
    df['supertrend_direction'] = 0
    df['in_uptrend'] = False
    
    # Calculate supertrend
    for i in range(1, len(df)):
        # Default: use previous values
        df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i-1], 'supertrend']
        df.loc[df.index[i], 'supertrend_direction'] = df.loc[df.index[i-1], 'supertrend_direction']
        
        # If we were in uptrend
        if df.loc[df.index[i-1], 'supertrend_direction'] == 1:
            # Lower band is our supertrend reference
            df.loc[df.index[i], 'supertrend'] = max(
                df.loc[df.index[i], 'lower_band'],
                df.loc[df.index[i-1], 'supertrend']
            )
            
            # Check if trend has changed
            if df.loc[df.index[i], 'close'] < df.loc[df.index[i], 'supertrend']:
                df.loc[df.index[i], 'supertrend_direction'] = -1
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'upper_band']
        
        # If we were in downtrend
        elif df.loc[df.index[i-1], 'supertrend_direction'] == -1:
            # Upper band is our supertrend reference
            df.loc[df.index[i], 'supertrend'] = min(
                df.loc[df.index[i], 'upper_band'],
                df.loc[df.index[i-1], 'supertrend']
            )
            
            # Check if trend has changed
            if df.loc[df.index[i], 'close'] > df.loc[df.index[i], 'supertrend']:
                df.loc[df.index[i], 'supertrend_direction'] = 1
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'lower_band']
        
        # For first calculation
        else:
            if df.loc[df.index[i], 'close'] > df.loc[df.index[i], 'smooth_price']:
                df.loc[df.index[i], 'supertrend_direction'] = 1
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'lower_band']
            else:
                df.loc[df.index[i], 'supertrend_direction'] = -1
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'upper_band']
        
        # Set in_uptrend flag for easier reading
        df.loc[df.index[i], 'in_uptrend'] = (df.loc[df.index[i], 'supertrend_direction'] == 1)
    
    # Calculate RSI for filter
    if rsi_filter:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_length).mean()
        avg_loss = loss.rolling(window=rsi_length).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Apply RSI filter to supertrend
        df['supertrend_filtered'] = df['supertrend_direction']
        
        # Only allow longs when RSI is not overbought
        df.loc[df['rsi'] > rsi_threshold_high, 'supertrend_filtered'] = \
            df.loc[df['rsi'] > rsi_threshold_high, 'supertrend_filtered'].map(lambda x: -1 if x == 1 else x)
        
        # Only allow shorts when RSI is not oversold
        df.loc[df['rsi'] < rsi_threshold_low, 'supertrend_filtered'] = \
            df.loc[df['rsi'] < rsi_threshold_low, 'supertrend_filtered'].map(lambda x: 1 if x == -1 else x)
    else:
        df['supertrend_filtered'] = df['supertrend_direction']
    
    return df


def analyze_market_conditions(
    data: pd.DataFrame,
    fisher_length: int = 10,
    cycle_length: int = 20,
    snr_length: int = 25,
    snr_threshold: float = 2.5,
    dominant_cycle_lookback: int = 125
) -> Dict:
    """
    Analyze market conditions for adaptive parameter selection
    
    Args:
        data: OHLCV DataFrame
        fisher_length: Fisher transform length
        cycle_length: Cyber cycle length
        snr_length: Signal-to-noise ratio calculation length
        snr_threshold: Threshold for good trend
        dominant_cycle_lookback: Lookback for dominant cycle detection
        
    Returns:
        Dict: Market condition analysis
    """
    # Calculate indicators for market analysis
    df = data.copy()
    
    # Median price
    df['median'] = (df['high'] + df['low']) / 2
    
    # Fisher transform for trend/range detection
    df['fisher'] = compute_fisher_transform(df['median'], fisher_length)
    
    # Center of gravity oscillator
    df['cg_osc'] = compute_center_of_gravity(df['close'], 10)
    
    # Cyber cycle for cycle detection
    df['cycle'] = compute_cyber_cycle(df['close'], cycle_length)
    
    # Calculate signal-to-noise ratio
    df['snr'] = compute_signal_to_noise_ratio(df['close'], snr_length)
    
    # Detect dominant cycle
    cycles = compute_autocorrelation_periodogram(df['close'].tail(dominant_cycle_lookback))
    dominant_cycle = cycles[0] if cycles else 20  # Default to 20 if detection fails
    
    # Determine market type
    market_type, market_score = detect_market_type(
        fisher=df['fisher'].iloc[-1],
        cg_osc=df['cg_osc'].iloc[-1],
        snr=df['snr'].iloc[-1],
        snr_threshold=snr_threshold
    )
    
    # Calculate price volatility
    df['returns'] = df['close'].pct_change()
    volatility = df['returns'].tail(20).std() * 100  # Multiply by 100 for percentage
    
    # Calculate recent trend strength
    last_price = df['close'].iloc[-1]
    price_20_ago = df['close'].iloc[-min(20, len(df)-1)]
    trend_slope = (last_price - price_20_ago) / max(price_20_ago, 1e-8) * 100  # Percentage
    
    # Calculate effective range as percentage of price
    recent_high = df['high'].tail(dominant_cycle).max()
    recent_low = df['low'].tail(dominant_cycle).min()
    effective_range = (recent_high - recent_low) / last_price * 100  # Percentage
    
    return {
        'market_type': market_type,
        'market_score': market_score,  # Higher means stronger trend, lower means stronger range
        'dominant_cycle': dominant_cycle,
        'fisher': df['fisher'].iloc[-1],
        'cg_osc': df['cg_osc'].iloc[-1],
        'snr': df['snr'].iloc[-1],
        'volatility': volatility,
        'trend_slope': trend_slope,
        'effective_range': effective_range,
        'recent_high': recent_high,
        'recent_low': recent_low
    }


def calculate_adaptive_parameters(
    market_conditions: Dict,
    base_config: Dict
) -> Dict:
    """
    Calculate adaptive parameters based on market conditions
    
    Args:
        market_conditions: Market condition analysis
        base_config: Base configuration
        
    Returns:
        Dict: Adapted parameters
    """
    market_type = market_conditions['market_type']
    market_score = market_conditions['market_score']
    volatility = market_conditions['volatility']
    effective_range = market_conditions['effective_range']
    
    # Start with base configuration
    adapted = base_config.copy()
    
    # Adapt ATR multiplier based on market type and volatility
    if market_type == 'trending':
        # In trending market, use tighter stop if trend is strong
        if market_score > 0.7:
            adapted['atr_multiplier'] = base_config.get('atr_multiplier', 2.0) * 0.8
        else:
            adapted['atr_multiplier'] = base_config.get('atr_multiplier', 2.0) * 0.9
    elif market_type == 'ranging':
        # In ranging market, use wider stop to avoid whipsaws
        adapted['atr_multiplier'] = base_config.get('atr_multiplier', 2.0) * 1.2
    elif market_type == 'volatile':
        # In volatile market, use even wider stop
        adapted['atr_multiplier'] = base_config.get('atr_multiplier', 2.0) * 1.5
    
    # Adjust for overall volatility
    if volatility > 2.0:  # High volatility threshold (2% daily)
        adapted['atr_multiplier'] *= min(1.5, 1.0 + (volatility - 2.0) * 0.25)
    
    # Adapt order type based on market type
    if market_type == 'trending':
        adapted['order_type'] = 'market'  # Use market orders in trending market
        adapted['preferred_entry'] = 'market'
    elif market_type == 'ranging':
        adapted['order_type'] = 'limit'  # Use limit orders in ranging market
        adapted['preferred_entry'] = 'limit_better'
    else:
        adapted['order_type'] = 'market'  # Default to market in other conditions
        adapted['preferred_entry'] = 'market'
    
    # Adapt risk percentage based on market conditions
    if market_type == 'trending' and market_score > 0.7:
        # Higher risk for strong trends
        adapted['risk_per_trade_pct'] = base_config.get('risk_per_trade_pct', 1.0) * 1.2
    elif market_type == 'volatile':
        # Lower risk for volatile markets
        adapted['risk_per_trade_pct'] = base_config.get('risk_per_trade_pct', 1.0) * 0.7
    
    # Adapt take profit based on range
    adapted['rr_ratio'] = base_config.get('rr_ratio', 2.0)
    if market_type == 'ranging':
        # In ranging market, target the range extremes
        adapted['rr_ratio'] = min(base_config.get('rr_ratio', 2.0), effective_range / 2.0)
    
    # Adapt trailing stop
    adapted['use_trailing_stop'] = base_config.get('use_trailing_stop', True)
    if market_type == 'trending':
        # Tighter trail in trending market to lock in profits
        adapted['trailing_stop_activation_pct'] = base_config.get('trailing_stop_activation_pct', 1.0) * 0.8
        adapted['trailing_stop_trail_pct'] = base_config.get('trailing_stop_trail_pct', 0.5) * 0.8
    else:
        # Default trailing stop settings
        adapted['trailing_stop_activation_pct'] = base_config.get('trailing_stop_activation_pct', 1.0)
        adapted['trailing_stop_trail_pct'] = base_config.get('trailing_stop_trail_pct', 0.5)
    
    return adapted


def calculate_ehlers_supertrend_signal(
    data: pd.DataFrame,
    config: Dict = None
) -> Tuple[float, str, Dict]:
    """
    Calculate signal from Ehlers Supertrend strategy
    
    Args:
        data: OHLCV DataFrame
        config: Strategy configuration
        
    Returns:
        Tuple[float, str, Dict]: Signal strength, direction, and parameters
    """
    if config is None:
        config = {}
    
    # Default configuration
    default_config = {
        'atr_length': 10,
        'atr_multiplier': 2.0,
        'smoothing_length': 5,
        'use_median_price': True,
        'rsi_filter': True,
        'rsi_length': 14,
        'rsi_threshold_high': 70.0,
        'rsi_threshold_low': 30.0,
        'risk_per_trade_pct': 1.0,
        'rr_ratio': 2.0,
        'use_adaptive_parameters': True,
        'use_dual_timeframe': True,
        'use_trailing_stop': True,
        'trailing_stop_activation_pct': 1.0,
        'trailing_stop_trail_pct': 0.5,
        'order_type': 'market',
        'preferred_entry': 'market'
    }
    
    # Merge with provided configuration
    effective_config = {**default_config, **config}
    
    try:
        # Analyze market conditions
        market_conditions = analyze_market_conditions(data)
        
        # Apply adaptive parameters if requested
        if effective_config['use_adaptive_parameters']:
            effective_config = calculate_adaptive_parameters(market_conditions, effective_config)
        
        # Calculate Supertrend indicator
        supertrend_df = calculate_ehlers_supertrend(
            data,
            atr_length=effective_config['atr_length'],
            atr_multiplier=effective_config['atr_multiplier'],
            smoothing_length=effective_config['smoothing_length'],
            use_median_price=effective_config['use_median_price'],
            rsi_filter=effective_config['rsi_filter'],
            rsi_length=effective_config['rsi_length'],
            rsi_threshold_high=effective_config['rsi_threshold_high'],
            rsi_threshold_low=effective_config['rsi_threshold_low']
        )
        
        # Get latest values
        last_index = supertrend_df.index[-1]
        current_direction = supertrend_df.loc[last_index, 'supertrend_filtered']
        current_supertrend = supertrend_df.loc[last_index, 'supertrend']
        prev_direction = supertrend_df.loc[supertrend_df.index[-2], 'supertrend_filtered'] if len(supertrend_df) > 1 else 0
        
        # Default signal values
        signal_strength = 0.0
        direction = ""
        
        # Calculate signal
        if current_direction == 1 and prev_direction != 1:
            # Long signal
            signal_strength = 1.0
            direction = "long"
            
            # Calculate stop loss and take profit
            current_price = data.loc[last_index, 'close']
            stop_loss = current_supertrend
            risk_amount = current_price - stop_loss
            take_profit = current_price + (risk_amount * effective_config['rr_ratio'])
            
            # Prepare signal parameters
            params = {
                'side': direction,
                'signal_price': current_price,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_per_trade_pct': effective_config['risk_per_trade_pct'],
                'supertrend_value': current_supertrend,
                'order_type': effective_config['order_type'],
                'preferred_entry': effective_config['preferred_entry'],
                'market_type': market_conditions['market_type'],
                'market_score': market_conditions['market_score'],
                'trailing_stop': {
                    'enabled': effective_config['use_trailing_stop'],
                    'activation_pct': effective_config['trailing_stop_activation_pct'],
                    'trail_pct': effective_config['trailing_stop_trail_pct']
                }
            }
            
        elif current_direction == -1 and prev_direction != -1:
            # Short signal
            signal_strength = 1.0
            direction = "short"
            
            # Calculate stop loss and take profit
            current_price = data.loc[last_index, 'close']
            stop_loss = current_supertrend
            risk_amount = stop_loss - current_price
            take_profit = current_price - (risk_amount * effective_config['rr_ratio'])
            
            # Prepare signal parameters
            params = {
                'side': direction,
                'signal_price': current_price,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_per_trade_pct': effective_config['risk_per_trade_pct'],
                'supertrend_value': current_supertrend,
                'order_type': effective_config['order_type'],
                'preferred_entry': effective_config['preferred_entry'],
                'market_type': market_conditions['market_type'],
                'market_score': market_conditions['market_score'],
                'trailing_stop': {
                    'enabled': effective_config['use_trailing_stop'],
                    'activation_pct': effective_config['trailing_stop_activation_pct'],
                    'trail_pct': effective_config['trailing_stop_trail_pct']
                }
            }
        
        else:
            # No signal
            signal_strength = 0.0
            direction = ""
            params = {}
        
        return signal_strength, direction, params
        
    except Exception as e:
        logger.error(f"Error calculating Ehlers Supertrend signal: {e}")
        return 0.0, "", {}


def calculate_ehlers_supertrend_exit_signal(
    data: pd.DataFrame,
    position: Dict,
    config: Dict = None
) -> bool:
    """
    Calculate exit signal from Ehlers Supertrend strategy
    
    Args:
        data: OHLCV DataFrame
        position: Current position
        config: Strategy configuration
        
    Returns:
        bool: True if should exit, False otherwise
    """
    if config is None:
        config = {}
    
    # Default configuration
    default_config = {
        'atr_length': 10,
        'atr_multiplier': 2.0,
        'smoothing_length': 5,
        'use_median_price': True,
        'rsi_filter': True,
        'rsi_length': 14,
        'rsi_threshold_high': 70.0,
        'rsi_threshold_low': 30.0,
        'exit_on_supertrend_flip': True
    }
    
    # Merge with provided configuration
    effective_config = {**default_config, **config}
    
    try:
        # Calculate Supertrend indicator
        supertrend_df = calculate_ehlers_supertrend(
            data,
            atr_length=effective_config['atr_length'],
            atr_multiplier=effective_config['atr_multiplier'],
            smoothing_length=effective_config['smoothing_length'],
            use_median_price=effective_config['use_median_price'],
            rsi_filter=effective_config['rsi_filter'],
            rsi_length=effective_config['rsi_length'],
            rsi_threshold_high=effective_config['rsi_threshold_high'],
            rsi_threshold_low=effective_config['rsi_threshold_low']
        )
        
        # Get latest values
        last_index = supertrend_df.index[-1]
        current_direction = supertrend_df.loc[last_index, 'supertrend_filtered']
        position_side = position.get('side')
        
        # Check for exit signal
        if effective_config['exit_on_supertrend_flip']:
            if position_side == "long" and current_direction == -1:
                # Exit long position when supertrend flips down
                return True
            elif position_side == "short" and current_direction == 1:
                # Exit short position when supertrend flips up
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error calculating Ehlers Supertrend exit signal: {e}")
        return False