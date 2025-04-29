"""
Risk Management Module

This module provides functions for calculating position sizes,
dynamic stop losses, take profit levels, and other risk management parameters.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

# Configure logger
logger = logging.getLogger("risk_management")


def calculate_position_size(
    balance: float,
    risk_percentage: float,
    entry_price: float,
    stop_loss: float,
    leverage: float = 1.0,
    min_position_size: Optional[float] = None,
    max_position_size: Optional[float] = None,
    precision: int = 8
) -> float:
    """
    Calculate position size based on risk percentage
    
    Args:
        balance: Account balance
        risk_percentage: Percentage of balance to risk (0-100)
        entry_price: Entry price
        stop_loss: Stop loss price
        leverage: Leverage multiplier
        min_position_size: Minimum position size
        max_position_size: Maximum position size
        precision: Decimal precision for rounding
        
    Returns:
        float: Position size
    """
    if balance <= 0 or entry_price <= 0:
        logger.warning("Invalid balance or entry price for position sizing")
        return 0.0
    
    # Convert percentage to decimal
    risk_ratio = risk_percentage / 100.0
    
    # Calculate risk amount in account currency
    risk_amount = balance * risk_ratio
    
    # Calculate price distance to stop loss
    if stop_loss > 0:
        price_distance = abs(entry_price - stop_loss)
        price_distance_pct = price_distance / entry_price
    else:
        # Default to 2% if no stop loss
        price_distance_pct = 0.02
        price_distance = entry_price * price_distance_pct
    
    # Avoid division by zero
    if price_distance <= 0:
        logger.warning("Invalid price distance for position sizing")
        return 0.0
    
    # Calculate position size
    position_size = risk_amount / price_distance
    
    # Apply leverage
    position_size *= leverage
    
    # Convert to position units (e.g., BTC)
    position_size = position_size / entry_price
    
    # Apply min/max limits
    if min_position_size is not None and position_size < min_position_size:
        position_size = min_position_size
    
    if max_position_size is not None and position_size > max_position_size:
        position_size = max_position_size
    
    # Round to appropriate precision
    position_size = math.floor(position_size * (10 ** precision)) / (10 ** precision)
    
    return position_size


def calculate_dynamic_stop_loss(
    entry_price: float,
    direction: str,
    atr: Optional[float] = None,
    atr_multiplier: float = 1.5,
    fixed_percentage: Optional[float] = None,
    min_distance_percentage: float = 0.005,
    precision: int = 8
) -> float:
    """
    Calculate dynamic stop loss based on ATR or fixed percentage
    
    Args:
        entry_price: Entry price
        direction: Trade direction ('buy' or 'sell')
        atr: Average True Range value
        atr_multiplier: Multiplier for ATR
        fixed_percentage: Fixed percentage for stop loss (0-100)
        min_distance_percentage: Minimum distance from entry as percentage
        precision: Decimal precision for rounding
        
    Returns:
        float: Stop loss price
    """
    if entry_price <= 0:
        logger.warning("Invalid entry price for stop loss calculation")
        return 0.0
    
    # Determine distance from entry price
    if atr is not None and atr > 0:
        # Use ATR-based distance
        distance = atr * atr_multiplier
    elif fixed_percentage is not None:
        # Use fixed percentage distance
        distance = entry_price * (fixed_percentage / 100.0)
    else:
        # Default to minimum distance
        distance = entry_price * min_distance_percentage
    
    # Ensure minimum distance
    min_distance = entry_price * min_distance_percentage
    if distance < min_distance:
        distance = min_distance
    
    # Calculate stop loss price based on direction
    if direction.lower() in ['buy', 'long']:
        stop_loss = entry_price - distance
    else:  # sell/short
        stop_loss = entry_price + distance
    
    # Round to appropriate precision
    stop_loss = round(stop_loss, precision)
    
    return stop_loss


def calculate_take_profit(
    entry_price: float,
    direction: str,
    risk_reward_ratio: float = 2.0,
    stop_loss: Optional[float] = None,
    fixed_percentage: Optional[float] = None,
    precision: int = 8
) -> float:
    """
    Calculate take profit level based on risk-reward ratio or fixed percentage
    
    Args:
        entry_price: Entry price
        direction: Trade direction ('buy' or 'sell')
        risk_reward_ratio: Risk-reward ratio (TP distance = SL distance * ratio)
        stop_loss: Stop loss price (for risk-reward calculation)
        fixed_percentage: Fixed percentage for take profit (0-100)
        precision: Decimal precision for rounding
        
    Returns:
        float: Take profit price
    """
    if entry_price <= 0:
        logger.warning("Invalid entry price for take profit calculation")
        return 0.0
    
    # Determine distance from entry price
    if stop_loss is not None and stop_loss > 0:
        # Use risk-reward ratio based on stop loss
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = sl_distance * risk_reward_ratio
    elif fixed_percentage is not None:
        # Use fixed percentage distance
        tp_distance = entry_price * (fixed_percentage / 100.0)
    else:
        # Default to 2% distance
        tp_distance = entry_price * 0.02
    
    # Calculate take profit price based on direction
    if direction.lower() in ['buy', 'long']:
        take_profit = entry_price + tp_distance
    else:  # sell/short
        take_profit = entry_price - tp_distance
    
    # Round to appropriate precision
    take_profit = round(take_profit, precision)
    
    return take_profit


def update_trailing_stop(
    entry_price: float,
    current_price: float,
    current_stop: float,
    direction: str,
    trail_percentage: float = 0.5,
    activation_percentage: float = 1.0,
    precision: int = 8
) -> float:
    """
    Update trailing stop based on current price movement
    
    Args:
        entry_price: Entry price
        current_price: Current market price
        current_stop: Current stop loss level
        direction: Trade direction ('buy' or 'sell')
        trail_percentage: Trailing percentage (0-100)
        activation_percentage: Percentage profit required to activate trailing (0-100)
        precision: Decimal precision for rounding
        
    Returns:
        float: Updated stop loss price
    """
    if entry_price <= 0 or current_price <= 0:
        logger.warning("Invalid prices for trailing stop calculation")
        return current_stop
    
    # Convert percentages to decimals
    trail_ratio = trail_percentage / 100.0
    activation_ratio = activation_percentage / 100.0
    
    # Calculate activation threshold
    if direction.lower() in ['buy', 'long']:
        activation_level = entry_price * (1 + activation_ratio)
        
        # Check if price has reached activation level
        if current_price >= activation_level:
            # Calculate new stop based on trailing percentage
            new_stop = current_price * (1 - trail_ratio)
            
            # Only move stop upward
            if new_stop > current_stop:
                return round(new_stop, precision)
    else:  # sell/short
        activation_level = entry_price * (1 - activation_ratio)
        
        # Check if price has reached activation level
        if current_price <= activation_level:
            # Calculate new stop based on trailing percentage
            new_stop = current_price * (1 + trail_ratio)
            
            # Only move stop downward
            if new_stop < current_stop:
                return round(new_stop, precision)
    
    # Return current stop if conditions not met
    return current_stop


def calculate_position_exposure(
    position_size: float,
    entry_price: float,
    account_balance: float,
    leverage: float = 1.0
) -> float:
    """
    Calculate position exposure as percentage of account balance
    
    Args:
        position_size: Position size in units
        entry_price: Entry price per unit
        account_balance: Total account balance
        leverage: Leverage used
        
    Returns:
        float: Exposure percentage (0-100)
    """
    if account_balance <= 0:
        logger.warning("Invalid account balance for exposure calculation")
        return 0.0
    
    # Calculate position value
    position_value = position_size * entry_price
    
    # Adjust for leverage to get actual margin used
    margin_used = position_value / leverage
    
    # Calculate exposure percentage
    exposure_percentage = (margin_used / account_balance) * 100.0
    
    return exposure_percentage


def check_max_drawdown(
    current_balance: float,
    peak_balance: float,
    max_drawdown_percentage: float = 20.0
) -> Tuple[float, bool]:
    """
    Check current drawdown against maximum allowed drawdown
    
    Args:
        current_balance: Current account balance
        peak_balance: Peak account balance
        max_drawdown_percentage: Maximum allowed drawdown (0-100)
        
    Returns:
        Tuple[float, bool]: Current drawdown percentage and whether max drawdown is exceeded
    """
    if peak_balance <= 0:
        return 0.0, False
    
    # Calculate current drawdown
    drawdown_percentage = ((peak_balance - current_balance) / peak_balance) * 100.0
    
    # Check if max drawdown is exceeded
    max_exceeded = drawdown_percentage > max_drawdown_percentage
    
    return drawdown_percentage, max_exceeded


def calculate_portfolio_heat(
    positions: List[Dict],
    account_balance: float,
    max_heat: float = 15.0
) -> Tuple[float, bool]:
    """
    Calculate total portfolio heat (percentage of account at risk)
    
    Args:
        positions: List of position dictionaries with entry, stop, size
        account_balance: Total account balance
        max_heat: Maximum allowable heat percentage (0-100)
        
    Returns:
        Tuple[float, bool]: Portfolio heat percentage and whether max heat is exceeded
    """
    if account_balance <= 0:
        return 0.0, False
    
    total_risk = 0.0
    
    for position in positions:
        entry = position.get("entry_price", 0.0)
        stop = position.get("stop_loss", 0.0)
        size = position.get("size", 0.0)
        leverage = position.get("leverage", 1.0)
        direction = position.get("side", "long").lower()
        
        if entry <= 0 or stop <= 0 or size <= 0:
            continue
        
        # Calculate position risk
        if direction in ["buy", "long"]:
            risk_per_unit = entry - stop
        else:  # sell/short
            risk_per_unit = stop - entry
        
        # Convert to percentage
        risk_percentage = (risk_per_unit / entry) * 100.0
        
        # Calculate risk amount
        position_value = size * entry
        risk_amount = position_value * (risk_percentage / 100.0)
        
        # Adjust for leverage
        margin_used = position_value / leverage
        
        # Add to total risk
        total_risk += risk_amount
    
    # Calculate portfolio heat
    portfolio_heat = (total_risk / account_balance) * 100.0
    
    # Check if max heat is exceeded
    max_exceeded = portfolio_heat > max_heat
    
    return portfolio_heat, max_exceeded