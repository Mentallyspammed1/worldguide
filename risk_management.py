"""
Risk Management Module for Trading Bot

This module implements advanced risk management techniques:
- Position sizing based on account risk percentage
- Dynamic stop loss and take profit calculation
- Trailing stop functionality
- Drawdown control and account protection
- Volatility-based position sizing
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple, List, Union, Any, cast
from decimal import Decimal, ROUND_DOWN

# Configure logger
logger = logging.getLogger("risk_management")

def calculate_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    risk_pct: float,
    min_amount: float,
    max_amount: Optional[float] = None,
    amount_precision: int = 6
) -> float:
    """
    Calculate position size based on account risk percentage.
    
    Args:
        account_balance: Available account balance
        entry_price: Planned entry price
        stop_loss: Planned stop loss price
        risk_pct: Risk percentage (0-100)
        min_amount: Minimum allowed position size
        max_amount: Maximum allowed position size (optional)
        amount_precision: Decimal places for position size
        
    Returns:
        float: Position size in base currency
    """
    # Calculate risk amount in quote currency
    risk_amount = account_balance * (risk_pct / 100)
    
    # Calculate dollar risk per unit
    if entry_price > stop_loss:  # Long position
        risk_per_unit = entry_price - stop_loss
        side = "long"
    else:  # Short position
        risk_per_unit = stop_loss - entry_price
        side = "short"
    
    # Avoid division by zero
    if risk_per_unit <= 0:
        logger.warning("Invalid stop loss level, too close to entry price")
        return min_amount
    
    # Calculate position size
    position_size = risk_amount / risk_per_unit
    
    # Calculate position value
    position_value = position_size * entry_price
    
    # Apply maximum if specified
    if max_amount is not None and position_size > max_amount:
        position_size = max_amount
        logger.info(f"Position size capped at maximum: {max_amount}")
    
    # Apply minimum
    if position_size < min_amount:
        position_size = min_amount
        logger.warning(f"Position size increased to minimum: {min_amount}")
    
    # Round to precision
    decimal_places = amount_precision
    position_size = float(Decimal(str(position_size)).quantize(
        Decimal('0.' + '0' * decimal_places),
        rounding=ROUND_DOWN
    ))
    
    logger.info(
        f"Calculated position size: {position_size} (side: {side}, "
        f"risk: {risk_pct}%, risk amount: {risk_amount}, entry: {entry_price}, stop: {stop_loss})"
    )
    
    return position_size

def calculate_dynamic_stop_loss(
    entry_price: float,
    side: str,
    atr: float,
    multiplier: float = 2.0,
    recent_swing: Optional[float] = None,
    price_precision: int = 2
) -> float:
    """
    Calculate dynamic stop loss based on ATR and recent swing levels.
    
    Args:
        entry_price: Entry price
        side: Position side ('long' or 'short')
        atr: Current ATR value
        multiplier: ATR multiplier
        recent_swing: Recent swing high/low (optional)
        price_precision: Decimal places for price
        
    Returns:
        float: Stop loss price
    """
    # Calculate ATR-based stop loss
    atr_stop = 0.0
    if side == "long":
        atr_stop = entry_price - (atr * multiplier)
    else:
        atr_stop = entry_price + (atr * multiplier)
    
    # If swing level provided, use the more conservative stop
    if recent_swing is not None:
        if side == "long":
            # Use the higher of ATR stop and swing low
            stop_loss = max(atr_stop, recent_swing)
        else:
            # Use the lower of ATR stop and swing high
            stop_loss = min(atr_stop, recent_swing)
    else:
        stop_loss = atr_stop
    
    # Round to precision
    stop_loss = round(stop_loss, price_precision)
    
    return stop_loss

def calculate_take_profit(
    entry_price: float,
    stop_loss: float,
    rr_ratio: float = 2.0,
    atr_value: Optional[float] = None,
    atr_multiplier: Optional[float] = None,
    price_precision: int = 2,
    tp_mode: str = "rr_ratio",
    fixed_tp_pct: Optional[float] = None,
    multi_tp_levels: bool = False
) -> Union[float, List[Dict[str, float]]]:
    """
    Calculate take profit based on various strategies (risk-reward ratio, ATR, fixed percentage).
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        rr_ratio: Risk-reward ratio
        atr_value: Current ATR value for ATR-based take profit (optional)
        atr_multiplier: Multiplier for ATR-based take profit (optional)
        price_precision: Decimal places for price
        tp_mode: Take profit mode ('rr_ratio', 'atr', 'fixed', 'multi')
        fixed_tp_pct: Fixed percentage for take profit (optional)
        multi_tp_levels: Create multiple take profit levels
        
    Returns:
        Union[float, List[Dict]]: Take profit price or list of take profit levels
    """
    # Determine position side
    is_long = entry_price > stop_loss if stop_loss > 0 else True
    
    # Calculate take profit based on selected mode
    if tp_mode == "atr" and atr_value is not None and atr_multiplier is not None:
        # ATR-based take profit
        if is_long:
            take_profit = entry_price + (atr_value * atr_multiplier)
        else:
            take_profit = entry_price - (atr_value * atr_multiplier)
    elif tp_mode == "fixed" and fixed_tp_pct is not None:
        # Fixed percentage take profit
        if is_long:
            take_profit = entry_price * (1 + fixed_tp_pct / 100)
        else:
            take_profit = entry_price * (1 - fixed_tp_pct / 100)
    elif tp_mode == "multi":
        # Multiple take profit levels
        tp_levels = []
        
        if is_long:
            # Calculate risk for R:R-based levels (if stop loss is set)
            risk = entry_price - stop_loss if stop_loss > 0 else entry_price * 0.02
            
            # Level 1: 1:1 R:R or 1.5% (25% of position)
            tp1 = entry_price + risk if stop_loss > 0 else entry_price * 1.015
            tp_levels.append({
                "price": round(tp1, price_precision),
                "percentage": 25.0
            })
            
            # Level 2: 2:1 R:R or 3% (50% of position)
            tp2 = entry_price + (risk * 2) if stop_loss > 0 else entry_price * 1.03
            tp_levels.append({
                "price": round(tp2, price_precision),
                "percentage": 50.0
            })
            
            # Level 3: 3:1 R:R or 5% (25% of position)
            tp3 = entry_price + (risk * 3) if stop_loss > 0 else entry_price * 1.05
            tp_levels.append({
                "price": round(tp3, price_precision),
                "percentage": 25.0
            })
        else:
            # Calculate risk for R:R-based levels (if stop loss is set)
            risk = stop_loss - entry_price if stop_loss > 0 else entry_price * 0.02
            
            # Level 1: 1:1 R:R or 1.5% (25% of position)
            tp1 = entry_price - risk if stop_loss > 0 else entry_price * 0.985
            tp_levels.append({
                "price": round(tp1, price_precision),
                "percentage": 25.0
            })
            
            # Level 2: 2:1 R:R or 3% (50% of position)
            tp2 = entry_price - (risk * 2) if stop_loss > 0 else entry_price * 0.97
            tp_levels.append({
                "price": round(tp2, price_precision),
                "percentage": 50.0
            })
            
            # Level 3: 3:1 R:R or 5% (25% of position)
            tp3 = entry_price - (risk * 3) if stop_loss > 0 else entry_price * 0.95
            tp_levels.append({
                "price": round(tp3, price_precision),
                "percentage": 25.0
            })
        
        return tp_levels
    else:
        # Default: Risk-reward ratio based take profit
        if stop_loss <= 0:
            # If no stop loss is set, use a default percentage
            if is_long:
                take_profit = entry_price * (1 + 0.03)  # Default 3% profit
            else:
                take_profit = entry_price * (1 - 0.03)  # Default 3% profit
        else:
            # Calculate based on risk-reward ratio
            if is_long:
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * rr_ratio)
            else:
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * rr_ratio)
    
    # Handle multi-TP for non-multi mode
    if multi_tp_levels and tp_mode != "multi":
        base_tp = take_profit
        tp_levels = []
        
        if is_long:
            # Level 1: 1/3 distance to target (25% of position)
            tp1 = entry_price + (base_tp - entry_price) / 3
            tp_levels.append({
                "price": round(tp1, price_precision),
                "percentage": 25.0
            })
            
            # Level 2: 2/3 distance to target (50% of position)
            tp2 = entry_price + 2 * (base_tp - entry_price) / 3
            tp_levels.append({
                "price": round(tp2, price_precision),
                "percentage": 50.0
            })
            
            # Level 3: full target (25% of position)
            tp_levels.append({
                "price": round(base_tp, price_precision),
                "percentage": 25.0
            })
        else:
            # Level 1: 1/3 distance to target (25% of position)
            tp1 = entry_price - (entry_price - base_tp) / 3
            tp_levels.append({
                "price": round(tp1, price_precision),
                "percentage": 25.0
            })
            
            # Level 2: 2/3 distance to target (50% of position)
            tp2 = entry_price - 2 * (entry_price - base_tp) / 3
            tp_levels.append({
                "price": round(tp2, price_precision),
                "percentage": 50.0
            })
            
            # Level 3: full target (25% of position)
            tp_levels.append({
                "price": round(base_tp, price_precision),
                "percentage": 25.0
            })
        
        return tp_levels
    
    # Round to precision
    take_profit = round(take_profit, price_precision)
    
    return take_profit

def update_trailing_stop(
    current_price: float,
    side: str,
    entry_price: float,
    current_stop: float,
    activation_pct: float,
    trail_pct: float,
    atr_value: Optional[float] = None,
    atr_multiplier: Optional[float] = None,
    price_precision: int = 2,
    advanced_mode: bool = False
) -> float:
    """
    Update trailing stop if conditions are met, with support for ATR-based trailing.
    
    Args:
        current_price: Current market price
        side: Position side ('long' or 'short')
        entry_price: Original entry price
        current_stop: Current stop loss level
        activation_pct: Percentage move needed to activate trailing stop
        trail_pct: Percentage to trail by
        atr_value: Current ATR value for ATR-based trailing (optional)
        atr_multiplier: Multiplier for ATR-based trailing (optional)
        price_precision: Decimal places for price
        advanced_mode: Use advanced trailing stop logic with multiple step levels
        
    Returns:
        float: Updated stop loss price (or current if no update)
    """
    # Check if trailing stop should be activated
    activation_threshold = 0.0
    if side == "long":
        activation_threshold = entry_price * (1 + activation_pct / 100)
        
        if current_price >= activation_threshold:
            # Calculate trailing stop
            if atr_value is not None and atr_multiplier is not None:
                # ATR-based trailing stop
                new_stop = current_price - (atr_value * atr_multiplier)
            else:
                # Percentage-based trailing stop
                new_stop = current_price * (1 - trail_pct / 100)
            
            # Advanced mode - step up trail percentage as profit increases
            if advanced_mode:
                # Calculate profit percentage
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Adjust trail percentage based on profit level
                if profit_pct > 10:
                    # Tighten trail to 25% of original trail percentage for substantial profits
                    new_stop = current_price * (1 - (trail_pct * 0.25) / 100)
                elif profit_pct > 5:
                    # Tighten trail to 50% of original trail percentage for good profits
                    new_stop = current_price * (1 - (trail_pct * 0.5) / 100)
                elif profit_pct > activation_pct * 2:
                    # Tighten trail to 75% of original trail percentage after profit above 2x activation
                    new_stop = current_price * (1 - (trail_pct * 0.75) / 100)
            
            # Only update if new stop is higher than current
            if new_stop > current_stop:
                logger.info(
                    f"Trailing stop updated: {current_stop:.{price_precision}f} -> {new_stop:.{price_precision}f} "
                    f"(price: {current_price:.{price_precision}f}, activation: {activation_threshold:.{price_precision}f})"
                )
                return round(new_stop, price_precision)
    else:  # Short
        activation_threshold = entry_price * (1 - activation_pct / 100)
        
        if current_price <= activation_threshold:
            # Calculate trailing stop
            if atr_value is not None and atr_multiplier is not None:
                # ATR-based trailing stop
                new_stop = current_price + (atr_value * atr_multiplier)
            else:
                # Percentage-based trailing stop
                new_stop = current_price * (1 + trail_pct / 100)
            
            # Advanced mode - step up trail percentage as profit increases
            if advanced_mode:
                # Calculate profit percentage
                profit_pct = ((entry_price - current_price) / entry_price) * 100
                
                # Adjust trail percentage based on profit level
                if profit_pct > 10:
                    # Tighten trail to 25% of original trail percentage for substantial profits
                    new_stop = current_price * (1 + (trail_pct * 0.25) / 100)
                elif profit_pct > 5:
                    # Tighten trail to 50% of original trail percentage for good profits
                    new_stop = current_price * (1 + (trail_pct * 0.5) / 100)
                elif profit_pct > activation_pct * 2:
                    # Tighten trail to 75% of original trail percentage after profit above 2x activation
                    new_stop = current_price * (1 + (trail_pct * 0.75) / 100)
            
            # Only update if new stop is lower than current
            if new_stop < current_stop:
                logger.info(
                    f"Trailing stop updated: {current_stop:.{price_precision}f} -> {new_stop:.{price_precision}f} "
                    f"(price: {current_price:.{price_precision}f}, activation: {activation_threshold:.{price_precision}f})"
                )
                return round(new_stop, price_precision)
    
    # No update
    return current_stop

def check_max_drawdown(
    balance_history: List[float],
    max_drawdown_pct: float = 10.0,
    lookback_periods: int = 30
) -> Tuple[bool, float]:
    """
    Check if maximum drawdown has been exceeded in recent periods.
    
    Args:
        balance_history: List of account balance history
        max_drawdown_pct: Maximum allowable drawdown percentage
        lookback_periods: Number of periods to check
        
    Returns:
        Tuple[bool, float]: (threshold exceeded, current drawdown %)
    """
    if len(balance_history) < 2:
        return False, 0.0
    
    # Get recent balance history
    recent_balance = balance_history[-lookback_periods:] if len(balance_history) > lookback_periods else balance_history
    
    # Calculate peak and current drawdown
    peak_balance = max(recent_balance)
    current_balance = recent_balance[-1]
    
    # Calculate drawdown percentage
    drawdown_pct = ((peak_balance - current_balance) / peak_balance) * 100 if peak_balance > 0 else 0.0
    
    # Check if threshold is exceeded
    threshold_exceeded = drawdown_pct >= max_drawdown_pct
    
    if threshold_exceeded:
        logger.warning(
            f"Maximum drawdown threshold exceeded: {drawdown_pct:.2f}% > {max_drawdown_pct}% "
            f"(peak: {peak_balance}, current: {current_balance})"
        )
    
    return threshold_exceeded, drawdown_pct

def adjust_risk_after_losses(
    base_risk_pct: float,
    consecutive_losses: int,
    reduction_factor: float = 0.5,
    min_risk_pct: float = 0.25
) -> float:
    """
    Reduce risk percentage after consecutive losses.
    
    Args:
        base_risk_pct: Base risk percentage
        consecutive_losses: Number of consecutive losses
        reduction_factor: Factor to reduce risk by each loss
        min_risk_pct: Minimum risk percentage
        
    Returns:
        float: Adjusted risk percentage
    """
    if consecutive_losses <= 0:
        return base_risk_pct
    
    # Calculate adjusted risk
    adjusted_risk = base_risk_pct * (reduction_factor ** consecutive_losses)
    
    # Ensure minimum risk
    adjusted_risk = max(adjusted_risk, min_risk_pct)
    
    logger.info(
        f"Adjusted risk after {consecutive_losses} consecutive losses: "
        f"{base_risk_pct}% -> {adjusted_risk}%"
    )
    
    return adjusted_risk