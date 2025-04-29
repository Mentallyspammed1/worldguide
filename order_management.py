"""
Order Management Module

This module implements advanced order management features for the trading bot:
- Limit order handling
- OCO (One-Cancels-Other) orders for stop loss and take profit
- Trailing stop orders
- Partial position entry and exit
- Scale-in and scale-out strategies
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from decimal import Decimal, ROUND_DOWN

import ccxt
import numpy as np
import pandas as pd

from utils import retry_api_call


# Configure logger
logger = logging.getLogger("order_management")


def execute_market_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    params: Dict = None
) -> Dict:
    """
    Execute a market order.
    
    Args:
        exchange: Exchange instance
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        amount: Order amount in base currency
        params: Additional parameters for the order
        
    Returns:
        Dict: Order result
    """
    logger.info(f"Executing market {side} order for {amount} {symbol}")
    
    try:
        order = retry_api_call(
            exchange.create_order,
            symbol,
            "market",
            side,
            amount,
            params=params or {}
        )
        
        logger.info(f"Market order executed: {order['id']}")
        return order
    except Exception as e:
        logger.error(f"Failed to execute market order: {e}")
        raise


def execute_limit_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    price: float,
    params: Dict = None
) -> Dict:
    """
    Execute a limit order.
    
    Args:
        exchange: Exchange instance
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        amount: Order amount in base currency
        price: Limit price
        params: Additional parameters for the order
        
    Returns:
        Dict: Order result
    """
    logger.info(f"Executing limit {side} order for {amount} {symbol} at {price}")
    
    try:
        order = retry_api_call(
            exchange.create_order,
            symbol,
            "limit",
            side,
            amount,
            price,
            params=params or {}
        )
        
        logger.info(f"Limit order placed: {order['id']}")
        return order
    except Exception as e:
        logger.error(f"Failed to execute limit order: {e}")
        raise


def execute_stop_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    price: float,
    stop_price: float = None,
    params: Dict = None
) -> Dict:
    """
    Execute a stop order.
    
    Args:
        exchange: Exchange instance
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        amount: Order amount in base currency
        price: Order price
        stop_price: Stop price (same as price if None)
        params: Additional parameters for the order
        
    Returns:
        Dict: Order result
    """
    stop_price = stop_price or price
    logger.info(f"Executing stop {side} order for {amount} {symbol} "
               f"at {price} (stop: {stop_price})")
    
    params = params or {}
    params["stopPrice"] = stop_price
    
    try:
        order = retry_api_call(
            exchange.create_order,
            symbol,
            "stop",
            side,
            amount,
            price,
            params=params
        )
        
        logger.info(f"Stop order placed: {order['id']}")
        return order
    except Exception as e:
        logger.error(f"Failed to execute stop order: {e}")
        raise


def execute_trailing_stop_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    activation_price: float,
    trail_value: float,
    is_percentage: bool = True,
    params: Dict = None
) -> Dict:
    """
    Execute a trailing stop order if supported by the exchange.
    
    Args:
        exchange: Exchange instance
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        amount: Order amount in base currency
        activation_price: Price at which trailing begins
        trail_value: Amount or percentage to trail by
        is_percentage: Whether trail_value is a percentage (True) or absolute value (False)
        params: Additional parameters for the order
        
    Returns:
        Dict: Order result
    """
    logger.info(f"Executing trailing stop {side} order for {amount} {symbol}")
    
    params = params or {}
    
    # Add exchange-specific parameters
    if exchange.id == "bybit":
        # Bybit uses callbackRate for percentage-based trailing stops
        if is_percentage:
            params["callbackRate"] = trail_value
            logger.info(f"Trailing by {trail_value}%")
        else:
            # Convert to percentage for Bybit
            if activation_price > 0:
                callback_rate = (trail_value / activation_price) * 100
                params["callbackRate"] = callback_rate
                logger.info(f"Trailing by {callback_rate}% (converted from {trail_value})")
            else:
                raise ValueError("Invalid activation price for calculating callback rate")
        
        if side == "buy":
            # For short positions closing (buying)
            params["activationPrice"] = activation_price
        else:
            # For long positions closing (selling)
            params["activationPrice"] = activation_price
    elif exchange.id in ["binance", "binanceusdm"]:
        if is_percentage:
            params["callbackRate"] = trail_value
        else:
            params["callbackValue"] = trail_value
        
        params["activatePrice"] = activation_price
    else:
        # For other exchanges that may have different parameter names
        logger.warning(f"Trailing stop implementation not specifically adapted for {exchange.id}. "
                      "Using generic parameters.")
        params["trailValue"] = trail_value
        params["activationPrice"] = activation_price
        params["isPercentage"] = is_percentage
    
    try:
        order = retry_api_call(
            exchange.create_order,
            symbol,
            "TRAILING_STOP_MARKET",  # Type might differ by exchange
            side,
            amount,
            None,  # No price for trailing stops
            params=params
        )
        
        logger.info(f"Trailing stop order placed: {order['id']}")
        return order
    except Exception as e:
        logger.error(f"Failed to execute trailing stop order: {e}")
        
        # Fall back to regular stop order if trailing stops not supported
        if "not supported" in str(e).lower():
            logger.warning(f"Trailing stops not supported. Falling back to regular stop order.")
            stop_price = activation_price
            
            if side == "sell" and is_percentage:  # Long position closing
                stop_price = activation_price * (1 - trail_value/100)
            elif side == "buy" and is_percentage:  # Short position closing
                stop_price = activation_price * (1 + trail_value/100)
            elif side == "sell":  # Long position closing with absolute value
                stop_price = activation_price - trail_value
            elif side == "buy":  # Short position closing with absolute value
                stop_price = activation_price + trail_value
            
            return execute_stop_order(
                exchange,
                symbol,
                side,
                amount,
                stop_price,
                stop_price,
                params=params
            )
        else:
            raise


def execute_oco_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    price: float,
    stop_price: float,
    params: Dict = None
) -> Dict:
    """
    Execute a One-Cancels-Other (OCO) order if supported by the exchange.
    This places a limit order and a stop order, and if one executes, the other is canceled.
    
    Args:
        exchange: Exchange instance
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        amount: Order amount in base currency
        price: Limit price (take profit)
        stop_price: Stop price (stop loss)
        params: Additional parameters for the order
        
    Returns:
        Dict: Order result
    """
    logger.info(f"Executing OCO {side} order for {amount} {symbol} "
               f"limit: {price}, stop: {stop_price}")
    
    params = params or {}
    
    # Add exchange-specific parameters
    if exchange.id in ["binance", "binanceusdm"]:
        params["stopLimitPrice"] = stop_price  # On Binance, stopLimitPrice is required
        params["stopPrice"] = stop_price
        params["price"] = price
        
        try:
            order = retry_api_call(
                exchange.create_order,
                symbol,
                "OCO",
                side,
                amount,
                price,
                params=params
            )
            
            logger.info(f"OCO order placed: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Failed to execute OCO order: {e}")
    
    # For exchanges that don't support OCO orders directly
    logger.warning(f"OCO orders not directly supported for {exchange.id}. "
                  "Placing separate limit and stop orders.")
    
    # Place limit order (take profit)
    try:
        limit_order = execute_limit_order(
            exchange,
            symbol,
            side,
            amount,
            price,
            params=params
        )
        
        # Place stop order (stop loss)
        stop_order = execute_stop_order(
            exchange,
            symbol,
            side,
            amount,
            stop_price,
            stop_price,
            params=params
        )
        
        # Return both orders
        return {
            "type": "manual_oco",
            "limit_order": limit_order,
            "stop_order": stop_order
        }
    except Exception as e:
        logger.error(f"Failed to execute manual OCO orders: {e}")
        raise


def execute_partial_position_entry(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    total_amount: float,
    partitions: List[float],
    price_levels: List[float] = None,
    order_types: List[str] = None,
    params: Dict = None
) -> Dict:
    """
    Execute a partial position entry with multiple orders at different price levels.
    
    Args:
        exchange: Exchange instance
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        total_amount: Total position size in base currency
        partitions: List of percentages for each order (should sum to 100)
        price_levels: List of price levels for each order (None for market price)
        order_types: List of order types ('market', 'limit') for each partition
        params: Additional parameters for the orders
        
    Returns:
        Dict: Order results
    """
    if price_levels is None:
        # Use current price for all orders
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker["last"]
        price_levels = [current_price] * len(partitions)
    
    if order_types is None:
        # Default to market orders
        order_types = ["market"] * len(partitions)
    
    if len(partitions) != len(price_levels) or len(partitions) != len(order_types):
        raise ValueError("Partitions, price levels, and order types must have the same length")
    
    # Normalize partitions to percentages
    total_percentage = sum(partitions)
    partitions = [p / total_percentage * 100 for p in partitions]
    
    orders = []
    params = params or {}
    
    logger.info(f"Executing partial position entry for {total_amount} {symbol} in {len(partitions)} parts")
    
    for i, (partition, price, order_type) in enumerate(zip(partitions, price_levels, order_types)):
        # Calculate amount for this partition
        amount = total_amount * partition / 100
        
        try:
            if order_type == "market":
                order = execute_market_order(
                    exchange, 
                    symbol, 
                    side, 
                    amount, 
                    params=params
                )
            elif order_type == "limit":
                order = execute_limit_order(
                    exchange, 
                    symbol, 
                    side, 
                    amount, 
                    price, 
                    params=params
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            orders.append(order)
            logger.info(f"Executed part {i+1}/{len(partitions)}: {amount} at {price} ({order_type})")
            
        except Exception as e:
            logger.error(f"Failed to execute part {i+1}/{len(partitions)}: {e}")
            # Continue with other parts even if one fails
    
    return {
        "type": "partial_entry",
        "orders": orders,
        "total_amount": total_amount,
        "executed_amount": sum(order["amount"] for order in orders if "amount" in order)
    }


def execute_position_with_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    stop_loss_price: float,
    take_profit_price: float = None,
    entry_price: float = None,
    order_type: str = "market",
    trailing_stop: Dict = None,
    partial_take_profits: List[Dict] = None,
    params: Dict = None
) -> Dict:
    """
    Execute a position with comprehensive protection orders (stop loss, take profit, trailing stop).
    
    Args:
        exchange: Exchange instance
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        amount: Order amount in base currency
        stop_loss_price: Stop loss price
        take_profit_price: Take profit price (None for no take profit)
        entry_price: Entry price for limit orders (None for market price)
        order_type: Order type ('market' or 'limit')
        trailing_stop: Trailing stop configuration (None for no trailing stop)
        partial_take_profits: List of partial take profit levels (None for no partial take profits)
        params: Additional parameters for the orders
        
    Returns:
        Dict: Complete position setup with all orders
    """
    logger.info(f"Setting up {order_type} {side} position for {amount} {symbol} "
               f"with stop loss at {stop_loss_price}")
    
    position_setup = {
        "symbol": symbol,
        "side": side,
        "amount": amount,
        "entry_order": None,
        "stop_loss_order": None,
        "take_profit_orders": [],
        "trailing_stop_order": None,
        "partial_take_profit_orders": []
    }
    
    params = params or {}
    
    # 1. Execute entry order
    try:
        if order_type == "market":
            entry_order = execute_market_order(
                exchange, 
                symbol, 
                side, 
                amount, 
                params=params
            )
        elif order_type == "limit" and entry_price is not None:
            entry_order = execute_limit_order(
                exchange, 
                symbol, 
                side, 
                amount, 
                entry_price, 
                params=params
            )
        else:
            raise ValueError(f"Invalid order type '{order_type}' or missing entry price")
        
        position_setup["entry_order"] = entry_order
        logger.info(f"Entry order executed: {entry_order['id']}")
        
        # For limit orders, we need to wait for fill before placing protection orders
        if order_type == "limit":
            logger.info(f"Limit order placed. Protection orders will be handled separately once filled.")
            return position_setup
        
        # Continue with protection orders for market orders
        
        # 2. Place stop loss order
        opposite_side = "sell" if side == "buy" else "buy"
        
        stop_loss_order = execute_stop_order(
            exchange, 
            symbol, 
            opposite_side, 
            amount, 
            stop_loss_price, 
            stop_loss_price, 
            params=params
        )
        
        position_setup["stop_loss_order"] = stop_loss_order
        logger.info(f"Stop loss order placed: {stop_loss_order['id']}")
        
        # 3. Place take profit order if specified
        if take_profit_price is not None:
            take_profit_order = execute_limit_order(
                exchange, 
                symbol, 
                opposite_side, 
                amount, 
                take_profit_price, 
                params=params
            )
            
            position_setup["take_profit_orders"].append(take_profit_order)
            logger.info(f"Take profit order placed: {take_profit_order['id']}")
        
        # 4. Place partial take profit orders if specified
        if partial_take_profits:
            for tp_level in partial_take_profits:
                tp_price = tp_level.get("level")
                tp_percentage = tp_level.get("percentage", 0)
                
                if tp_price and tp_percentage > 0:
                    # Calculate the amount for this partial TP
                    tp_amount = amount * (tp_percentage / 100)
                    
                    tp_order = execute_limit_order(
                        exchange, 
                        symbol, 
                        opposite_side, 
                        tp_amount, 
                        tp_price, 
                        params=params
                    )
                    
                    position_setup["partial_take_profit_orders"].append(tp_order)
                    logger.info(f"Partial take profit order placed: {tp_order['id']} "
                               f"for {tp_percentage}% at {tp_price}")
        
        # 5. Place trailing stop if specified
        if trailing_stop and trailing_stop.get("enabled", False):
            activation_pct = trailing_stop.get("activation_pct", 1.0)
            trail_pct = trailing_stop.get("trail_pct", 0.5)
            
            # Calculate activation price
            entry_execution_price = float(entry_order["price"])
            if entry_execution_price == 0 and "average" in entry_order:
                entry_execution_price = float(entry_order["average"])
            
            if side == "buy":  # Long position
                activation_price = entry_execution_price * (1.0 + activation_pct / 100.0)
            else:  # Short position
                activation_price = entry_execution_price * (1.0 - activation_pct / 100.0)
            
            try:
                trailing_stop_order = execute_trailing_stop_order(
                    exchange, 
                    symbol, 
                    opposite_side, 
                    amount, 
                    activation_price, 
                    trail_pct, 
                    is_percentage=True, 
                    params=params
                )
                
                position_setup["trailing_stop_order"] = trailing_stop_order
                logger.info(f"Trailing stop order placed: {trailing_stop_order['id']} "
                           f"activation: {activation_price}, trail: {trail_pct}%")
            except Exception as e:
                logger.error(f"Failed to place trailing stop order: {e}")
                logger.info("Will handle trailing stop through bot logic instead")
        
        return position_setup
        
    except Exception as e:
        logger.error(f"Failed to setup position with protection: {e}")
        
        # Cancel any orders placed if entry fails
        if position_setup["entry_order"] and position_setup["entry_order"]["status"] != "closed":
            try:
                exchange.cancel_order(position_setup["entry_order"]["id"], symbol)
                logger.info(f"Canceled entry order {position_setup['entry_order']['id']} after error")
            except Exception as cancel_error:
                logger.error(f"Failed to cancel entry order: {cancel_error}")
        
        raise