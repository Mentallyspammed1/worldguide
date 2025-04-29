"""
Order Management Module

This module handles order creation, submission, tracking, and management
with advanced features such as limit order placement, OCO orders,
trailing stops, and dynamic order sizing.
"""

import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from error_handling import handle_api_errors, retry, ExchangeAPIError, NetworkError
from utils import round_price, round_amount, generate_id

# Configure logger
logger = logging.getLogger("trading_bot.orders")

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"

class OrderBook:
    """Class to track and manage orders"""
    
    def __init__(self, max_orders: int = 1000):
        """
        Initialize order book
        
        Args:
            max_orders: Maximum number of orders to store
        """
        self.orders: Dict[str, Dict] = {}
        self.max_orders = max_orders
    
    def add_order(self, order: Dict) -> None:
        """
        Add order to order book
        
        Args:
            order: Order data
        """
        if "id" not in order:
            order["id"] = generate_id("order-")
        
        self.orders[order["id"]] = order
        
        # Prune if needed
        if len(self.orders) > self.max_orders:
            self._prune_old_orders()
    
    def update_order(self, order_id: str, updates: Dict) -> None:
        """
        Update order in order book
        
        Args:
            order_id: Order ID
            updates: Updates to apply
        """
        if order_id in self.orders:
            self.orders[order_id].update(updates)
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """
        Get order by ID
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict: Order data or None
        """
        return self.orders.get(order_id)
    
    def get_orders_by_status(self, status: str) -> List[Dict]:
        """
        Get orders by status
        
        Args:
            status: Order status
            
        Returns:
            List[Dict]: Matching orders
        """
        return [order for order in self.orders.values() 
                if order.get("status") == status]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Dict]:
        """
        Get orders for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List[Dict]: Matching orders
        """
        return [order for order in self.orders.values() 
                if order.get("symbol") == symbol]
    
    def _prune_old_orders(self) -> None:
        """Remove oldest closed/canceled orders to stay under max_orders"""
        # First try to remove canceled or rejected orders
        closed_orders = sorted(
            [order for order in self.orders.values() 
             if order.get("status") in ["canceled", "rejected", "expired"]],
            key=lambda x: x.get("timestamp", 0)
        )
        
        # Then try closed orders if still needed
        if len(self.orders) - len(closed_orders) > self.max_orders:
            closed_orders += sorted(
                [order for order in self.orders.values() 
                 if order.get("status") == "closed"],
                key=lambda x: x.get("timestamp", 0)
            )
        
        # Remove oldest orders first
        for order in closed_orders:
            if len(self.orders) <= self.max_orders:
                break
            if order["id"] in self.orders:
                del self.orders[order["id"]]
                logger.debug(f"Pruned old order {order['id']} from order book")


@handle_api_errors
@retry(max_tries=3, delay=1.0, backoff_factor=2.0)
def create_market_order(
    exchange: Any,
    symbol: str,
    side: str,
    amount: float,
    price: Optional[float] = None,
    params: Dict = None
) -> Dict:
    """
    Create and submit a market order
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        amount: Order amount
        price: Current price for estimation (optional)
        params: Additional parameters (optional)
        
    Returns:
        Dict: Order result
    """
    logger.info(f"Creating market {side} order for {amount} {symbol}")
    
    try:
        # Create order
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=amount,
            params=params or {}
        )
        
        logger.info(f"Market order created: {order['id']} ({side} {amount} {symbol})")
        
        return order
    except Exception as e:
        logger.error(f"Error creating market order: {e}")
        raise


@handle_api_errors
@retry(max_tries=3, delay=1.0, backoff_factor=2.0)
def create_limit_order(
    exchange: Any,
    symbol: str,
    side: str,
    amount: float,
    price: float,
    params: Dict = None
) -> Dict:
    """
    Create and submit a limit order
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        amount: Order amount
        price: Limit price
        params: Additional parameters (optional)
        
    Returns:
        Dict: Order result
    """
    logger.info(f"Creating limit {side} order for {amount} {symbol} at {price}")
    
    try:
        # Create order
        order = exchange.create_order(
            symbol=symbol,
            type='limit',
            side=side,
            amount=amount,
            price=price,
            params=params or {}
        )
        
        logger.info(f"Limit order created: {order['id']} ({side} {amount} {symbol} @ {price})")
        
        return order
    except Exception as e:
        logger.error(f"Error creating limit order: {e}")
        raise


@handle_api_errors
@retry(max_tries=3, delay=1.0, backoff_factor=2.0)
def create_stop_order(
    exchange: Any,
    symbol: str,
    side: str,
    amount: float,
    stop_price: float,
    params: Dict = None
) -> Dict:
    """
    Create and submit a stop order
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        amount: Order amount
        stop_price: Stop price
        params: Additional parameters (optional)
        
    Returns:
        Dict: Order result
    """
    logger.info(f"Creating stop {side} order for {amount} {symbol} at {stop_price}")
    
    try:
        # Check if exchange has native stop order support
        if 'createStopOrder' in dir(exchange):
            # Create stop order
            order = exchange.create_stop_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=stop_price,
                params=params or {}
            )
        else:
            # Fallback to stop market or stop limit order
            order_type = 'stop_market'  # Default
            
            # Check if exchange supports stop_market
            if order_type not in exchange.has or not exchange.has[order_type]:
                order_type = 'stop'  # Try regular stop
            
            if order_type not in exchange.has or not exchange.has[order_type]:
                order_type = 'stop_limit'  # Try stop limit
                
            if order_type not in exchange.has or not exchange.has[order_type]:
                raise ValueError(f"Exchange does not support stop orders: {exchange.id}")
            
            # Merge stop price into params
            stop_params = params.copy() if params else {}
            if order_type == 'stop_limit':
                stop_params['stopPrice'] = stop_price
                # For stop limit, we need a limit price too, slightly worse than stop
                limit_price = stop_price * (0.99 if side == 'sell' else 1.01)
                order = exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=limit_price,
                    params=stop_params
                )
            else:
                stop_params['stopPrice'] = stop_price
                order = exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=stop_price,  # Some exchanges use this as trigger
                    params=stop_params
                )
        
        logger.info(f"Stop order created: {order['id']} ({side} {amount} {symbol} @ {stop_price})")
        
        return order
    except Exception as e:
        logger.error(f"Error creating stop order: {e}")
        raise


@handle_api_errors
@retry(max_tries=3, delay=1.0, backoff_factor=2.0)
def create_oco_order(
    exchange: Any,
    symbol: str,
    side: str,
    amount: float,
    stop_price: float,
    limit_price: float,
    params: Dict = None
) -> Union[Dict, List[Dict]]:
    """
    Create a one-cancels-other (OCO) order (take profit and stop loss)
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        amount: Order amount
        stop_price: Stop loss price
        limit_price: Take profit price
        params: Additional parameters (optional)
        
    Returns:
        Union[Dict, List[Dict]]: Order result(s)
    """
    logger.info(f"Creating OCO {side} order for {amount} {symbol} (stop: {stop_price}, limit: {limit_price})")
    
    try:
        # Check if exchange has native OCO support
        if 'createOCOOrder' in dir(exchange) or ('oco' in exchange.has and exchange.has['oco']):
            if hasattr(exchange, 'create_oco_order'):
                # Use native OCO function
                order = exchange.create_oco_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=limit_price,
                    stop_price=stop_price,
                    params=params or {}
                )
                logger.info(f"OCO order created: {order['id']}")
                return order
            else:
                # Try standard format with params
                oco_params = params.copy() if params else {}
                oco_params['stopPrice'] = stop_price
                
                order = exchange.create_order(
                    symbol=symbol,
                    type='oco',
                    side=side,
                    amount=amount,
                    price=limit_price,
                    params=oco_params
                )
                logger.info(f"OCO order created: {order['id']}")
                return order
        else:
            # Fallback: Create two separate orders
            logger.info(f"Exchange does not support OCO orders, creating separate stop and limit orders")
            
            # Create limit order (take profit)
            limit_order = create_limit_order(
                exchange=exchange,
                symbol=symbol,
                side=side,
                amount=amount,
                price=limit_price,
                params=params
            )
            
            # Create stop order (stop loss)
            stop_order = create_stop_order(
                exchange=exchange,
                symbol=symbol,
                side=side,
                amount=amount,
                stop_price=stop_price,
                params=params
            )
            
            # Return both orders
            return [limit_order, stop_order]
    except Exception as e:
        logger.error(f"Error creating OCO order: {e}")
        raise


@handle_api_errors
@retry(max_tries=3, delay=1.0, backoff_factor=2.0)
def cancel_order(
    exchange: Any,
    order_id: str,
    symbol: str,
    params: Dict = None
) -> Dict:
    """
    Cancel an open order
    
    Args:
        exchange: CCXT exchange instance
        order_id: Order ID
        symbol: Trading symbol
        params: Additional parameters (optional)
        
    Returns:
        Dict: Cancel result
    """
    logger.info(f"Canceling order {order_id} for {symbol}")
    
    try:
        result = exchange.cancel_order(
            id=order_id,
            symbol=symbol,
            params=params or {}
        )
        
        logger.info(f"Order {order_id} canceled")
        
        return result
    except Exception as e:
        logger.error(f"Error canceling order: {e}")
        raise


@handle_api_errors
@retry(max_tries=3, delay=1.0, backoff_factor=2.0)
def get_order_status(
    exchange: Any,
    order_id: str,
    symbol: str,
    params: Dict = None
) -> Dict:
    """
    Get status of an order
    
    Args:
        exchange: CCXT exchange instance
        order_id: Order ID
        symbol: Trading symbol
        params: Additional parameters (optional)
        
    Returns:
        Dict: Order status
    """
    try:
        result = exchange.fetch_order(
            id=order_id,
            symbol=symbol,
            params=params or {}
        )
        
        logger.debug(f"Order {order_id} status: {result.get('status')}")
        
        return result
    except Exception as e:
        logger.error(f"Error fetching order status: {e}")
        raise


@handle_api_errors
@retry(max_tries=3, delay=1.0, backoff_factor=2.0)
def get_open_orders(
    exchange: Any,
    symbol: Optional[str] = None,
    params: Dict = None
) -> List[Dict]:
    """
    Get all open orders
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol (optional)
        params: Additional parameters (optional)
        
    Returns:
        List[Dict]: List of open orders
    """
    try:
        if symbol:
            result = exchange.fetch_open_orders(
                symbol=symbol,
                params=params or {}
            )
            logger.debug(f"Found {len(result)} open orders for {symbol}")
        else:
            result = exchange.fetch_open_orders(
                params=params or {}
            )
            logger.debug(f"Found {len(result)} open orders across all symbols")
        
        return result
    except Exception as e:
        logger.error(f"Error fetching open orders: {e}")
        raise


@handle_api_errors
@retry(max_tries=3, delay=1.0, backoff_factor=2.0)
def cancel_all_orders(
    exchange: Any,
    symbol: Optional[str] = None,
    params: Dict = None
) -> List[Dict]:
    """
    Cancel all open orders
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading symbol (optional)
        params: Additional parameters (optional)
        
    Returns:
        List[Dict]: Cancel results
    """
    try:
        if hasattr(exchange, 'cancel_all_orders'):
            # Use native function if available
            if symbol:
                result = exchange.cancel_all_orders(
                    symbol=symbol,
                    params=params or {}
                )
                logger.info(f"Canceled all orders for {symbol}")
            else:
                result = exchange.cancel_all_orders(
                    params=params or {}
                )
                logger.info("Canceled all orders across all symbols")
            
            return result
        else:
            # Fallback: Fetch open orders and cancel individually
            open_orders = get_open_orders(exchange, symbol, params)
            
            results = []
            for order in open_orders:
                try:
                    result = cancel_order(
                        exchange=exchange,
                        order_id=order['id'],
                        symbol=order['symbol'],
                        params=params
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error canceling order {order['id']}: {e}")
            
            logger.info(f"Canceled {len(results)} orders out of {len(open_orders)} open orders")
            
            return results
    except Exception as e:
        logger.error(f"Error canceling all orders: {e}")
        raise


def place_entry_order(
    exchange: Any,
    strategy: str,
    symbol: str,
    side: str,
    amount: float,
    price: float,
    order_type: str,
    signal_params: Dict = None,
    prefered_entry: str = "limit_better",
    max_slippage_pct: float = 0.5,  # Max slippage for limit orders
    order_params: Dict = None
) -> Dict:
    """
    Place an entry order with smart order type selection
    
    Args:
        exchange: CCXT exchange instance
        strategy: Strategy name
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        amount: Order amount
        price: Current price
        order_type: Order type from signal ('market', 'limit')
        signal_params: Signal parameters (optional)
        prefered_entry: Entry preference ('market', 'limit_better', 'limit_chase')
        max_slippage_pct: Maximum allowed slippage percentage
        order_params: Additional order parameters (optional)
        
    Returns:
        Dict: Order result
    """
    # Normalize side
    side = side.lower()
    
    # Get market info
    market = exchange.market(symbol)
    price_precision = market.get('precision', {}).get('price', 8)
    
    # Override order type if preferred entry is market
    if prefered_entry == "market":
        order_type = "market"
    
    # For limit_better, use limit at a better price
    if prefered_entry == "limit_better" and order_type == "limit":
        if side == "buy":
            # For buy, set limit below market
            limit_price = price * (1 - max_slippage_pct / 100)
        else:
            # For sell, set limit above market
            limit_price = price * (1 + max_slippage_pct / 100)
        
        limit_price = round_price(limit_price, price_precision)
        
        logger.info(f"Using better limit price for {side}: {price} -> {limit_price}")
        
        return create_limit_order(
            exchange=exchange,
            symbol=symbol,
            side=side,
            amount=amount,
            price=limit_price,
            params=order_params
        )
    
    # Default to market order
    return create_market_order(
        exchange=exchange,
        symbol=symbol,
        side=side,
        amount=amount,
        price=price,
        params=order_params
    )


def place_exit_orders(
    exchange: Any,
    position: Dict,
    stop_price: float,
    take_profit_price: float,
    order_types: Dict = None,
    use_oco: bool = True,
    order_params: Dict = None
) -> Union[Dict, List[Dict]]:
    """
    Place exit orders for a position (stop loss and take profit)
    
    Args:
        exchange: CCXT exchange instance
        position: Position data
        stop_price: Stop loss price
        take_profit_price: Take profit price
        order_types: Order types for each exit (optional)
        use_oco: Whether to use OCO orders if supported
        order_params: Additional order parameters (optional)
        
    Returns:
        Union[Dict, List[Dict]]: Order result(s)
    """
    # Extract position details
    symbol = position["symbol"]
    side = position["side"]
    amount = position["amount"]
    
    # Determine exit side (opposite of position side)
    exit_side = "sell" if side == "long" else "buy"
    
    # If exchange supports OCO, use it
    if use_oco:
        try:
            oco_result = create_oco_order(
                exchange=exchange,
                symbol=symbol,
                side=exit_side,
                amount=amount,
                stop_price=stop_price,
                limit_price=take_profit_price,
                params=order_params
            )
            
            return oco_result
        except Exception as e:
            logger.warning(f"Failed to create OCO order, falling back to separate orders: {e}")
    
    # Create separate stop and limit orders
    orders = []
    
    # Stop loss order
    stop_order_type = order_types.get("stop", "stop") if order_types else "stop"
    try:
        stop_order = create_stop_order(
            exchange=exchange,
            symbol=symbol,
            side=exit_side,
            amount=amount,
            stop_price=stop_price,
            params=order_params
        )
        orders.append(stop_order)
    except Exception as e:
        logger.error(f"Failed to create stop loss order: {e}")
    
    # Take profit order
    try:
        take_profit_order = create_limit_order(
            exchange=exchange,
            symbol=symbol,
            side=exit_side,
            amount=amount,
            price=take_profit_price,
            params=order_params
        )
        orders.append(take_profit_order)
    except Exception as e:
        logger.error(f"Failed to create take profit order: {e}")
    
    return orders


def update_exit_orders(
    exchange: Any,
    position: Dict,
    orders: Dict,
    new_stop_price: Optional[float] = None,
    new_take_profit_price: Optional[float] = None,
    order_params: Dict = None
) -> Dict:
    """
    Update exit orders for a position
    
    Args:
        exchange: CCXT exchange instance
        position: Position data
        orders: Current exit orders
        new_stop_price: New stop loss price (optional)
        new_take_profit_price: New take profit price (optional)
        order_params: Additional order parameters (optional)
        
    Returns:
        Dict: Updated orders
    """
    # Extract position details
    symbol = position["symbol"]
    side = position["side"]
    amount = position["amount"]
    
    # Determine exit side (opposite of position side)
    exit_side = "sell" if side == "long" else "buy"
    
    # Update stop loss if needed
    if new_stop_price and "stop_loss" in orders:
        try:
            # Cancel current stop loss
            cancel_order(
                exchange=exchange,
                order_id=orders["stop_loss"]["id"],
                symbol=symbol
            )
            
            # Create new stop loss
            stop_order = create_stop_order(
                exchange=exchange,
                symbol=symbol,
                side=exit_side,
                amount=amount,
                stop_price=new_stop_price,
                params=order_params
            )
            
            orders["stop_loss"] = stop_order
            logger.info(f"Updated stop loss for {symbol}: {new_stop_price}")
            
        except Exception as e:
            logger.error(f"Failed to update stop loss: {e}")
    
    # Update take profit if needed
    if new_take_profit_price and "take_profit" in orders:
        try:
            # Cancel current take profit
            cancel_order(
                exchange=exchange,
                order_id=orders["take_profit"]["id"],
                symbol=symbol
            )
            
            # Create new take profit
            take_profit_order = create_limit_order(
                exchange=exchange,
                symbol=symbol,
                side=exit_side,
                amount=amount,
                price=new_take_profit_price,
                params=order_params
            )
            
            orders["take_profit"] = take_profit_order
            logger.info(f"Updated take profit for {symbol}: {new_take_profit_price}")
            
        except Exception as e:
            logger.error(f"Failed to update take profit: {e}")
    
    return orders