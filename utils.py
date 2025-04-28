"""
Utility Functions for Trading Bot

This module contains utility functions for the trading bot:
- API interaction with retry mechanism
- Exchange setup and configuration
- Timeframe parsing and conversion
- Error handling and logging utilities
"""

import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ccxt
import pandas as pd

# Configure logger
logger = logging.getLogger("utils")

# Mapping from string timeframes to seconds
TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000,
}


def parse_timeframe(timeframe: str) -> int:
    """
    Parse timeframe string to seconds.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        int: Timeframe in seconds
    """
    return TIMEFRAME_SECONDS.get(timeframe, 0)


def setup_ccxt_exchange(
    exchange_id: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    options: Optional[Dict] = None,
    params: Optional[Dict] = None
) -> Optional[ccxt.Exchange]:
    """
    Set up and configure a CCXT exchange instance.
    
    Args:
        exchange_id: Exchange ID (e.g., 'bybit', 'binance')
        api_key: API key for authentication
        api_secret: API secret for authentication
        options: Exchange-specific options
        params: Additional parameters for exchange initialization
        
    Returns:
        ccxt.Exchange: Configured exchange instance
    """
    # Default config
    config = {
        "apiKey": api_key,
        "secret": api_secret,
        "timeout": 30000,  # 30 seconds timeout
        "enableRateLimit": True,
    }
    
    # Add options if provided
    if options:
        config["options"] = options
    
    # Add params if provided
    if params:
        for key, value in params.items():
            config[key] = value
    
    try:
        # Get exchange class
        if not hasattr(ccxt, exchange_id):
            logger.error(f"Unsupported exchange: {exchange_id}")
            return None
        
        exchange_class = getattr(ccxt, exchange_id)
        
        # Create exchange instance
        exchange = exchange_class(config)
        
        logger.info(f"CCXT {exchange_id} exchange initialized")
        return exchange
    except Exception as e:
        logger.error(f"Error initializing {exchange_id} exchange: {e}")
        return None


def retry_api_call(
    func: Callable,
    *args: Any,
    max_retries: int = 3,
    retry_delay: int = 5,
    **kwargs: Any
) -> Any:
    """
    Retry an API call with exponential backoff.
    
    Args:
        func: Function to call
        *args: Positional arguments for the function
        max_retries: Maximum number of retries
        retry_delay: Initial delay in seconds
        **kwargs: Keyword arguments for the function
        
    Returns:
        Any: Result of the function call
        
    Raises:
        Exception: Last encountered exception after all retries
    """
    last_exception = None
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            result = func(*args, **kwargs)
            return result
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            # Rate limit reached, use a longer delay
            sleep_time = retry_delay * (2 ** retry_count) + 0.5  # Add extra buffer
            logger.warning(
                f"Rate limit exceeded. Retrying in {sleep_time:.1f} seconds... "
                f"({retry_count+1}/{max_retries+1})"
            )
            time.sleep(sleep_time)
        except ccxt.NetworkError as e:
            last_exception = e
            # Network error, use standard delay
            sleep_time = retry_delay * (1.5 ** retry_count)
            logger.warning(
                f"Network error: {str(e)}. Retrying in {sleep_time:.1f} seconds... "
                f"({retry_count+1}/{max_retries+1})"
            )
            time.sleep(sleep_time)
        except ccxt.ExchangeError as e:
            last_exception = e
            if "Too many requests" in str(e) or "rate limit" in str(e).lower():
                # Handle exchange-specific rate limit errors
                sleep_time = retry_delay * (2 ** retry_count) + 1  # Add extra buffer
                logger.warning(
                    f"Exchange rate limit: {str(e)}. Retrying in {sleep_time:.1f} seconds... "
                    f"({retry_count+1}/{max_retries+1})"
                )
                time.sleep(sleep_time)
            else:
                # General exchange error, might be permanent
                logger.error(f"Exchange error: {str(e)}. Retrying...")
                time.sleep(retry_delay)
        except Exception as e:
            # Unexpected error, log and raise
            logger.error(f"Unexpected error in API call: {str(e)}")
            raise
        
        retry_count += 1
    
    # If we get here, we've exhausted retries
    if last_exception:
        logger.error(f"Failed after {max_retries} retries. Last error: {str(last_exception)}")
        raise last_exception
    
    return None


def round_to_tick(
    price: float,
    tick_size: float,
    rounding_mode: str = "down"
) -> float:
    """
    Round price to the nearest tick size.
    
    Args:
        price: Price to round
        tick_size: Tick size (minimum price movement)
        rounding_mode: Rounding mode ('up', 'down', or 'nearest')
        
    Returns:
        float: Rounded price
    """
    decimal_tick = Decimal(str(tick_size))
    decimal_price = Decimal(str(price))
    
    # Count decimal places in tick size
    tick_decimals = abs(decimal_tick.as_tuple().exponent)
    
    # Calculate division
    ticks = decimal_price / decimal_tick
    
    # Round according to mode
    if rounding_mode == "up":
        rounded_ticks = ticks.quantize(Decimal("1"), rounding=ROUND_UP)
    elif rounding_mode == "down":
        rounded_ticks = ticks.quantize(Decimal("1"), rounding=ROUND_DOWN)
    else:  # nearest
        rounded_ticks = ticks.quantize(Decimal("1"))
    
    # Multiply back by tick size
    rounded_price = rounded_ticks * decimal_tick
    
    # Convert back to float with appropriate precision
    return float(rounded_price.quantize(Decimal("0." + "0" * tick_decimals)))


def calculate_candle_interval(timeframe: str) -> Tuple[datetime, datetime]:
    """
    Calculate start and end datetime for the current candle interval.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        Tuple[datetime, datetime]: Start and end datetime
    """
    now = datetime.utcnow()
    seconds = parse_timeframe(timeframe)
    
    if seconds == 0:
        # Invalid timeframe
        return now, now
    
    # Special handling for daily, weekly, monthly candles
    if timeframe == "1d":
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)
    elif timeframe == "1w":
        # Get current week's Monday
        start = now - timedelta(days=now.weekday())
        start = datetime(start.year, start.month, start.day)
        end = start + timedelta(days=7)
    elif timeframe == "1M":
        # Get current month's first day
        start = datetime(now.year, now.month, 1)
        # Move to next month's first day
        if now.month == 12:
            end = datetime(now.year + 1, 1, 1)
        else:
            end = datetime(now.year, now.month + 1, 1)
    else:
        # Regular intervals
        timestamp = int(now.timestamp())
        interval_count = timestamp // seconds
        start = datetime.fromtimestamp(interval_count * seconds)
        end = datetime.fromtimestamp((interval_count + 1) * seconds)
    
    return start, end


def format_price_quantity(
    price: float,
    quantity: float,
    price_precision: int,
    quantity_precision: int
) -> Tuple[float, float]:
    """
    Format price and quantity to exchange precision.
    
    Args:
        price: Price value
        quantity: Quantity value
        price_precision: Decimal places for price
        quantity_precision: Decimal places for quantity
        
    Returns:
        Tuple[float, float]: Formatted price and quantity
    """
    # Format price
    formatted_price = round(price, price_precision)
    
    # Format quantity
    formatted_quantity = round(quantity, quantity_precision)
    
    return formatted_price, formatted_quantity