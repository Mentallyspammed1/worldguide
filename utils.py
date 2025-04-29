"""
Utility Functions for Trading Bot

This module provides utility functions for the trading bot,
including CCXT exchange setup, API retry mechanism, and other helpers.
"""

import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import ccxt

# Type variable for generic return type
T = TypeVar('T')

# Configure logger
logger = logging.getLogger("utils")


def setup_ccxt_exchange(
    exchange_id: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = False
) -> ccxt.Exchange:
    """
    Set up a CCXT exchange with proper configuration
    
    Args:
        exchange_id: ID of the exchange (e.g., 'bybit', 'binance')
        api_key: API key for the exchange
        api_secret: API secret for the exchange
        testnet: Whether to use testnet/sandbox
        
    Returns:
        ccxt.Exchange: Configured exchange instance
    """
    # Validate exchange ID
    if exchange_id not in ccxt.exchanges:
        raise ValueError(f"Exchange '{exchange_id}' not supported by CCXT")
    
    # Prepare exchange options
    options = {
        "adjustForTimeDifference": True,  # Sync time with exchange
        "recvWindow": 60000,  # Extended window for high-latency connections
    }
    
    # Exchange-specific configurations
    if exchange_id == 'bybit':
        options["defaultType"] = "linear"
    elif exchange_id == 'binance':
        options["defaultType"] = "future"
    
    # Determine URLs based on testnet setting
    urls = {}
    if testnet:
        if exchange_id == 'bybit':
            urls["api"] = "https://api-testnet.bybit.com"
        elif exchange_id == 'binance':
            urls["api"] = "https://testnet.binancefuture.com"
    
    # Create the exchange instance
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': options,
        'urls': urls if urls else None,
        'timeout': 30000,  # 30 seconds timeout
    })
    
    # Load markets to ensure exchange is working
    try:
        exchange.load_markets()
        logger.info(f"CCXT {exchange_id} exchange instance created and markets loaded")
    except Exception as e:
        logger.error(f"Error loading markets for {exchange_id}: {e}")
        # Still return the exchange instance, as markets can be loaded later
    
    return exchange


def retry_api_call(
    func: Callable[[], T],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    allowed_exceptions: tuple = (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout)
) -> T:
    """
    Retry an API call with exponential backoff
    
    Args:
        func: Function to call
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier applied to delay between retries
        allowed_exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        T: Result of the function call
    """
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        try:
            return func()
        except allowed_exceptions as e:
            retries += 1
            last_exception = e
            
            if retries <= max_retries:
                sleep_time = retry_delay * (backoff_factor ** (retries - 1))
                logger.warning(f"API call failed: {e}. Retrying in {sleep_time:.2f} seconds ({retries}/{max_retries})")
                time.sleep(sleep_time)
            else:
                logger.error(f"API call failed after {max_retries} retries: {e}")
                raise
        except Exception as e:
            # For other exceptions, don't retry
            logger.error(f"API call failed with non-retryable error: {e}")
            raise
    
    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError("Retry mechanism failed for unknown reason")


def parse_timeframe(timeframe: str) -> Dict[str, int]:
    """
    Parse a timeframe string (e.g., '15m', '1h', '1d') to get the unit and value
    
    Args:
        timeframe: Timeframe string
        
    Returns:
        Dict[str, int]: Dictionary with 'unit' and 'value' keys
    """
    units = {
        's': 'seconds',
        'm': 'minutes',
        'h': 'hours',
        'd': 'days',
        'w': 'weeks',
        'M': 'months'
    }
    
    if not timeframe:
        raise ValueError("Timeframe cannot be empty")
    
    # Extract value and unit
    for unit_char, unit_name in units.items():
        if timeframe.endswith(unit_char):
            try:
                value = int(timeframe[:-1])
                return {'unit': unit_name, 'value': value}
            except ValueError:
                raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    raise ValueError(f"Unknown timeframe unit in: {timeframe}")


def safe_float(obj: Dict, key: str, default: float = 0.0) -> float:
    """
    Safely extract a float value from a dictionary
    
    Args:
        obj: Dictionary to extract from
        key: Key to look up
        default: Default value if key is missing or value is not a valid float
        
    Returns:
        float: Extracted float value or default
    """
    try:
        value = obj.get(key)
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def timestamp_to_datetime(timestamp: int, ms: bool = True) -> str:
    """
    Convert a Unix timestamp to a human-readable datetime string
    
    Args:
        timestamp: Unix timestamp
        ms: Whether the timestamp is in milliseconds (True) or seconds (False)
        
    Returns:
        str: Formatted datetime string
    """
    if ms and timestamp > 1e10:  # Likely milliseconds if very large
        timestamp = timestamp / 1000
    
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(timestamp))


def limit_decimal_places(value: float, places: int = 8) -> float:
    """
    Limit a float to a specific number of decimal places (truncate, not round)
    
    Args:
        value: Float value to limit
        places: Number of decimal places
        
    Returns:
        float: Value with limited decimal places
    """
    if not isinstance(value, (int, float)):
        return 0.0
    
    # Truncate decimal places
    multiplier = 10 ** places
    truncated = int(value * multiplier) / multiplier
    
    return truncated


def calculate_change_percent(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        float: Percentage change
    """
    if old_value == 0:
        return 0.0
    
    return ((new_value - old_value) / abs(old_value)) * 100.0