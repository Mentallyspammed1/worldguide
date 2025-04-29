"""
Utility Functions

This module provides utility functions used throughout the trading bot,
including retry mechanisms, time formatting, validation, and logging.
"""

import time
import random
import logging
import functools
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, cast, Tuple

# Type variable for generic function
T = TypeVar('T')

# Configure logger
logger = logging.getLogger("trading_bot.utils")

def setup_directory_structure():
    """
    Create directory structure for the trading bot.
    """
    # Create essential directories
    dirs = ["bot_logs", "data", "backtest_results", "models", "config"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"Created directory: {d}")

def retry_api_call(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Retry an API call with exponential backoff.
    
    Args:
        func: Function to call
        *args: Arguments to pass to func
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Result of function call
    """
    max_retries = 5
    retry_delay = 1.0  # seconds
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                # Rate limit error - use longer delays
                retry_delay = min(60, retry_delay * 2)
            elif "network" in str(e).lower() or "timeout" in str(e).lower():
                # Network error - use moderate delays
                retry_delay = min(30, retry_delay * 1.5)
            else:
                # Other errors - use standard delays
                retry_delay = min(15, retry_delay * 1.2)
            
            # Add randomness to avoid thundering herd
            jitter = random.uniform(0.1, 0.3) * retry_delay
            sleep_time = retry_delay + jitter
            
            # Last attempt - raise the error
            if attempt == max_retries - 1:
                raise
            
            # Log and sleep before retry
            logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}. "
                          f"Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)
    
    # This should never be reached due to the raise in the loop
    raise RuntimeError("Retry logic failed")

def format_timestamp(timestamp_ms: int, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp in milliseconds to a human-readable string.
    
    Args:
        timestamp_ms: Timestamp in milliseconds
        format_str: Format string for strftime
        
    Returns:
        str: Formatted timestamp
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime(format_str)

def time_ago(timestamp_ms: int) -> str:
    """
    Convert a timestamp to a "time ago" string.
    
    Args:
        timestamp_ms: Timestamp in milliseconds
        
    Returns:
        str: Human-readable time ago
    """
    now = datetime.now()
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds/60)} minutes ago"
    elif seconds < 86400:
        return f"{int(seconds/3600)} hours ago"
    elif seconds < 604800:
        return f"{int(seconds/86400)} days ago"
    elif seconds < 2592000:
        return f"{int(seconds/604800)} weeks ago"
    elif seconds < 31536000:
        return f"{int(seconds/2592000)} months ago"
    else:
        return f"{int(seconds/31536000)} years ago"

def round_price(price: float, precision: int) -> float:
    """
    Round price to specified precision.
    
    Args:
        price: Price to round
        precision: Decimal places
        
    Returns:
        float: Rounded price
    """
    factor = 10 ** precision
    return round(price * factor) / factor

def round_amount(amount: float, precision: int) -> float:
    """
    Round amount to specified precision.
    
    Args:
        amount: Amount to round
        precision: Decimal places
        
    Returns:
        float: Rounded amount
    """
    factor = 10 ** precision
    return round(amount * factor) / factor

def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format.
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        bool: True if valid
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Check for common formats: BTC/USDT, BTC/USDT:USDT, BTC-USDT
    parts = symbol.replace(':', '/').replace('-', '/').split('/')
    
    if len(parts) < 2:
        return False
    
    base = parts[0]
    quote = parts[1]
    
    return len(base) > 0 and len(quote) > 0

def validate_timeframe(timeframe: str) -> bool:
    """
    Validate timeframe format.
    
    Args:
        timeframe: Timeframe to validate
        
    Returns:
        bool: True if valid
    """
    if not timeframe or not isinstance(timeframe, str):
        return False
    
    # Check common timeframe formats: 1m, 5m, 15m, 1h, 4h, 1d
    valid_units = ['m', 'h', 'd', 'w', 'M']
    if len(timeframe) < 2:
        return False
    
    number = timeframe[:-1]
    unit = timeframe[-1]
    
    if not number.isdigit() or unit not in valid_units:
        return False
    
    return True

def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert timeframe to seconds.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        int: Timeframe in seconds
    """
    if not validate_timeframe(timeframe):
        raise ValueError(f"Invalid timeframe: {timeframe}")
    
    number = int(timeframe[:-1])
    unit = timeframe[-1]
    
    if unit == 'm':
        return number * 60
    elif unit == 'h':
        return number * 60 * 60
    elif unit == 'd':
        return number * 24 * 60 * 60
    elif unit == 'w':
        return number * 7 * 24 * 60 * 60
    elif unit == 'M':
        return number * 30 * 24 * 60 * 60
    
    raise ValueError(f"Unsupported timeframe unit: {unit}")

def timeframe_to_milliseconds(timeframe: str) -> int:
    """
    Convert timeframe to milliseconds.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        int: Timeframe in milliseconds
    """
    return timeframe_to_seconds(timeframe) * 1000

def generate_id(prefix: str = "") -> str:
    """
    Generate a unique ID.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        str: Unique ID
    """
    timestamp = int(time.time() * 1000)
    random_part = random.randint(0, 999999)
    hash_input = f"{timestamp}{random_part}{os.getpid()}"
    hash_part = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    return f"{prefix}{timestamp}-{hash_part}"

def load_json_file(filepath: str, default: Optional[Any] = None) -> Any:
    """
    Load JSON from file with error handling.
    
    Args:
        filepath: Path to JSON file
        default: Default value if file cannot be loaded
        
    Returns:
        Any: Loaded JSON data or default
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return default if default is not None else {}

def save_json_file(data: Any, filepath: str, indent: int = 2) -> bool:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        filepath: Path to save to
        indent: JSON indentation
        
    Returns:
        bool: True if successful
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except (PermissionError, IOError) as e:
        logger.error(f"Failed to save {filepath}: {e}")
        return False

def parse_iso_timestamp(timestamp_str: str) -> int:
    """
    Parse ISO timestamp to milliseconds.
    
    Args:
        timestamp_str: ISO timestamp string
        
    Returns:
        int: Timestamp in milliseconds
    """
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return int(dt.timestamp() * 1000)
    except ValueError:
        logger.warning(f"Failed to parse timestamp: {timestamp_str}")
        return 0

def format_money(amount: float, currency: str = "USD", precision: int = 2) -> str:
    """
    Format amount as money.
    
    Args:
        amount: Amount to format
        currency: Currency code
        precision: Decimal places
        
    Returns:
        str: Formatted money string
    """
    if currency.upper() in ["USD", "USDT", "USDC"]:
        prefix = "$"
        suffix = ""
    elif currency.upper() in ["BTC", "ETH", "XBT"]:
        prefix = ""
        suffix = f" {currency.upper()}"
    else:
        prefix = ""
        suffix = f" {currency.upper()}"
    
    return f"{prefix}{amount:.{precision}f}{suffix}"

def format_pnl(pnl: float, percentage: bool = False) -> str:
    """
    Format PnL with color indicator.
    
    Args:
        pnl: PnL value
        percentage: Whether value is a percentage
        
    Returns:
        str: Formatted PnL string with color indicator
    """
    if pnl > 0:
        color_code = "\033[92m"  # Green
        prefix = "+"
    elif pnl < 0:
        color_code = "\033[91m"  # Red
        prefix = ""
    else:
        color_code = "\033[93m"  # Yellow
        prefix = ""
    
    reset_code = "\033[0m"
    
    if percentage:
        return f"{color_code}{prefix}{pnl:.2f}%{reset_code}"
    else:
        return f"{color_code}{prefix}{pnl:.2f}{reset_code}"

def calculate_change(old_value: float, new_value: float) -> Tuple[float, float]:
    """
    Calculate absolute and percentage change.
    
    Args:
        old_value: Old value
        new_value: New value
        
    Returns:
        Tuple[float, float]: (absolute_change, percentage_change)
    """
    absolute_change = new_value - old_value
    
    if old_value == 0:
        percentage_change = 0 if absolute_change == 0 else float('inf')
    else:
        percentage_change = (absolute_change / old_value) * 100
    
    return absolute_change, percentage_change

def throttle(wait_time: float = 1.0):
    """
    Throttle a function to limit call frequency.
    
    Args:
        wait_time: Minimum time between calls in seconds
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        last_called = 0.0
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            nonlocal last_called
            
            elapsed = time.time() - last_called
            remaining = wait_time - elapsed
            
            if remaining > 0:
                time.sleep(remaining)
            
            result = func(*args, **kwargs)
            last_called = time.time()
            
            return result
        
        return wrapper
    
    return decorator

def memoize(ttl: float = 60.0):
    """
    Memoize a function with time-to-live.
    
    Args:
        ttl: Time-to-live in seconds
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: Dict[str, Tuple[T, float]] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create a key from the function arguments
            key = str((args, sorted(kwargs.items())))
            
            # Check if result is in cache and still valid
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
            
            # Call function and update cache
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    
    return decorator

def extract_error_message(exception: Exception) -> str:
    """
    Extract useful error message from exception.
    
    Args:
        exception: Exception object
        
    Returns:
        str: Formatted error message
    """
    error_str = str(exception)
    
    # If it's a JSON string, try to parse it
    if error_str.startswith('{') and error_str.endswith('}'):
        try:
            error_data = json.loads(error_str)
            if isinstance(error_data, dict):
                if 'message' in error_data:
                    return error_data['message']
                elif 'error' in error_data:
                    return error_data['error']
                elif 'msg' in error_data:
                    return error_data['msg']
        except json.JSONDecodeError:
            pass
    
    # Clean up common API error formats
    if ":" in error_str:
        parts = error_str.split(":", 1)
        if len(parts[1].strip()) > 0:
            return parts[1].strip()
    
    return error_str

def is_valid_json(json_str: str) -> bool:
    """
    Check if a string is valid JSON.
    
    Args:
        json_str: String to check
        
    Returns:
        bool: True if valid JSON
    """
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def parse_timeframe(timeframe_str: str) -> str:
    """
    Parse timeframe string to standard format.
    
    Args:
        timeframe_str: Timeframe string (e.g., '1min', '2hours', '1m')
        
    Returns:
        str: Normalized timeframe string (e.g., '1m', '2h')
    """
    if validate_timeframe(timeframe_str):
        return timeframe_str  # Already in correct format
    
    return string_to_timeframe(timeframe_str)


def string_to_timeframe(timeframe_str: str) -> str:
    """
    Convert human-readable time string to timeframe.
    
    Args:
        timeframe_str: Human-readable time string (e.g., '1 minute', '2 hours')
        
    Returns:
        str: Timeframe string (e.g., '1m', '2h')
    """
    # Normalize and clean up input
    timeframe_str = timeframe_str.lower().strip()
    
    # Handle plurals
    timeframe_str = timeframe_str.replace('minutes', 'minute')
    timeframe_str = timeframe_str.replace('hours', 'hour')
    timeframe_str = timeframe_str.replace('days', 'day')
    timeframe_str = timeframe_str.replace('weeks', 'week')
    timeframe_str = timeframe_str.replace('months', 'month')
    
    # Split into number and unit
    parts = timeframe_str.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid timeframe string: {timeframe_str}")
    
    try:
        number = int(parts[0])
    except ValueError:
        raise ValueError(f"Invalid timeframe number: {parts[0]}")
    
    unit = parts[1]
    
    # Map unit to timeframe unit
    unit_map = {
        'minute': 'm',
        'min': 'm',
        'm': 'm',
        'hour': 'h',
        'hr': 'h',
        'h': 'h',
        'day': 'd',
        'd': 'd',
        'week': 'w',
        'wk': 'w',
        'w': 'w',
        'month': 'M',
        'mo': 'M',
        'M': 'M'
    }
    
    if unit not in unit_map:
        raise ValueError(f"Invalid timeframe unit: {unit}")
    
    return f"{number}{unit_map[unit]}"

def setup_ccxt_exchange(exchange_id: str, api_key: str = None, secret: str = None, params: Dict = None) -> Any:
    """
    Set up a CCXT exchange instance with proper configuration
    
    Args:
        exchange_id: Exchange ID (e.g., 'binance', 'bybit')
        api_key: API key (optional)
        secret: API secret (optional)
        params: Additional exchange parameters (optional)
        
    Returns:
        ccxt.Exchange: Configured exchange instance
    """
    import ccxt
    
    # Default configuration
    config = {
        'enableRateLimit': True,  # Enable built-in rate limiter
        'timeout': 30000,  # Timeout in milliseconds
        'options': {
            'adjustForTimeDifference': True,
            'recvWindow': 60000,  # For Binance-like exchanges
        }
    }
    
    # Add credentials if provided
    if api_key and secret:
        config['apiKey'] = api_key
        config['secret'] = secret
    
    # Add additional parameters if provided
    if params:
        for key, value in params.items():
            if key == 'options' and isinstance(value, dict):
                config['options'].update(value)
            else:
                config[key] = value
    
    # Instantiate the exchange
    if exchange_id.lower() not in ccxt.exchanges:
        supported = ', '.join(ccxt.exchanges)
        raise ValueError(f"Exchange '{exchange_id}' not supported. Supported exchanges: {supported}")
    
    exchange_class = getattr(ccxt, exchange_id.lower())
    exchange = exchange_class(config)
    
    logger.info(f"CCXT {exchange_id} exchange instance created")
    
    return exchange