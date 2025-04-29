"""
Error Handling Module

This module provides advanced error handling and retry mechanisms for API calls,
network operations, and other error-prone activities in the trading bot.
"""

import time
import logging
import functools
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union

# Configure logger
logger = logging.getLogger("trading_bot.error_handling")

class ExchangeAPIError(Exception):
    """Exception raised for errors in exchange API calls"""
    
    def __init__(self, message, exchange=None, endpoint=None, status_code=None, 
                 response=None, retry_after=None):
        self.message = message
        self.exchange = exchange
        self.endpoint = endpoint
        self.status_code = status_code
        self.response = response
        self.retry_after = retry_after
        super().__init__(self.message)
    
    def __str__(self):
        return (f"{self.message} [Exchange: {self.exchange}, Endpoint: {self.endpoint}, "
                f"Status: {self.status_code}]")

class RateLimitError(ExchangeAPIError):
    """Exception raised for rate limit errors"""
    pass

class NetworkError(Exception):
    """Exception raised for network-related errors"""
    pass

class InsufficientFundsError(ExchangeAPIError):
    """Exception raised when account has insufficient funds for an operation"""
    pass

class DataError(Exception):
    """Exception raised for data-related errors"""
    pass

class ConfigError(Exception):
    """Exception raised for configuration errors"""
    pass

def retry(max_tries: int = 3, 
          delay: float = 1.0, 
          backoff_factor: float = 2.0,
          exceptions: tuple = (Exception,), 
          hook: Optional[Callable] = None):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_tries: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier applied to delay between retries
        exceptions: Tuple of exceptions to catch and retry
        hook: Function to call after catching an exception
        
    Returns:
        Callable: Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tries, delay_time = 0, delay
            
            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    tries += 1
                    if tries == max_tries:
                        logger.error(f"Function {func.__name__} failed after {max_tries} tries: {str(e)}")
                        raise
                    
                    # Check for rate limit and adjust delay if needed
                    if isinstance(e, RateLimitError) and e.retry_after:
                        delay_time = max(delay_time, e.retry_after + 0.5)
                    
                    logger.warning(
                        f"Retry {tries}/{max_tries} for {func.__name__} after {delay_time:.2f}s: {str(e)}"
                    )
                    
                    # Call hook if provided
                    if hook:
                        hook(tries=tries, delay=delay_time, exception=e, 
                             function=func.__name__, args=args, kwargs=kwargs)
                    
                    time.sleep(delay_time)
                    delay_time *= backoff_factor
                    
        return wrapper
    return decorator


def handle_api_errors(func):
    """
    Decorator to handle common API errors and translate them to specific exceptions
    
    Args:
        func: Function to decorate
        
    Returns:
        Callable: Decorated function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
            
        except Exception as e:
            error_message = str(e).lower()
            
            # Attempt to determine the exchange name from the function or args
            exchange = None
            if args and hasattr(args[0], 'id'):
                exchange = args[0].id
            
            # Handle rate limit errors
            if any(term in error_message for term in ['rate limit', 'ratelimit', 'too many requests']):
                retry_after = None
                # Try to extract retry_after from error message if available
                retry_after_indicators = ['retry after', 'retry-after', 'available in']
                for indicator in retry_after_indicators:
                    if indicator in error_message:
                        try:
                            parts = error_message.split(indicator)[1].strip().split()
                            retry_after = float(parts[0])
                            break
                        except (IndexError, ValueError):
                            pass
                            
                raise RateLimitError(
                    f"Rate limit exceeded: {str(e)}", 
                    exchange=exchange,
                    retry_after=retry_after
                ) from e
                
            # Handle insufficient funds errors
            elif any(term in error_message for term in ['insufficient', 'not enough', 'balance']):
                raise InsufficientFundsError(
                    f"Insufficient funds: {str(e)}", 
                    exchange=exchange
                ) from e
                
            # Handle network errors
            elif any(term in error_message for term in ['network', 'timeout', 'connection', 'socket']):
                raise NetworkError(f"Network error: {str(e)}") from e
                
            # Re-raise the original exception for other cases
            else:
                logger.error(f"API error in {func.__name__}: {str(e)}")
                logger.debug(f"Error details: {traceback.format_exc()}")
                raise
                
    return wrapper


def safe_execute(default_return=None, log_exception=True):
    """
    Decorator to safely execute a function and return a default value on exception
    
    Args:
        default_return: Value to return if an exception occurs
        log_exception: Whether to log the exception
        
    Returns:
        Callable: Decorated function with safe execution
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_exception:
                    logger.error(f"Exception in {func.__name__}: {str(e)}")
                    logger.debug(f"Exception details: {traceback.format_exc()}")
                return default_return
        return wrapper
    return decorator


def categorize_exchange_error(e: Exception, exchange: str = None) -> Dict:
    """
    Analyze an exchange error and return structured information about it
    
    Args:
        e: The exception to analyze
        exchange: Optional exchange name for context
        
    Returns:
        Dict: Structured error information
    """
    error_info = {
        "message": str(e),
        "type": type(e).__name__,
        "exchange": exchange,
        "category": "unknown",
        "is_recoverable": False,
        "retry_suggestion": None,
        "user_message": None
    }
    
    error_message = str(e).lower()
    
    # Categorize the error
    if any(term in error_message for term in ['rate limit', 'ratelimit', 'too many requests']):
        error_info["category"] = "rate_limit"
        error_info["is_recoverable"] = True
        error_info["retry_suggestion"] = "exponential_backoff"
        error_info["user_message"] = "Rate limit reached. The bot will automatically retry after a delay."
        
    elif any(term in error_message for term in ['insufficient', 'not enough', 'balance']):
        error_info["category"] = "insufficient_funds"
        error_info["is_recoverable"] = False
        error_info["user_message"] = "Insufficient funds to execute the order. Please deposit more funds."
        
    elif any(term in error_message for term in ['network', 'timeout', 'connection', 'socket']):
        error_info["category"] = "network"
        error_info["is_recoverable"] = True
        error_info["retry_suggestion"] = "incremental_backoff"
        error_info["user_message"] = "Network error occurred. The bot will retry automatically."
        
    elif any(term in error_message for term in ['key', 'secret', 'credential', 'permission', 'auth']):
        error_info["category"] = "authentication"
        error_info["is_recoverable"] = False
        error_info["user_message"] = "API key error. Please check your API key and permissions."
        
    elif any(term in error_message for term in ['invalid', 'parameter', 'argument']):
        error_info["category"] = "invalid_parameter"
        error_info["is_recoverable"] = False
        error_info["user_message"] = "Invalid order parameters. Please check your configuration."
        
    elif any(term in error_message for term in ['maintenance', 'unavailable']):
        error_info["category"] = "service_unavailable"
        error_info["is_recoverable"] = True
        error_info["retry_suggestion"] = "linear_backoff"
        error_info["user_message"] = "Exchange temporarily unavailable. The bot will retry later."
    
    return error_info