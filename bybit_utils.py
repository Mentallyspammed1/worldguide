import ccxt
from decimal import Decimal, InvalidOperation
from typing import Optional, Any, Callable, TypeVar
from colorama import Fore, Style
import logging
import time
import functools

# Assume logger is configured in the importing scope (e.g., ps.py)
logger = logging.getLogger(__name__)

def safe_decimal_conversion(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """
    Convert various inputs to Decimal, returning default or None on failure.
    
    Args:
        value: Input value (e.g., str, float, int, Decimal).
        default: Value to return if conversion fails (default: None).
    
    Returns:
        Decimal value, or default/None on failure.
    """
    try:
        if value is None:
            return default
        return Decimal(str(value))
    except (ValueError, TypeError, InvalidOperation):
        logger.debug(f"{Fore.YELLOW}Failed to convert {value} to Decimal, returning {default}{Style.RESET_ALL}")
        return default

def format_price(exchange: ccxt.Exchange, symbol: str, price: Any) -> str:
    """
    Format a price value according to the market's precision rules.
    
    Args:
        exchange: CCXT exchange object (e.g., ccxt.bybit).
        symbol: Trading pair (e.g., 'BTC/USDT:USDT').
        price: Price value (e.g., Decimal, float, str).
    
    Returns:
        Formatted price string adhering to market precision.
    """
    try:
        market = exchange.market(symbol)
        precision = market['precision']['price']
        price_decimal = safe_decimal_conversion(price, Decimal('0'))
        if price_decimal is None:
            logger.error(f"{Fore.RED}Invalid price {price} for {symbol}{Style.RESET_ALL}")
            raise ValueError(f"Invalid price {price}")
        return f"{price_decimal:.{precision}f}"
    except Exception as e:
        logger.critical(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}{Style.RESET_ALL}")
        raise

def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Any) -> str:
    """
    Format an amount value according to the market's precision rules.
    
    Args:
        exchange: CCXT exchange object (e.g., ccxt.bybit).
        symbol: Trading pair (e.g., 'BTC/USDT:USDT').
        amount: Amount value (e.g., Decimal, float, str).
    
    Returns:
        Formatted amount string adhering to market precision.
    """
    try:
        market = exchange.market(symbol)
        precision = market['precision']['amount']
        amount_decimal = safe_decimal_conversion(amount, Decimal('0'))
        if amount_decimal is None:
            logger.error(f"{Fore.RED}Invalid amount {amount} for {symbol}{Style.RESET_ALL}")
            raise ValueError(f"Invalid amount {amount}")
        return f"{amount_decimal:.{precision}f}"
    except Exception as e:
        logger.critical(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}{Style.RESET_ALL}")
        raise

def format_order_id(order_id: Any) -> str:
    """
    Format an order ID for concise logging (shows last 6 digits).
    
    Args:
        order_id: Order ID (e.g., str, int).
    
    Returns:
        Formatted order ID string, or 'UNKNOWN' if invalid.
    """
    try:
        if order_id is None or not str(order_id).strip():
            return 'UNKNOWN'
        return str(order_id)[-6:]
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting order ID {order_id}: {e}{Style.RESET_ALL}")
        return 'UNKNOWN'

def send_sms_alert(message: str, config: Optional[Any] = None) -> bool:
    """
    Send an SMS alert for critical errors or actions (placeholder implementation).
    
    Args:
        message: Message to send.
        config: Optional Config object with SMS settings (e.g., Twilio credentials).
    
    Returns:
        True if alert sent successfully, False otherwise.
    """
    try:
        # Placeholder: Log message instead of sending SMS (replace with Twilio or similar)
        logger.warning(f"{Fore.YELLOW}SMS Alert: {message}{Style.RESET_ALL}")
        # Example Twilio implementation (uncomment and configure if needed):
        """
        from twilio.rest import Client
        client = Client(config.twilio_account_sid, config.twilio_auth_token)
        client.messages.create(
            body=message,
            from_=config.twilio_from_number,
            to=config.twilio_to_number
        )
        """
        return True
    except Exception as e:
        logger.critical(f"{Fore.RED}Failed to send SMS alert '{message}': {e}{Style.RESET_ALL}")
        return False

T = TypeVar('T')
def retry_api_call(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to retry API calls on common CCXT exceptions with exponential backoff.
    
    Handles NetworkError, RateLimitExceeded, and ExchangeNotAvailable.
    Uses Config.RETRY_COUNT and Config.RETRY_DELAY_SECONDS from the importing scope.
    
    Args:
        func: Function to decorate (e.g., API-calling function like fetch_ohlcv).
    
    Returns:
        Wrapped function with retry logic.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        # Assume config is passed in kwargs or args (e.g., as in snippets)
        config = kwargs.get('config') or next((arg for arg in args if hasattr(arg, 'RETRY_COUNT')), None)
        if not config:
            logger.error(f"{Fore.RED}No config provided for retry_api_call in {func.__name__}{Style.RESET_ALL}")
            raise ValueError("Config object required for retry_api_call")
        
        max_retries = config.RETRY_COUNT
        base_delay = config.RETRY_DELAY_SECONDS
        
        attempt = 0
        while attempt < max_retries:
            try:
                return func(*args, **kwargs)
            except ccxt.RateLimitExceeded as e:
                attempt += 1
                delay = base_delay * (2 ** attempt)
                logger.warning(f"{Fore.YELLOW}Rate limit exceeded in {func.__name__}. Retry {attempt}/{max_retries} after {delay:.2f}s: {e}{Style.RESET_ALL}")
                if attempt == max_retries:
                    logger.error(f"{Fore.RED}Max retries reached for {func.__name__}.{Style.RESET_ALL}")
                    send_sms_alert(f"Rate limit exceeded for {func.__name__} after {max_retries} retries.", config)
                    raise
                time.sleep(delay)
            except ccxt.NetworkError as e:
                attempt += 1
                delay = base_delay
                logger.error(f"{Fore.RED}Network error in {func.__name__}. Retry {attempt}/{max_retries} after {delay:.2f}s: {e}{Style.RESET_ALL}")
                if attempt == max_retries:
                    logger.error(f"{Fore.RED}Max retries reached for {func.__name__}.{Style.RESET_ALL}")
                    send_sms_alert(f"Network error in {func.__name__} after {max_retries} retries.", config)
                    raise
                time.sleep(delay)
            except ccxt.ExchangeNotAvailable as e:
                attempt += 1
                delay = base_delay
                logger.error(f"{Fore.RED}Exchange not available in {func.__name__}. Retry {attempt}/{max_retries} after {delay:.2f}s: {e}{Style.RESET_ALL}")
                if attempt == max_retries:
                    logger.error(f"{Fore.RED}Max retries reached for {func.__name__}.{Style.RESET_ALL}")
                    send_sms_alert(f"Exchange not available in {func.__name__} after {max_retries} retries.", config)
                    raise
                time.sleep(delay)
            except Exception as e:
                logger.critical(f"{Fore.RED}Unexpected error in {func.__name__}: {e}{Style.RESET_ALL}")
                raise
        raise Exception(f"Failed to execute {func.__name__} after {max_retries} retries")
    return wrapper