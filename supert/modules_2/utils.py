# File: utils.py
import subprocess
import traceback
from decimal import Decimal, InvalidOperation
from typing import Any

# Third-party Libraries
try:
    import ccxt
    from colorama import Fore, Style
except ImportError:
    # Define dummy ccxt and colorama elements if not available,
    # though parts of the script will fail if ccxt is missing.
    class DummyCCXT:
        Exchange = type('Exchange', (), {})
    ccxt = DummyCCXT() # type: ignore[assignment]
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""
    Fore, Style = DummyColor(), DummyColor()


# Custom module imports
from logger_setup import logger
from config import CONFIG


def safe_decimal_conversion(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely converts a value to Decimal, returning default if conversion fails."""
    try:
        return Decimal(str(value)) if value is not None else default
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(f"Could not convert '{value}' to Decimal, using default {default}")
        return default


def format_order_id(order_id: str | int | None) -> str:
    """Returns the last 6 characters of an order ID or 'N/A'."""
    return str(order_id)[-6:] if order_id else "N/A"


def format_price(exchange: ccxt.Exchange, symbol: str, price: float | Decimal) -> str:
    """Formats price according to market precision rules."""
    try:
        return exchange.price_to_precision(symbol, float(price))
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}{Style.RESET_ALL}")
        return str(Decimal(str(price)).normalize())


def format_amount(exchange: ccxt.Exchange, symbol: str, amount: float | Decimal) -> str:
    """Formats amount according to market precision rules."""
    try:
        return exchange.amount_to_precision(symbol, float(amount))
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}{Style.RESET_ALL}")
        return str(Decimal(str(amount)).normalize())


def send_sms_alert(message: str) -> bool:
    """Sends an SMS alert using Termux API."""
    if not CONFIG.enable_sms_alerts:
        logger.debug("SMS alerts disabled.")
        return False
    if not CONFIG.sms_recipient_number:
        logger.warning("SMS alerts enabled, but SMS_RECIPIENT_NUMBER not set.")
        return False
    try:
        command: list[str] = ["termux-sms-send", "-n", CONFIG.sms_recipient_number, message]
        logger.info(
            f"{Fore.MAGENTA}Attempting SMS to {CONFIG.sms_recipient_number} (Timeout: {CONFIG.sms_timeout_seconds}s)...{Style.RESET_ALL}"
        )
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=CONFIG.sms_timeout_seconds
        )
        if result.returncode == 0:
            logger.success(f"{Fore.MAGENTA}SMS command executed successfully.{Style.RESET_ALL}") # type: ignore[attr-defined]
            return True
        else:
            logger.error(
                f"{Fore.RED}SMS command failed. RC: {result.returncode}, Stderr: {result.stderr.strip()}{Style.RESET_ALL}"
            )
            return False
    except FileNotFoundError:
        logger.error(f"{Fore.RED}SMS failed: 'termux-sms-send' not found. Install Termux:API.{Style.RESET_ALL}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"{Fore.RED}SMS failed: command timed out after {CONFIG.sms_timeout_seconds}s.{Style.RESET_ALL}")
        return False
    except Exception as e:
        logger.error(f"{Fore.RED}SMS failed: Unexpected error: {e}{Style.RESET_ALL}")
        logger.debug(traceback.format_exc())
        return False

# End of utils.py
```

```python
