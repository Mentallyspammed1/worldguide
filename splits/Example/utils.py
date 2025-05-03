# File: utils.py
# -*- coding: utf-8 -*-

"""
Utility Functions and Placeholders for Bybit Helpers
"""

import logging
import sys
import time
import traceback
import random
from decimal import Decimal, InvalidOperation, getcontext
from typing import Optional, Dict, List, Tuple, Any, Literal, TypeVar, Callable, Union
from functools import wraps

try:
    import ccxt
except ImportError:
    print("Error: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)
try:
    from colorama import Fore, Style, Back
except ImportError:
    print("Warning: colorama library not found. Logs will not be colored. Install: pip install colorama")
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor()

# Set Decimal context precision
getcontext().prec = 28

# Logger setup for this module
logger = logging.getLogger(__name__)

# --- Placeholder for retry decorator ---
# This is critical: The actual implementation MUST be provided elsewhere (e.g., main.py).
T = TypeVar('T')
def retry_api_call(max_retries: int = 3, initial_delay: float = 1.0, **decorator_kwargs) -> Callable[[Callable[..., T]], Callable[..., T]]:
     """
     Placeholder for the actual retry decorator. The real decorator MUST handle
     CCXT exceptions and implement backoff logic.
     """
     def decorator(func: Callable[..., T]) -> Callable[..., T]:
         @wraps(func)
         def wrapper(*args: Any, **kwargs: Any) -> T:
             # Placeholder: Just call the function directly.
             # logger.debug(f"Placeholder retry decorator executing for {func.__name__}")
             try:
                 return func(*args, **kwargs)
             except Exception as e:
                 # logger.error(f"Placeholder retry decorator caught unhandled exception in {func.__name__}: {e}", exc_info=True)
                 raise # Re-raise; the real decorator would handle specific exceptions for retry
         return wrapper
     return decorator

# --- Placeholder Helper Functions ---
# These need to be replaced or implemented robustly in the main script or a base utility class.

def safe_decimal_conversion(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Safely converts a value to Decimal, returning default or None on failure."""
    if value is None: return default
    try: return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError): return default

def format_price(exchange: ccxt.Exchange, symbol: str, price: Any) -> str:
    """Formats price using market precision (placeholder)."""
    if price is None: return "N/A"
    try: return exchange.price_to_precision(symbol, price)
    except:
        try: return f"{Decimal(str(price)):.4f}" # Fallback: 4 decimal places
        except: return str(price) # Last resort

def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Any) -> str:
    """Formats amount using market precision (placeholder)."""
    if amount is None: return "N/A"
    try: return exchange.amount_to_precision(symbol, amount)
    except:
        try: return f"{Decimal(str(amount)):.8f}" # Fallback: 8 decimal places
        except: return str(amount) # Last resort

def format_order_id(order_id: Any) -> str:
    """Formats order ID for logging (placeholder)."""
    id_str = str(order_id) if order_id else "N/A"
    return id_str[-6:] if len(id_str) > 6 else id_str

def send_sms_alert(message: str, config: Optional['Config'] = None) -> bool:
    """Sends SMS alert via Termux (placeholder simulation)."""
    # Import config locally to avoid circular dependency if Config is passed
    from config import Config as ConfigClass
    cfg = config if config else ConfigClass() # Use passed config or default

    is_enabled = getattr(cfg, 'ENABLE_SMS_ALERTS', False)
    if is_enabled:
        recipient = getattr(cfg, 'SMS_RECIPIENT_NUMBER', None)
        timeout = getattr(cfg, 'SMS_TIMEOUT_SECONDS', 30)
        if recipient:
            logger.info(f"--- SIMULATING SMS Alert to {recipient} (Timeout: {timeout}s) ---")
            print(f"SMS: {message}")
            # Placeholder for actual Termux call
            # Example: os.system(f"termux-sms-send -n {recipient} '{message}'")
            return True # Simulation success
        else:
            logger.warning("SMS alerts enabled but no recipient number configured.")
            return False
    else:
        # logger.debug(f"SMS alert suppressed (disabled): {message}")
        return False

# --- Internal Utility Functions ---

def _get_v5_category(market: Dict[str, Any]) -> Optional[Literal['linear', 'inverse', 'spot', 'option']]:
    """Internal helper to determine the Bybit V5 category from a market object."""
    if not market: return None
    # Prefer explicit flags if available
    if market.get('linear'): return 'linear'
    if market.get('inverse'): return 'inverse'
    if market.get('spot'): return 'spot'
    if market.get('option'): return 'option'
    # Fallback based on market type
    market_type = market.get('type')
    if market_type == 'swap': return 'linear' # Assume linear swap if flags missing
    if market_type == 'future': return 'linear' # Assume linear future
    if market_type == 'spot': return 'spot'
    if market_type == 'option': return 'option'
    logger.warning(f"_get_v5_category: Could not determine category for market: {market.get('symbol')}")
    return None

# Snippet 20 / Function 20: Validate Symbol/Market
# No API call if markets already loaded, so no decorator needed here.
def validate_market(
    exchange: ccxt.bybit, symbol: str, config: 'Config', expected_type: Optional[Literal['swap', 'future', 'spot', 'option']] = None,
    expected_logic: Optional[Literal['linear', 'inverse']] = None, check_active: bool = True, require_contract: bool = True
) -> Optional[Dict]:
    """
    Validates if a symbol exists on the exchange, is active, and optionally matches
    expected type (swap, spot, etc.) and logic (linear, inverse). Loads markets if needed.
    """
    # Import config locally to avoid circular dependency if Config is passed
    from config import Config as ConfigClass
    cfg = config if config else ConfigClass() # Use passed config or default

    func_name = "validate_market"; eff_expected_type = expected_type if expected_type is not None else cfg.EXPECTED_MARKET_TYPE; eff_expected_logic = expected_logic if expected_logic is not None else cfg.EXPECTED_MARKET_LOGIC
    logger.debug(f"[{func_name}] Validating '{symbol}'. Checks: Type='{eff_expected_type or 'Any'}', Logic='{eff_expected_logic or 'Any'}', Active={check_active}, Contract={require_contract}")
    try:
        if not exchange.markets or not exchange.markets_by_id: # Check both
             logger.info(f"[{func_name}] Loading markets for validation...")
             exchange.load_markets(reload=True)
        if not exchange.markets:
             logger.error(f"{Fore.RED}[{func_name}] Failed to load markets.{Style.RESET_ALL}")
             return None

        market = exchange.market(symbol) # Raises BadSymbol if not found

        is_active = market.get('active', False);
        if check_active and not is_active:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Validation Warning: '{symbol}' market is INACTIVE.{Style.RESET_ALL}")
            # Decide if inactive market should cause failure
            # return None # Uncomment to fail validation for inactive markets

        actual_type = market.get('type');
        if eff_expected_type and actual_type != eff_expected_type:
             logger.error(f"{Fore.RED}[{func_name}] Validation Failed: '{symbol}' type mismatch. Expected '{eff_expected_type}', Got '{actual_type}'.{Style.RESET_ALL}")
             return None

        is_contract = market.get('contract', False);
        if require_contract and not is_contract:
            logger.error(f"{Fore.RED}[{func_name}] Validation Failed: '{symbol}' is not a contract market, but contract was required.{Style.RESET_ALL}")
            return None

        actual_logic_str: Optional[str] = None
        if is_contract:
            actual_logic_str = _get_v5_category(market); # Use internal helper
            if eff_expected_logic and actual_logic_str != eff_expected_logic:
                 logger.error(f"{Fore.RED}[{func_name}] Validation Failed: '{symbol}' contract logic mismatch. Expected '{eff_expected_logic}', Got '{actual_logic_str}'.{Style.RESET_ALL}")
                 return None

        logger.info(f"{Fore.GREEN}[{func_name}] Market OK: '{symbol}' (Type:{actual_type}, Logic:{actual_logic_str or 'N/A'}, Active:{is_active}).{Style.RESET_ALL}")
        return market

    except ccxt.BadSymbol as e:
        logger.error(f"{Fore.RED}[{func_name}] Validation Failed: Symbol '{symbol}' not found on exchange. Error: {e}{Style.RESET_ALL}")
        return None
    except ccxt.NetworkError as e:
        logger.error(f"{Fore.RED}[{func_name}] Network error during market validation/loading for '{symbol}': {e}{Style.RESET_ALL}")
        return None # Or raise depending on desired behavior
    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error validating '{symbol}': {e}{Style.RESET_ALL}", exc_info=True)
        return None

# --- END OF FILE utils.py ---

# ---------------------------------------------------------------------------

