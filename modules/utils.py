# File: utils.py
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Tuple

# Attempt to import zoneinfo, handle potential ImportError
try:
    from zoneinfo import ZoneInfo as PythonZoneInfo
except ImportError:
    print("Error: 'zoneinfo' requires Python 3.9+ and potentially the 'tzdata' package.")
    print("Please install it if needed: pip install tzdata")
    print("Warning: Using basic UTC fallback for timezone handling. Displayed times will be UTC.")
    class ZoneInfoFallback: # Basic fallback implementation
        """Basic fallback for ZoneInfo if not available (Python < 3.9 or tzdata missing). Only supports UTC."""
        def __init__(self, key: str):
            self._offset = timedelta(0)
            if key.lower() != 'utc':
                print(f"Warning: Timezone '{key}' not supported by fallback. Using UTC.")
            self._name = "UTC" # Always UTC in fallback

        def __call__(self): # Make it callable like the real ZoneInfo
             return self

        def fromutc(self, dt: datetime) -> datetime:
             # In UTC fallback, fromutc doesn't change the time, just adds tzinfo
             return dt.replace(tzinfo=timezone(self._offset, self._name))

        def utcoffset(self, dt: Optional[datetime]) -> timedelta:
             return self._offset

        def dst(self, dt: Optional[datetime]) -> timedelta:
             # Fallback doesn't support DST
             return timedelta(0)

        def tzname(self, dt: Optional[datetime]) -> str:
             return self._name

        def __repr__(self) -> str:
            return f"ZoneInfoFallback(key='{self._name}')" # Show the effective key

        def __str__(self) -> str:
            return self._name
    # Assign the fallback class to the ZoneInfo name for consistent usage
    ZoneInfo = ZoneInfoFallback
else:
    # If import succeeded, use the real ZoneInfo
    ZoneInfo = PythonZoneInfo


# --- Market Data Utility Functions ---

def get_price_precision(market_info: Dict[str, Any], logger: logging.Logger) -> int:
    """
    Determines the number of decimal places required for price values
    based on the market information provided by the exchange. (Static Method)

    Uses 'precision' and 'limits' fields from the market info.
    Falls back to a default value.

    Args:
        market_info: Dictionary containing market details (precision, limits, etc.).
        logger: Logger instance for logging messages.

    Returns:
        The number of decimal places (integer).
    """
    symbol = market_info.get('symbol', 'UNKNOWN')
    try:
        # 1. Check 'precision.price' (most common field)
        precision_info = market_info.get('precision', {})
        price_precision_val = precision_info.get('price')

        if price_precision_val is not None:
             # If it's an integer, it usually represents decimal places directly
             if isinstance(price_precision_val, int):
                  if price_precision_val >= 0:
                       logger.debug(f"Using price precision (decimal places) from market_info.precision.price: {price_precision_val} for {symbol}")
                       return price_precision_val
             # If it's float/str, it often represents the tick size
             elif isinstance(price_precision_val, (float, str)):
                  try:
                       tick_size = Decimal(str(price_precision_val))
                       # Ensure tick size is positive
                       if tick_size > 0:
                            # Calculate decimal places from tick size
                            # normalize() removes trailing zeros, as_tuple().exponent gives the exponent
                            precision = abs(tick_size.normalize().as_tuple().exponent)
                            logger.debug(f"Calculated price precision from market_info.precision.price (tick size {tick_size}): {precision} for {symbol}")
                            return precision
                  except (InvalidOperation, ValueError, TypeError) as e:
                       logger.warning(f"Could not parse precision.price '{price_precision_val}' as tick size for {symbol}: {e}")

        # 2. Fallback: Check 'limits.price.min' (sometimes represents tick size)
        limits_info = market_info.get('limits', {})
        price_limits = limits_info.get('price', {})
        min_price_val = price_limits.get('min')

        if min_price_val is not None:
             try:
                  min_price_tick = Decimal(str(min_price_val))
                  if min_price_tick > 0:
                       # Heuristic: Check if min_price looks like a tick size (small value)
                       # rather than just a minimum orderable price (e.g., 0.1).
                       # Tick sizes are usually << 1. Adjust threshold if needed.
                       if min_price_tick < Decimal('0.1'):
                            precision = abs(min_price_tick.normalize().as_tuple().exponent)
                            logger.debug(f"Inferred price precision from limits.price.min ({min_price_tick}): {precision} for {symbol}")
                            return precision
                       else:
                            logger.debug(f"limits.price.min ({min_price_tick}) for {symbol} seems too large for tick size, likely minimum order price. Ignoring for precision.")
             except (InvalidOperation, ValueError, TypeError) as e:
                  logger.warning(f"Could not parse limits.price.min '{min_price_val}' for precision inference for {symbol}: {e}")

    except Exception as e:
        logger.warning(f"Error determining price precision for {symbol} from market info: {e}. Falling back.")

    # --- Final Fallback ---
    # Use a reasonable default if no other method worked
    default_precision = 4 # Common default, adjust if needed for your typical markets
    logger.warning(f"Could not determine price precision for {symbol}. Using default: {default_precision}.")
    return default_precision


def get_min_tick_size(market_info: Dict[str, Any], logger: logging.Logger) -> Decimal:
    """
    Gets the minimum price increment (tick size) from market info using Decimal. (Static Method)

    Args:
        market_info: Dictionary containing market details (precision, limits, etc.).
        logger: Logger instance for logging messages.

    Returns:
        The minimum tick size as a Decimal object. Falls back based on precision.
    """
    symbol = market_info.get('symbol', 'UNKNOWN')
    try:
        # 1. Try precision.price (often the tick size as float/str)
        precision_info = market_info.get('precision', {})
        price_precision_val = precision_info.get('price')
        if price_precision_val is not None:
             if isinstance(price_precision_val, (float, str)):
                  try:
                       tick_size = Decimal(str(price_precision_val))
                       if tick_size > 0:
                            logger.debug(f"Using tick size from precision.price: {tick_size} for {symbol}")
                            return tick_size
                  except (InvalidOperation, ValueError, TypeError) as e:
                        logger.warning(f"Could not parse precision.price '{price_precision_val}' as tick size for {symbol}: {e}")
             # If it's an integer (decimal places), calculate tick size
             elif isinstance(price_precision_val, int) and price_precision_val >= 0:
                  tick_size = Decimal('1e-' + str(price_precision_val))
                  logger.debug(f"Calculated tick size from precision.price (decimal places {price_precision_val}): {tick_size} for {symbol}")
                  return tick_size

        # 2. Fallback: Try limits.price.min (sometimes represents tick size)
        limits_info = market_info.get('limits', {})
        price_limits = limits_info.get('price', {})
        min_price_val = price_limits.get('min')
        if min_price_val is not None:
            try:
                min_tick_from_limit = Decimal(str(min_price_val))
                if min_tick_from_limit > 0:
                    # Heuristic check: if it's very small, assume it's the tick size
                    if min_tick_from_limit < Decimal('0.1'):
                         logger.debug(f"Using tick size from limits.price.min: {min_tick_from_limit} for {symbol}")
                         return min_tick_from_limit
                    else:
                         logger.debug(f"limits.price.min ({min_tick_from_limit}) for {symbol} seems too large for tick size, potentially min order price.")
            except (InvalidOperation, ValueError, TypeError) as e:
                logger.warning(f"Could not parse limits.price.min '{min_price_val}' for tick size inference for {symbol}: {e}")

    except Exception as e:
         logger.warning(f"Could not determine min tick size for {symbol} from market info: {e}. Using precision fallback.")

    # --- Final Fallback: Calculate from get_price_precision (decimal places) ---
    price_precision_places = get_price_precision(market_info, logger) # Call the other util function
    fallback_tick = Decimal('1e-' + str(price_precision_places))
    logger.debug(f"Using fallback tick size based on derived precision places ({price_precision_places}): {fallback_tick} for {symbol}")
    return fallback_tick
```

```python
