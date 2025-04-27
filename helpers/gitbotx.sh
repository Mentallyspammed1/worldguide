Okay, the tracebacks reveal several key issues stemming from a mix of `AttributeError`s (missing functions/attributes), `TypeError`s (incorrect function arguments, possibly related to async wrappers), and `ValueError`s (invalid data). The `aiohttp` warnings strongly suggest an asynchronous context (`asyncio`) is being used somewhere (perhaps in the `main.py` mentioned), but the provided helper files and the core strategy logic requested were synchronous.

This refactoring will:

1.  **Enforce Synchronous Structure:** Ensure all provided code is consistently synchronous for clarity and to match the initial request. I'll add notes about `asyncio` adaptation.
2.  **Fix AttributeErrors:** Ensure functions like `get_current_position_bybit_v5` are correctly called (using the appropriate alias or instance attribute). Remove checks for non-existent attributes like `exchange.closed`.
3.  **Fix TypeErrors:** Correct the `cancel_all_orders` call signature issues. Ensure helper functions are called with the right arguments.
4.  **Fix ValueErrors:** Address the ticker timestamp validation logic in `fetch_ticker_validated`.
5.  **Implement Strategy Class:** Refactor `ehlers_volumatic_straregy.py` into a class for better state management.
6.  **Add Missing Helpers:** Include `cancel_order` and `fetch_order` in `bybit_helpers.py`.
7.  **Implement Take Profit:** Add basic ATR-based TP logic.
8.  **Enhance Robustness:** Improve error handling and checks for `None` values.

---

**1. `neon_logger.py` (Enhanced v1.2 - No changes from previous response)**

```python
# --- START OF FILE neon_logger.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neon Logger Setup (v1.2) - Enhanced Robustness & Features

Provides a function `setup_logger` to configure a Python logger instance with:
- Colorized console output using a "neon" theme via colorama (TTY only).
- Uses a custom Formatter for cleaner color handling.
- Clean, non-colorized file output.
- Optional log file rotation (size-based).
- Extensive log formatting (timestamp, level, function, line, thread).
- Custom SUCCESS log level.
- Configurable log levels via args or environment variables.
- Option to control verbosity of third-party libraries.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any

# --- Attempt to import colorama ---
try:
    from colorama import Fore, Style, Back, init as colorama_init
    # Initialize colorama (autoreset=True ensures colors reset after each print)
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    # Define dummy color objects if colorama is not installed
    class DummyColor:
        def __getattr__(self, name: str) -> str: return "" # Return empty string
    Fore = DummyColor(); Back = DummyColor(); Style = DummyColor()
    COLORAMA_AVAILABLE = False
    print("Warning: 'colorama' library not found. Neon console logging disabled.", file=sys.stderr)
    print("         Install using: pip install colorama", file=sys.stderr)

# --- Custom Log Level ---
SUCCESS_LEVEL = 25 # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Adds a custom 'success' log method to the Logger instance."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

# Add the method to the Logger class dynamically
if not hasattr(logging.Logger, 'success'):
    logging.Logger.success = log_success # type: ignore[attr-defined]


# --- Neon Color Theme Mapping ---
LOG_LEVEL_COLORS: Dict[int, str] = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.BLUE,
    SUCCESS_LEVEL: Fore.MAGENTA,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}",
}

# --- Custom Formatter for Colored Console Output ---
class ColoredConsoleFormatter(logging.Formatter):
    """
    A custom logging formatter that adds colors to console output based on log level,
    only if colorama is available and output is a TTY.
    """
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, style: str = '%', validate: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        self.use_colors = COLORAMA_AVAILABLE and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Formats the record and applies colors to the level name."""
        # Store original levelname before potential modification
        original_levelname = record.levelname
        color = LOG_LEVEL_COLORS.get(record.levelno, Fore.WHITE) # Default to white

        if self.use_colors:
            # Temporarily add color codes to the levelname for formatting
            record.levelname = f"{color}{original_levelname}{Style.RESET_ALL}"

        # Use the parent class's formatting method
        formatted_message = super().format(record)

        # Restore original levelname to prevent colored output in file logs etc.
        record.levelname = original_levelname

        return formatted_message


# --- Log Format Strings ---
# Include thread name for better context in concurrent applications
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s [%(threadName)s %(funcName)s:%(lineno)d] - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create formatters
console_formatter = ColoredConsoleFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


# --- Main Setup Function ---
def setup_logger(
    logger_name: str = "AppLogger",
    log_file: Optional[str] = "app.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_rotation_bytes: int = 5 * 1024 * 1024, # 5 MB default max size
    log_backup_count: int = 5, # Keep 5 backup files
    propagate: bool = False,
    third_party_log_level: int = logging.WARNING # Default level for noisy libraries
) -> logging.Logger:
    """
    Sets up and configures a logger instance with neon console, clean file output,
    optional rotation, and control over third-party library logging.

    Looks for environment variables to override default levels/file path:
        - LOG_CONSOLE_LEVEL: (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - LOG_FILE_LEVEL: (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - LOG_FILE_PATH: (e.g., /path/to/your/bot.log)

    Args:
        logger_name: Name for the logger instance.
        log_file: Path to the log file. None disables file logging. Rotation enabled by default.
        console_level: Logging level for console output. Overridden by LOG_CONSOLE_LEVEL env var.
        file_level: Logging level for file output. Overridden by LOG_FILE_LEVEL env var.
        log_rotation_bytes: Max size in bytes before rotating log file. 0 disables rotation.
        log_backup_count: Number of backup log files to keep. Ignored if rotation is disabled.
        propagate: Whether to propagate messages to the root logger (default False).
        third_party_log_level: Level for common noisy libraries (ccxt, urllib3, etc.).

    Returns:
        The configured logging.Logger instance.
    """
    func_name = "setup_logger" # For internal logging if needed

    # --- Environment Variable Overrides ---
    env_console_level_str = os.getenv("LOG_CONSOLE_LEVEL")
    env_file_level_str = os.getenv("LOG_FILE_LEVEL")
    env_log_file = os.getenv("LOG_FILE_PATH")

    if env_console_level_str:
        env_console_level = logging.getLevelName(env_console_level_str.upper())
        if isinstance(env_console_level, int):
            print(f"Neon Logger: Overriding console level from env LOG_CONSOLE_LEVEL='{env_console_level_str}' -> {logging.getLevelName(env_console_level)}")
            console_level = env_console_level
        else:
            print(f"Warning: Invalid LOG_CONSOLE_LEVEL '{env_console_level_str}'. Using default: {logging.getLevelName(console_level)}", file=sys.stderr)

    if env_file_level_str:
        env_file_level = logging.getLevelName(env_file_level_str.upper())
        if isinstance(env_file_level, int):
            print(f"Neon Logger: Overriding file level from env LOG_FILE_LEVEL='{env_file_level_str}' -> {logging.getLevelName(env_file_level)}")
            file_level = env_file_level
        else:
            print(f"Warning: Invalid LOG_FILE_LEVEL '{env_file_level_str}'. Using default: {logging.getLevelName(file_level)}", file=sys.stderr)

    if env_log_file:
        print(f"Neon Logger: Overriding log file path from env LOG_FILE_PATH='{env_log_file}'")
        log_file = env_log_file

    # --- Get Logger and Set Base Level/Propagation ---
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) # Set logger to lowest level to capture all messages for handlers
    logger.propagate = propagate

    # --- Clear Existing Handlers (if re-configuring) ---
    if logger.hasHandlers():
        print(f"Logger '{logger_name}' already configured. Clearing existing handlers.", file=sys.stderr)
        for handler in logger.handlers[:]: # Iterate a copy
            try:
                handler.close() # Close file handles etc.
                logger.removeHandler(handler)
            except Exception as e:
                 print(f"Warning: Error removing/closing existing handler: {e}", file=sys.stderr)

    # --- Console Handler ---
    if console_level is not None and console_level >= 0:
        try:
            console_h = logging.StreamHandler(sys.stdout)
            console_h.setLevel(console_level)
            console_h.setFormatter(console_formatter) # Use the colored formatter
            logger.addHandler(console_h)
            print(f"Neon Logger: Console logging active at level [{logging.getLevelName(console_level)}].")
        except Exception as e:
             print(f"{Fore.RED}Error setting up console handler: {e}{Style.RESET_ALL}", file=sys.stderr)
    else:
        print("Neon Logger: Console logging disabled.")

    # --- File Handler (with optional rotation) ---
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                 os.makedirs(log_dir, exist_ok=True) # Ensure directory exists

            if log_rotation_bytes > 0 and log_backup_count >= 0:
                # Use Rotating File Handler
                file_h = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=log_rotation_bytes,
                    backupCount=log_backup_count,
                    encoding='utf-8'
                )
                print(f"Neon Logger: Rotating file logging active at level [{logging.getLevelName(file_level)}] to '{log_file}' (Max: {log_rotation_bytes / 1024 / 1024:.1f} MB, Backups: {log_backup_count}).")
            else:
                # Use basic File Handler (no rotation)
                file_h = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                print(f"Neon Logger: Basic file logging active at level [{logging.getLevelName(file_level)}] to '{log_file}' (Rotation disabled).")

            file_h.setLevel(file_level)
            file_h.setFormatter(file_formatter) # Use the plain (non-colored) formatter
            logger.addHandler(file_h)

        except IOError as e:
            print(f"{Fore.RED}{Style.BRIGHT}Fatal Error configuring log file '{log_file}': {e}{Style.RESET_ALL}", file=sys.stderr)
        except Exception as e:
             print(f"{Fore.RED}{Style.BRIGHT}Unexpected error setting up file logging: {e}{Style.RESET_ALL}", file=sys.stderr)
    else:
        print("Neon Logger: File logging disabled.")

    # --- Configure Third-Party Log Levels ---
    if third_party_log_level is not None and third_party_log_level >= 0:
        noisy_libraries = ["ccxt", "urllib3", "requests", "asyncio"] # Add others if needed
        print(f"Neon Logger: Setting third-party library log level to [{logging.getLevelName(third_party_log_level)}].")
        for lib_name in noisy_libraries:
            try:
                logging.getLogger(lib_name).setLevel(third_party_log_level)
            except Exception:
                 pass # Ignore errors setting levels for libs that don't use standard logging
    else:
         print("Neon Logger: Third-party library log level control disabled.")

    # --- Log Test Messages ---
    logger.debug("--- Logger Setup Complete (DEBUG Test) ---")
    logger.info("--- Logger Setup Complete (INFO Test) ---")
    logger.success("--- Logger Setup Complete (SUCCESS Test) ---")
    logger.warning("--- Logger Setup Complete (WARNING Test) ---")
    logger.error("--- Logger Setup Complete (ERROR Test) ---")
    logger.critical("--- Logger Setup Complete (CRITICAL Test) ---")

    return logger

# --- Example Usage ---
if __name__ == "__main__":
    print("-" * 60)
    print("--- Example Neon Logger v1.2 Usage ---")
    print("-" * 60)
    # ... (Example code remains the same) ...
    print(f"\nCheck console output and log files created.")

# --- END OF FILE neon_logger.py ---
```

**2. `bybit_utils.py` (Enhanced v1.1 - No changes from previous response)**

```python
# --- START OF FILE bybit_utils.py ---

import ccxt
from decimal import Decimal, InvalidOperation
import time
import functools
import logging
import subprocess # For Termux API call
from typing import Optional, Any, Callable, TypeVar, Dict, List, Tuple

try:
    from colorama import Fore, Style, Back, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor()

# Assume logger is configured in the importing scope (e.g., strategy script)
# If not, create a basic placeholder
if 'logger' not in globals():
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.info("Placeholder logger initialized for bybit_utils.py.")
else:
    # If logger exists, ensure it's the correct one for this module context
    logger = logging.getLogger(__name__)


# Placeholder TypeVar for Config object (structure defined in importing script)
ConfigPlaceholder = TypeVar('ConfigPlaceholder')

# --- Utility Functions ---

def safe_decimal_conversion(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """Convert various inputs to Decimal, returning default or None on failure."""
    if value is None:
        return default
    try:
        if isinstance(value, str) and 'e' in value.lower():
            return Decimal(value)
        return Decimal(str(value))
    except (ValueError, TypeError, InvalidOperation):
        return default

def format_price(exchange: ccxt.Exchange, symbol: str, price: Any) -> str:
    """Format a price value according to the market's precision rules."""
    if price is None: return "N/A"
    try:
        return exchange.price_to_precision(symbol, price)
    except (KeyError, AttributeError, TypeError, ValueError):
        logger.warning(f"{Fore.YELLOW}Market data/precision method issue for symbol '{symbol}' in format_price. Using fallback.{Style.RESET_ALL}", exc_info=True)
        price_dec = safe_decimal_conversion(price)
        return f"{price_dec:.8f}" if price_dec is not None else "Invalid"
    except Exception as e:
        logger.critical(f"{Fore.RED}Error formatting price '{price}' for {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
        return "Error"

def format_amount(exchange: ccxt.Exchange, symbol: str, amount: Any) -> str:
    """Format an amount value according to the market's precision rules."""
    if amount is None: return "N/A"
    try:
        return exchange.amount_to_precision(symbol, amount)
    except (KeyError, AttributeError, TypeError, ValueError):
        logger.warning(f"{Fore.YELLOW}Market data/precision method issue for symbol '{symbol}' in format_amount. Using fallback.{Style.RESET_ALL}", exc_info=True)
        amount_dec = safe_decimal_conversion(amount)
        return f"{amount_dec:.8f}" if amount_dec is not None else "Invalid"
    except Exception as e:
        logger.critical(f"{Fore.RED}Error formatting amount '{amount}' for {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
        return "Error"

def format_order_id(order_id: Any) -> str:
    """Format an order ID for concise logging (shows last 6 digits)."""
    try:
        id_str = str(order_id).strip() if order_id else ""
        if not id_str: return 'UNKNOWN'
        return "..." + id_str[-6:] if len(id_str) > 6 else id_str
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting order ID {order_id}: {e}{Style.RESET_ALL}")
        return 'UNKNOWN'

def send_sms_alert(message: str, config: Optional[ConfigPlaceholder] = None) -> bool:
    """Send an SMS alert using Termux API."""
    enabled = getattr(config, 'ENABLE_SMS_ALERTS', False) if config else False
    if not enabled: return False
    recipient = getattr(config, 'SMS_RECIPIENT_NUMBER', None)
    if not recipient:
        logger.warning("SMS alerts enabled but no SMS_RECIPIENT_NUMBER configured.")
        return False
    timeout = getattr(config, 'SMS_TIMEOUT_SECONDS', 30)
    try:
        logger.info(f"Attempting to send SMS alert via Termux to {recipient}...")
        command = ["termux-sms-send", "-n", recipient, message]
        result = subprocess.run(command, timeout=timeout, check=True, capture_output=True, text=True)
        logger.info(f"{Fore.GREEN}SMS Alert Sent Successfully via Termux.{Style.RESET_ALL} Output: {result.stdout.strip() if result.stdout else '(No output)'}")
        return True
    except FileNotFoundError:
        logger.error(f"{Fore.RED}Termux API command 'termux-sms-send' not found. Is Termux:API installed and configured?{Style.RESET_ALL}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"{Fore.RED}Termux SMS command timed out after {timeout} seconds.{Style.RESET_ALL}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"{Fore.RED}Termux SMS command failed with exit code {e.returncode}.{Style.RESET_ALL}")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Stderr: {e.stderr.strip() if e.stderr else '(No stderr)'}")
        return False
    except Exception as e:
        logger.critical(f"{Fore.RED}Unexpected error sending SMS alert via Termux: {e}{Style.RESET_ALL}", exc_info=True)
        return False

# --- Retry Decorator Factory ---
T = TypeVar('T')

def retry_api_call(
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    handled_exceptions=(ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout),
    error_message_prefix: str = "API Call Failed"
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory to retry API calls with configurable settings."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            config = kwargs.get('config') or next((arg for arg in args if hasattr(arg, 'RETRY_COUNT') and hasattr(arg, 'RETRY_DELAY_SECONDS')), None)
            if not config:
                logger.error(f"{Fore.RED}No config object found for retry_api_call in {func.__name__}{Style.RESET_ALL}")
                raise ValueError("Config object required for retry_api_call")

            effective_max_retries = max_retries if max_retries is not None else getattr(config, 'RETRY_COUNT', 3)
            effective_base_delay = initial_delay if initial_delay is not None else getattr(config, 'RETRY_DELAY_SECONDS', 1.0)
            func_name = func.__name__
            last_exception = None # Store last exception for final raise

            for attempt in range(effective_max_retries + 1):
                try:
                    if attempt > 0: logger.debug(f"Retrying {func_name} (Attempt {attempt}/{effective_max_retries})")
                    return func(*args, **kwargs)
                except handled_exceptions as e:
                    last_exception = e
                    if attempt >= effective_max_retries:
                        logger.error(f"{Fore.RED}{error_message_prefix}: Max retries ({effective_max_retries}) reached for {func_name}. Last error: {type(e).__name__} - {e}{Style.RESET_ALL}")
                        send_sms_alert(f"{error_message_prefix}: Max retries for {func_name} ({type(e).__name__})", config)
                        break # Exit loop, will raise last_exception below

                    delay = effective_base_delay
                    if isinstance(e, ccxt.RateLimitExceeded):
                        delay *= (2 ** attempt) # Exponential backoff for rate limits
                        logger.warning(f"{Fore.YELLOW}Rate limit exceeded in {func_name}. Retry {attempt + 1}/{effective_max_retries} after {delay:.2f}s: {e}{Style.RESET_ALL}")
                    else: # NetworkError, ExchangeNotAvailable, RequestTimeout
                        logger.error(f"{Fore.RED}{type(e).__name__} in {func_name}. Retry {attempt + 1}/{effective_max_retries} after {delay:.2f}s: {e}{Style.RESET_ALL}")

                    time.sleep(delay)

                except Exception as e:
                    logger.critical(f"{Back.RED}{Fore.WHITE}Unexpected critical error in {func_name}: {type(e).__name__} - {e}{Style.RESET_ALL}", exc_info=True)
                    send_sms_alert(f"CRITICAL Error in {func_name}: {type(e).__name__}", config)
                    raise e # Re-raise unexpected exceptions immediately

            # If loop finished without returning, raise the last handled exception
            if last_exception:
                raise last_exception
            else: # Should not happen unless max_retries is negative, but safeguard
                 raise Exception(f"Failed to execute {func_name} after {effective_max_retries} retries (unknown error)")

        return wrapper
    return decorator

# --- Order Book Analysis ---

@retry_api_call() # Use default retry settings from config
def analyze_order_book(
    exchange: ccxt.bybit,
    symbol: str,
    depth: int,
    fetch_limit: int,
    config: ConfigPlaceholder
) -> Dict[str, Optional[Decimal]]:
    """Fetches and analyzes the L2 order book."""
    func_name = "analyze_order_book"
    logger.debug(f"[{func_name}] Analyzing OB for {symbol} (Fetch Limit: {fetch_limit}, Analysis Depth: {depth})")

    analysis_result = {
        'best_bid': None, 'best_ask': None, 'mid_price': None, 'spread': None,
        'spread_pct': None, 'bid_volume_depth': None, 'ask_volume_depth': None,
        'bid_ask_ratio_depth': None
    }

    try:
        logger.debug(f"[{func_name}] Fetching order book data (limit={fetch_limit})...")
        effective_fetch_limit = max(depth, fetch_limit)
        order_book = exchange.fetch_order_book(symbol, limit=effective_fetch_limit)

        if not isinstance(order_book, dict) or 'bids' not in order_book or 'asks' not in order_book:
            raise ValueError("Invalid order book structure received.")

        bids_raw = order_book['bids']
        asks_raw = order_book['asks']
        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            raise ValueError("Order book 'bids' or 'asks' data is not a list.")
        if not bids_raw or not asks_raw:
            logger.warning(f"[{func_name}] Order book for {symbol} is empty (no bids or asks).")
            return analysis_result

        bids: List[Tuple[Decimal, Decimal]] = []
        asks: List[Tuple[Decimal, Decimal]] = []
        for p, a in bids_raw:
            price = safe_decimal_conversion(p); amount = safe_decimal_conversion(a)
            if price and amount and price > 0 and amount >= 0: bids.append((price, amount))
        for p, a in asks_raw:
            price = safe_decimal_conversion(p); amount = safe_decimal_conversion(a)
            if price and amount and price > 0 and amount >= 0: asks.append((price, amount))

        if not bids or not asks:
            logger.warning(f"[{func_name}] Order book for {symbol} has empty validated bids or asks.")
            return analysis_result

        best_bid = bids[0][0]; best_ask = asks[0][0]
        analysis_result['best_bid'] = best_bid; analysis_result['best_ask'] = best_ask

        if best_bid >= best_ask:
            logger.error(f"{Fore.RED}[{func_name}] Order book crossed: Bid ({best_bid}) >= Ask ({best_ask}) for {symbol}.{Style.RESET_ALL}")
            return analysis_result

        analysis_result['mid_price'] = (best_bid + best_ask) / Decimal("2")
        analysis_result['spread'] = best_ask - best_bid
        analysis_result['spread_pct'] = (analysis_result['spread'] / best_bid) * Decimal("100") if best_bid > 0 else Decimal("0")

        bid_vol_depth = sum(b[1] for b in bids[:depth] if b[1] is not None)
        ask_vol_depth = sum(a[1] for a in asks[:depth] if a[1] is not None)
        analysis_result['bid_volume_depth'] = bid_vol_depth
        analysis_result['ask_volume_depth'] = ask_vol_depth
        analysis_result['bid_ask_ratio_depth'] = (bid_vol_depth / ask_vol_depth) if ask_vol_depth and ask_vol_depth > 0 else None

        logger.debug(f"[{func_name}] OB Analysis OK: Spread={analysis_result['spread_pct']:.4f}%, "
                     f"Ratio(d{depth})={analysis_result['bid_ask_ratio_depth']:.2f if analysis_result['bid_ask_ratio_depth'] is not None else 'N/A'}")

        return analysis_result

    except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, TypeError) as e:
        logger.warning(f"{Fore.YELLOW}[{func_name}] Error fetching/analyzing order book for {symbol}: {e}{Style.RESET_ALL}")
        raise # Re-raise for decorator
    except Exception as e:
        logger.critical(f"{Back.RED}{Fore.WHITE}[{func_name}] Unexpected error analyzing order book for {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"CRITICAL: OB Analysis failed for {symbol}", config)
        return analysis_result

# --- END OF FILE bybit_utils.py ---
```

**3. `bybit_helpers.py` (v2.9 - Added `cancel_order`, `fetch_order`, Fixed Ticker)**

```python
# --- START OF FILE bybit_helpers.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bybit V5 CCXT Helper Functions (v2.9 - Add cancel/fetch order, Fix Ticker Validation)

This module provides a collection of robust, reusable, and enhanced helper functions
designed for interacting with the Bybit exchange, specifically targeting the
V5 API (Unified Trading Account - UTA), using the CCXT library.

Core Functionality Includes:
- Exchange Initialization: Securely sets up the ccxt.bybit exchange instance,
  handling testnet mode, V5 defaults, and initial validation.
- Account Configuration: Functions to set leverage, margin mode (cross/isolated),
  and position mode (one-way/hedge) with V5 specific endpoints and validation.
- Market Data Retrieval: Validated fetching of tickers, OHLCV (with pagination
  and DataFrame conversion), L2 order books, funding rates, and recent trades,
  all using Decimals and V5 parameters.
- Order Management: Placing market, limit, native stop-loss (market), and native
  trailing-stop orders with V5 parameters. Includes options for Time-In-Force (TIF),
  reduce-only, post-only, client order IDs, and slippage checks for market orders.
  Also provides functions for cancelling single or all open orders, fetching open
  orders (filtered), and updating existing limit orders (edit). Added cancel_order, fetch_order.
- Position Management: Fetching detailed current position information (V5 specific),
  closing positions using reduce-only market orders, and retrieving detailed position
  risk metrics (IMR, MMR, Liq. Price, etc.) using V5 logic.
- Balance & Margin: Fetching USDT balances (equity/available) using V5 UNIFIED
  account logic, and calculating estimated margin requirements for potential orders.
- Utilities: Market validation against exchange data (type, logic, active status).

Key Enhancements in v2.9:
- Added cancel_order and fetch_order helper functions.
- Fixed ValueError in fetch_ticker_validated when timestamp is missing.
- Improved exception message clarity in fetch_ticker_validated.
- Previous Fixes from v2.1-v2.8.
- Explicitly imports necessary utility functions and decorator from bybit_utils.py.

Dependencies from Importing Script:
- This module now primarily relies on the importing script for:
    1. `logger`: A pre-configured `logging.Logger` object.
    2. `Config`: A configuration class/object instance containing API keys,
       settings, constants (e.g., Config.RETRY_COUNT, Config.SYMBOL, etc.).
       Passed explicitly to functions requiring it.
- Utility functions (`safe_decimal_conversion`, `format_*`, `send_sms_alert`,
  `retry_api_call`, `analyze_order_book`) are imported from `bybit_utils`.
- Ensure `bybit_utils.py` exists and is accessible.
"""

# Standard Library Imports
import logging
import os
import sys
import time
import random # Used in fetch_ohlcv_paginated retry delay jitter
from decimal import Decimal, ROUND_HALF_UP, DivisionByZero, InvalidOperation, getcontext
from typing import Optional, Dict, List, Tuple, Any, Literal, Union, Callable, TypeVar

# Third-party Libraries
try:
    import ccxt
except ImportError:
    print("Error: CCXT library not found. Please install it: pip install ccxt")
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    print("Error: pandas library not found. Please install it: pip install pandas")
    sys.exit(1)
try:
    from colorama import Fore, Style, Back
except ImportError:
    print("Warning: colorama library not found. Logs will not be colored. Install: pip install colorama")
    # Define dummy color constants if colorama is not available
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor()

# --- Import Utilities from bybit_utils ---
try:
    from bybit_utils import (
        safe_decimal_conversion, format_price, format_amount,
        format_order_id, send_sms_alert, retry_api_call,
        analyze_order_book # Ensure analyze_order_book is defined in bybit_utils
    )
    print("Successfully imported utilities from bybit_utils.")
except ImportError as e:
    print(f"FATAL ERROR: Failed to import required functions/decorator from bybit_utils.py: {e}")
    print("Ensure bybit_utils.py is in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)
except NameError as e:
    # This might happen if retry_api_call itself isn't defined correctly *within* bybit_utils
    print(f"FATAL ERROR: A required name (likely 'retry_api_call') is not defined in bybit_utils.py: {e}")
    sys.exit(1)

# Set Decimal context precision
getcontext().prec = 28

# --- Logger Placeholder (Actual logger MUST be provided by importing script) ---
if 'logger' not in globals():
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)d:%(funcName)s} - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    logger.info("Placeholder logger initialized for bybit_helpers module. Ensure a configured logger is provided by the importing script.")

# --- External Dependencies Placeholder Reminder ---
# ======================================================================
# class Config: pass # Must be defined in importing script/passed to functions
# ======================================================================


# --- Helper Function Implementations ---

def _get_v5_category(market: Dict[str, Any]) -> Optional[Literal['linear', 'inverse', 'spot', 'option']]:
    """Internal helper to determine the Bybit V5 category from a market object."""
    func_name = "_get_v5_category"
    if not market: return None
    if market.get('linear'): return 'linear'
    if market.get('inverse'): return 'inverse'
    if market.get('spot'): return 'spot'
    if market.get('option'): return 'option'
    market_type = market.get('type')
    if market_type == 'swap':
        contract_type = market.get('contractType', '').lower()
        return 'linear' if contract_type == 'linear' else ('inverse' if contract_type == 'inverse' else 'linear')
    elif market_type == 'future': return 'linear'
    elif market_type == 'spot': return 'spot'
    elif market_type == 'option': return 'option'
    else:
        logger.warning(f"[{func_name}] Could not determine V5 category for market: {market.get('symbol')}, Type: {market_type}")
        return None

# Snippet 1 / Function 1: Initialize Bybit Exchange
@retry_api_call(max_retries=3, initial_delay=2.0)
def initialize_bybit(config: 'Config') -> Optional[ccxt.bybit]:
    """Initializes and validates the Bybit CCXT exchange instance using V5 API settings."""
    func_name = "initialize_bybit"
    logger.info(f"{Fore.BLUE}[{func_name}] Initializing Bybit (V5) exchange instance...{Style.RESET_ALL}")
    try:
        exchange_class = getattr(ccxt, config.EXCHANGE_ID)
        exchange = exchange_class({
            'apiKey': config.API_KEY, 'secret': config.API_SECRET, 'enableRateLimit': True,
            'options': {
                'defaultType': 'swap', 'adjustForTimeDifference': True,
                'recvWindow': config.DEFAULT_RECV_WINDOW, 'brokerId': 'PyrmethusV2.9'
            }
        })
        if config.TESTNET_MODE:
            logger.info(f"[{func_name}] Enabling Bybit Sandbox (Testnet) mode.")
            exchange.set_sandbox_mode(True)
        logger.debug(f"[{func_name}] Loading markets...")
        exchange.load_markets(reload=True)
        if not exchange.markets: raise ccxt.ExchangeError(f"[{func_name}] Failed to load markets.")
        logger.debug(f"[{func_name}] Markets loaded successfully ({len(exchange.markets)} symbols).")
        logger.debug(f"[{func_name}] Performing initial balance fetch for validation...")
        exchange.fetch_balance({'accountType': 'UNIFIED'})
        logger.debug(f"[{func_name}] Initial balance check successful.")
        try:
            market = exchange.market(config.SYMBOL); category = _get_v5_category(market)
            if category and category in ['linear', 'inverse']:
                logger.debug(f"[{func_name}] Attempting to set initial margin mode '{config.DEFAULT_MARGIN_MODE}' for {config.SYMBOL} (Category: {category})...")
                exchange.set_margin_mode(config.DEFAULT_MARGIN_MODE, config.SYMBOL, params={'category': category})
                logger.info(f"[{func_name}] Initial margin mode potentially set to '{config.DEFAULT_MARGIN_MODE}' for {config.SYMBOL}.")
            else: logger.warning(f"[{func_name}] Cannot determine contract category for {config.SYMBOL}. Skipping initial margin mode set.")
        except (ccxt.NotSupported, ccxt.ExchangeError, ccxt.ArgumentsRequired, ccxt.BadSymbol) as e_margin:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Could not set initial margin mode for {config.SYMBOL}: {e_margin}. Verify account settings.{Style.RESET_ALL}")
        logger.success(f"{Fore.GREEN}[{func_name}] Bybit exchange initialized successfully. Testnet: {config.TESTNET_MODE}.{Style.RESET_ALL}")
        return exchange
    except (ccxt.AuthenticationError, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.ExchangeError) as e:
        logger.error(f"{Fore.RED}[{func_name}] Initialization attempt failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
        raise
    except Exception as e:
        logger.critical(f"{Back.RED}[{func_name}] Unexpected critical error during Bybit initialization: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[BybitHelper] CRITICAL: Bybit init failed! Unexpected: {type(e).__name__}", config)
        return None

# Snippet 2 / Function 2: Set Leverage
@retry_api_call(max_retries=3, initial_delay=1.0)
def set_leverage(exchange: ccxt.bybit, symbol: str, leverage: int, config: 'Config') -> bool:
    """Sets the leverage for a specific symbol on Bybit V5 (Linear/Inverse)."""
    func_name = "set_leverage"; logger.info(f"{Fore.CYAN}[{func_name}] Setting leverage to {leverage}x for {symbol}...{Style.RESET_ALL}")
    if leverage <= 0: logger.error(f"{Fore.RED}[{func_name}] Leverage must be positive: {leverage}{Style.RESET_ALL}"); return False
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}[{func_name}] Invalid market type for leverage: {symbol} ({category}).{Style.RESET_ALL}"); return False
        lev_filter = market.get('info', {}).get('leverageFilter', {}); max_lev_s = lev_filter.get('maxLeverage'); min_lev_s = lev_filter.get('minLeverage', '1')
        max_lev = int(safe_decimal_conversion(max_lev_s, Decimal('100'))); min_lev = int(safe_decimal_conversion(min_lev_s, Decimal('1')))
        if not (min_lev <= leverage <= max_lev): logger.error(f"{Fore.RED}[{func_name}] Invalid leverage {leverage}x. Allowed: {min_lev}x - {max_lev}x.{Style.RESET_ALL}"); return False
        params = {'category': category, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
        logger.debug(f"[{func_name}] Calling exchange.set_leverage with symbol='{symbol}', leverage={leverage}, params={params}")
        exchange.set_leverage(leverage, symbol, params=params)
        logger.success(f"{Fore.GREEN}[{func_name}] Leverage set/confirmed to {leverage}x for {symbol} (Category: {category}).{Style.RESET_ALL}"); return True
    except ccxt.ExchangeError as e:
        err_str = str(e).lower();
        if "leverage not modified" in err_str or "same as input" in err_str or "110044" in str(e): logger.info(f"{Fore.CYAN}[{func_name}] Leverage for {symbol} is already set to {leverage}x.{Style.RESET_ALL}"); return True
        else: logger.error(f"{Fore.RED}[{func_name}] ExchangeError setting leverage for {symbol} to {leverage}x: {e}{Style.RESET_ALL}"); raise
    except (ccxt.NetworkError, ccxt.AuthenticationError, ccxt.BadSymbol) as e: logger.error(f"{Fore.RED}[{func_name}] API/Symbol error setting leverage for {symbol}: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error setting leverage for {symbol} to {leverage}x: {e}{Style.RESET_ALL}", exc_info=True); send_sms_alert(f"[{symbol.split('/')[0]}] ERROR: Failed set leverage {leverage}x (Unexpected)", config); return False

# Snippet 3 / Function 3: Fetch USDT Balance (V5 UNIFIED)
@retry_api_call(max_retries=3, initial_delay=1.0)
def fetch_usdt_balance(exchange: ccxt.bybit, config: 'Config') -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Fetches the USDT balance (Total Equity and Available Balance) using Bybit V5 UNIFIED account logic."""
    func_name = "fetch_usdt_balance"; logger.debug(f"[{func_name}] Fetching USDT balance (Bybit V5 UNIFIED Account)...")
    try:
        balance_data = exchange.fetch_balance(params={'accountType': 'UNIFIED'})
        info = balance_data.get('info', {}); result_list = info.get('result', {}).get('list', [])
        equity, available, acct_type = None, None, "N/A"
        if result_list:
            unified_info = next((acc for acc in result_list if acc.get('accountType') == 'UNIFIED'), None)
            if unified_info:
                acct_type = "UNIFIED"; equity = safe_decimal_conversion(unified_info.get('totalEquity'))
                usdt_info = next((c for c in unified_info.get('coin', []) if c.get('coin') == config.USDT_SYMBOL), None)
                if usdt_info: available = safe_decimal_conversion(usdt_info.get('availableToWithdraw') or usdt_info.get('availableBalance'), Decimal("0.0"))
                else: logger.warning(f"[{func_name}] USDT coin data not found in UNIFIED. Assuming 0 available."); available = Decimal("0.0")
            elif len(result_list) >= 1:
                first_acct = result_list[0]; acct_type = first_acct.get('accountType', 'UNKNOWN'); logger.warning(f"[{func_name}] UNIFIED not found. Using first account: Type '{acct_type}'")
                equity = safe_decimal_conversion(first_acct.get('totalEquity') or first_acct.get('equity'))
                usdt_info = next((c for c in first_acct.get('coin', []) if c.get('coin') == config.USDT_SYMBOL), None)
                available = safe_decimal_conversion(usdt_info.get('availableBalance'), Decimal("0.0")) if usdt_info else Decimal("0.0")
            else: logger.error(f"[{func_name}] Balance response list empty.")
        if equity is None or available is None:
            logger.debug(f"[{func_name}] V5 parsing incomplete. Trying standard CCXT keys...")
            std_bal = balance_data.get(config.USDT_SYMBOL, {});
            if equity is None: equity = safe_decimal_conversion(std_bal.get('total'))
            if available is None: available = safe_decimal_conversion(std_bal.get('free'))
            if available is None and equity is not None: logger.warning(f"[{func_name}] CCXT 'free' missing, using 'total' ({equity:.4f}) as fallback."); available = equity
            if equity is not None and available is not None: acct_type = "CCXT Standard Fallback"
            else: raise ValueError(f"Failed to parse balance from V5 ({acct_type}) and Standard.")
        final_equity = max(Decimal("0.0"), equity or Decimal("0.0"))
        final_available = max(Decimal("0.0"), available or Decimal("0.0"))
        logger.info(f"[{func_name}] USDT Balance Fetched (Source: {acct_type}): Equity = {final_equity:.4f}, Available = {final_available:.4f}")
        return final_equity, final_available
    except (ccxt.NetworkError, ccxt.ExchangeError, ValueError) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] Error fetching/parsing balance: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.critical(f"{Back.RED}[{func_name}] Unexpected error fetching balance: {e}{Style.RESET_ALL}", exc_info=True); send_sms_alert("[BybitHelper] CRITICAL: Failed fetch USDT balance!", config); return None, None

# Snippet 4 / Function 4: Place Market Order with Slippage Check
@retry_api_call(max_retries=1, initial_delay=0)
def place_market_order_slippage_check(
    exchange: ccxt.bybit, symbol: str, side: Literal['buy', 'sell'], amount: Decimal,
    config: 'Config', max_slippage_pct: Optional[Decimal] = None,
    is_reduce_only: bool = False, client_order_id: Optional[str] = None
) -> Optional[Dict]:
    """Places a market order on Bybit V5 after checking the current spread against a slippage threshold."""
    func_name = "place_market_order_slippage_check"; market_base = symbol.split('/')[0]; action = "CLOSE" if is_reduce_only else "ENTRY"; log_prefix = f"Market Order ({action} {side.upper()})"
    effective_max_slippage = max_slippage_pct if max_slippage_pct is not None else config.DEFAULT_SLIPPAGE_PCT
    logger.info(f"{Fore.BLUE}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}. Max Slippage: {effective_max_slippage:.4%}, ReduceOnly: {is_reduce_only}{Style.RESET_ALL}")
    if amount <= config.POSITION_QTY_EPSILON: logger.error(f"{Fore.RED}{log_prefix}: Amount is zero or negative ({amount}). Aborting.{Style.RESET_ALL}"); return None
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}[{func_name}] Cannot determine category for {symbol}. Aborting.{Style.RESET_ALL}"); return None
        logger.debug(f"[{func_name}] Performing pre-order slippage check (Depth: {config.SHALLOW_OB_FETCH_DEPTH})...")
        ob_analysis = analyze_order_book(exchange, symbol, config.SHALLOW_OB_FETCH_DEPTH, config.ORDER_BOOK_FETCH_LIMIT, config)
        best_ask, best_bid = ob_analysis.get("best_ask"), ob_analysis.get("best_bid")
        if best_bid and best_ask and best_bid > Decimal("0"):
            spread = (best_ask - best_bid) / best_bid
            logger.debug(f"[{func_name}] Current OB: Bid={format_price(exchange, symbol, best_bid)}, Ask={format_price(exchange, symbol, best_ask)}, Spread={spread:.4%}")
            if spread > effective_max_slippage: logger.error(f"{Fore.RED}{log_prefix}: Aborted due to high slippage {spread:.4%} > Max {effective_max_slippage:.4%}.{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ORDER ABORT ({side.upper()}): High Slippage {spread:.4%}", config); return None
        else: logger.warning(f"{Fore.YELLOW}{log_prefix}: Could not get valid OB data for slippage check. Proceeding cautiously.{Style.RESET_ALL}")
        amount_str = format_amount(exchange, symbol, amount); amount_float = float(amount_str)
        params: Dict[str, Any] = {'category': category}
        if is_reduce_only: params['reduceOnly'] = True
        if client_order_id:
            max_coid_len = 36; original_len = len(client_order_id); valid_coid = client_order_id[:max_coid_len]
            params['clientOrderId'] = valid_coid
            if len(valid_coid) < original_len: logger.warning(f"[{func_name}] Client OID truncated: '{valid_coid}' (Orig len: {original_len})")
        bg = Back.GREEN if side == config.SIDE_BUY else Back.RED; fg = Fore.BLACK
        logger.warning(f"{bg}{fg}{Style.BRIGHT}*** PLACING MARKET {side.upper()} {'REDUCE' if is_reduce_only else 'ENTRY'}: {amount_str} {symbol} (Params: {params}) ***{Style.RESET_ALL}")
        order = exchange.create_market_order(symbol, side, amount_float, params=params)
        order_id = order.get('id'); client_oid_resp = order.get('clientOrderId', params.get('clientOrderId', 'N/A')); status = order.get('status', '?')
        filled_qty = safe_decimal_conversion(order.get('filled', '0.0')); avg_price = safe_decimal_conversion(order.get('average'))
        logger.success(f"{Fore.GREEN}{log_prefix}: Submitted. ID: ...{format_order_id(order_id)}, ClientOID: {client_oid_resp}, Status: {status}, Filled Qty: {format_amount(exchange, symbol, filled_qty)}, Avg Px: {format_price(exchange, symbol, avg_price)}{Style.RESET_ALL}")
        return order
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError) as e: logger.error(f"{Fore.RED}{log_prefix}: API Error: {type(e).__name__} - {e}{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()} {action}): {type(e).__name__}", config); return None
    except Exception as e: logger.critical(f"{Back.RED}[{func_name}] Unexpected error placing market order: {e}{Style.RESET_ALL}", exc_info=True); send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()} {action}): Unexpected {type(e).__name__}.", config); return None

# Snippet 5 / Function 5: Cancel All Open Orders
@retry_api_call(max_retries=2, initial_delay=1.0)
def cancel_all_orders(exchange: ccxt.bybit, symbol: str, config: 'Config', reason: str = "Cleanup", order_filter: Optional[str] = 'Order') -> bool:
    """Cancels all open orders matching a filter for a specific symbol on Bybit V5."""
    func_name = "cancel_all_orders"; market_base = symbol.split('/')[0]; log_prefix = f"Cancel All ({reason}, Filter: {order_filter})"
    logger.info(f"{Fore.CYAN}[{func_name}] {log_prefix}: Attempting for {symbol}...{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}[{func_name}] Cannot determine category for {symbol}. Aborting.{Style.RESET_ALL}"); return False
        fetch_params = {'category': category, 'orderFilter': order_filter}
        logger.debug(f"[{func_name}] Fetching open {order_filter} orders for {symbol} with params: {fetch_params}")
        open_orders = exchange.fetch_open_orders(symbol, params=fetch_params)
        if not open_orders: logger.info(f"{Fore.CYAN}[{func_name}] {log_prefix}: No open orders matching filter '{order_filter}' found.{Style.RESET_ALL}"); return True
        logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: Found {len(open_orders)} open order(s). Attempting cancellation...{Style.RESET_ALL}")
        success_count, fail_count = 0, 0; cancel_delay = max(0.05, 1.0 / (exchange.rateLimit or 2000) * 1.1) # Add buffer
        cancel_params = {'category': category}
        if order_filter == 'StopOrder': cancel_params['orderFilter'] = 'StopOrder' # Required by Bybit V5 sometimes
        for order in open_orders:
            order_id = order.get('id')
            if not order_id: logger.warning(f"[{func_name}] Skipping order with missing ID: {order}"); continue
            order_info_log = f"ID: ...{format_order_id(order_id)} ({order.get('type','?').upper()} {order.get('side','?').upper()} {format_amount(exchange, symbol, order.get('amount'))})"
            try:
                logger.debug(f"[{func_name}] Cancelling {order_info_log} with params: {cancel_params}")
                cancel_order(exchange, symbol, order_id, config=config) # Use the single cancel helper
                success_count += 1
            except (ccxt.NetworkError, ccxt.RateLimitExceeded) as e_cancel: logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: Network/RateLimit error cancelling {order_info_log}: {e_cancel}. Continuing...{Style.RESET_ALL}"); fail_count += 1
            except ccxt.OrderNotFound: logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: Order {order_info_log} already gone (Not Found). OK.{Style.RESET_ALL}"); success_count += 1 # Count as success
            except ccxt.ExchangeError as e_cancel: logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: FAILED to cancel {order_info_log}: {e_cancel}{Style.RESET_ALL}"); fail_count += 1
            except Exception as e_cancel: logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: Unexpected error cancelling {order_info_log}: {e_cancel}{Style.RESET_ALL}", exc_info=True); fail_count += 1
            time.sleep(cancel_delay) # Add small delay between cancels
        if fail_count > 0:
             try: # Re-check
                 remaining_orders = exchange.fetch_open_orders(symbol, params=fetch_params)
                 if not remaining_orders: logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: Initial failures, but re-check shows no open orders.{Style.RESET_ALL}"); return True
                 else: logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: Failed: {fail_count}, Success/Gone: {success_count}. {len(remaining_orders)} order(s) may remain.{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] ERROR: Failed cancel {fail_count} orders ({reason}).", config); return False
             except Exception as e_recheck: logger.error(f"[{func_name}] Error re-checking orders: {e_recheck}. Assuming failures."); send_sms_alert(f"[{market_base}] ERROR: Failed cancel {fail_count} orders ({reason}).", config); return False
        else: logger.success(f"{Fore.GREEN}[{func_name}] {log_prefix}: Successfully cancelled/confirmed gone all {len(open_orders)} open orders.{Style.RESET_ALL}"); return True
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e: logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: API error during cancel all: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: Unexpected error during cancel all: {e}{Style.RESET_ALL}", exc_info=True); return False

# --- Added cancel_order function ---
@retry_api_call(max_retries=2, initial_delay=0.5)
def cancel_order(exchange: ccxt.bybit, symbol: str, order_id: str, config: 'Config') -> bool:
    """Cancels a single specific order by ID."""
    func_name = "cancel_order"; log_prefix = f"Cancel Order ...{format_order_id(order_id)}"
    logger.info(f"{Fore.CYAN}[{func_name}] {log_prefix}: Attempting for {symbol}...{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}[{func_name}] Cannot determine category for {symbol}. Aborting.{Style.RESET_ALL}"); return False
        params = {'category': category}
        # Bybit V5 cancelOrder requires orderFilter for Stop/TP/SL orders
        # To be robust, we should try fetching the order first to determine type,
        # but that adds latency. Let's try standard first, then StopOrder if it fails?
        # Or pass order_filter explicitly if known?
        # Simplest: Assume standard cancel works for most cases. User should ensure correct call if needed.
        logger.debug(f"[{func_name}] Calling exchange.cancel_order ID={order_id}, Symbol={symbol}, Params={params}")
        exchange.cancel_order(order_id, symbol, params=params)
        logger.success(f"{Fore.GREEN}[{func_name}] {log_prefix}: Successfully cancelled order.{Style.RESET_ALL}")
        return True
    except ccxt.OrderNotFound:
        logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: Order already gone or not found.{Style.RESET_ALL}")
        return True # Treat as success
    except ccxt.ExchangeError as e:
         # Check if error indicates filter needed (e.g., order not found but might be stop) - Requires inspecting Bybit error codes
         # Example code: 110001: "Order does not exist." (might apply to wrong filter too)
         err_code = getattr(e, 'code', None) or str(e) # Get error code if available
         if "order does not exist" in str(e).lower() or "110001" in str(e):
              logger.warning(f"[{func_name}] Order {order_id} not found with standard cancel params. It might be a Stop/TP/SL or already gone.")
              # Optionally try again with filter='StopOrder', but could lead to infinite loop if not careful
              # return cancel_order_with_filter(exchange, symbol, order_id, config, 'StopOrder') # Example recursive call
              return True # Assume gone for now if not found
         logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: API error cancelling: {type(e).__name__} - {e}{Style.RESET_ALL}")
         raise # Re-raise other exchange errors
    except (ccxt.NetworkError) as e: # Separate NetworkError for retry
        logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: Network error cancelling: {e}{Style.RESET_ALL}")
        raise # Re-raise for retry
    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: Unexpected error cancelling: {e}{Style.RESET_ALL}", exc_info=True)
        return False

# --- Added fetch_order function ---
@retry_api_call(max_retries=3, initial_delay=0.5)
def fetch_order(exchange: ccxt.bybit, symbol: str, order_id: str, config: 'Config') -> Optional[Dict]:
    """Fetches details for a single specific order by ID."""
    func_name = "fetch_order"; log_prefix = f"Fetch Order ...{format_order_id(order_id)}"
    logger.debug(f"[{func_name}] {log_prefix}: Attempting for {symbol}...")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}[{func_name}] Cannot determine category for {symbol}.{Style.RESET_ALL}"); return None
        params = {'category': category}
        # Again, might need orderFilter for Stop orders
        logger.debug(f"[{func_name}] Calling exchange.fetch_order ID={order_id}, Symbol={symbol}, Params={params}")
        order_data = exchange.fetch_order(order_id, symbol, params=params)
        if order_data: logger.debug(f"[{func_name}] {log_prefix}: Order data fetched. Status: {order_data.get('status')}"); return order_data
        else: logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: fetch_order returned no data.{Style.RESET_ALL}"); return None
    except ccxt.OrderNotFound:
        logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: Order not found.{Style.RESET_ALL}")
        return None
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        err_code = getattr(e, 'code', None) or str(e)
        if "order does not exist" in str(e).lower() or "110001" in str(e): # Treat as OrderNotFound
             logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: Order not found (via ExchangeError).{Style.RESET_ALL}")
             return None
        logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: API error fetching: {type(e).__name__} - {e}{Style.RESET_ALL}")
        raise
    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: Unexpected error fetching: {e}{Style.RESET_ALL}", exc_info=True)
        return None


# --- fetch_ohlcv_paginated ---
# ... (Function is identical to v2.8) ...
def fetch_ohlcv_paginated(
    exchange: ccxt.bybit,
    symbol: str,
    timeframe: str,
    config: 'Config',
    since: Optional[int] = None,
    limit_per_req: int = 1000, # Bybit V5 max limit is 1000
    max_total_candles: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetches historical OHLCV data for a symbol using pagination to handle limits.
    """
    func_name = "fetch_ohlcv_paginated"
    if not exchange.has.get("fetchOHLCV"):
        logger.error(f"{Fore.RED}[{func_name}] The exchange '{exchange.id}' does not support fetchOHLCV.{Style.RESET_ALL}")
        return None

    try:
        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
        if limit_per_req > 1000:
            logger.warning(f"[{func_name}] Requested limit_per_req ({limit_per_req}) exceeds Bybit V5 max (1000). Clamping to 1000.")
            limit_per_req = 1000

        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.warning(f"[{func_name}] Could not determine category for {symbol}. Assuming 'linear'. This might fail for Spot/Inverse.")
            category = 'linear' # Default assumption

        params = {'category': category}

        logger.info(f"{Fore.BLUE}[{func_name}] Fetching {symbol} OHLCV ({timeframe}). "
                    f"Limit/Req: {limit_per_req}, Since: {pd.to_datetime(since, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M:%S') if since else 'Recent'}. "
                    f"Max Total: {max_total_candles or 'Unlimited'}{Style.RESET_ALL}")

        all_candles: List[list] = []
        current_since = since
        request_count = 0
        max_requests = float('inf')
        if max_total_candles:
            max_requests = (max_total_candles + limit_per_req - 1) // limit_per_req

        while request_count < max_requests:
            if max_total_candles and len(all_candles) >= max_total_candles:
                logger.info(f"[{func_name}] Reached max_total_candles limit ({max_total_candles}). Fetch complete.")
                break

            request_count += 1
            fetch_limit = limit_per_req
            if max_total_candles:
                remaining_needed = max_total_candles - len(all_candles)
                fetch_limit = min(limit_per_req, remaining_needed)

            logger.debug(f"[{func_name}] Fetch Chunk #{request_count}: Since={current_since}, Limit={fetch_limit}, Params={params}")

            candles_chunk: Optional[List[list]] = None
            last_fetch_error: Optional[Exception] = None

            # Internal retry loop for fetching this specific chunk
            for attempt in range(config.RETRY_COUNT):
                try:
                    candles_chunk = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=fetch_limit, params=params)
                    last_fetch_error = None; break
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.RateLimitExceeded) as e:
                    last_fetch_error = e
                    retry_delay = config.RETRY_DELAY_SECONDS * (attempt + 1) * (random.uniform(0.8, 1.2) if 'random' in globals() else 1.0)
                    logger.warning(f"{Fore.YELLOW}[{func_name}] API Error chunk #{request_count} (Try {attempt + 1}/{config.RETRY_COUNT}): {e}. Retrying in {retry_delay:.2f}s...{Style.RESET_ALL}")
                    time.sleep(retry_delay)
                except ccxt.ExchangeError as e: last_fetch_error = e; logger.error(f"{Fore.RED}[{func_name}] ExchangeError chunk #{request_count}: {e}. Aborting chunk.{Style.RESET_ALL}"); break
                except Exception as e: last_fetch_error = e; logger.error(f"[{func_name}] Unexpected fetch chunk #{request_count} err: {e}", exc_info=True); break

            if last_fetch_error:
                logger.error(f"{Fore.RED}[{func_name}] Failed to fetch chunk #{request_count} after {config.RETRY_COUNT} attempts. Last Error: {last_fetch_error}{Style.RESET_ALL}")
                logger.warning(f"[{func_name}] Returning potentially incomplete data ({len(all_candles)} candles) due to fetch failure.")
                break

            if not candles_chunk: logger.debug(f"[{func_name}] No more candles returned (Chunk #{request_count})."); break

            if all_candles and candles_chunk[0][0] <= all_candles[-1][0]:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Overlap detected chunk #{request_count}. Filtering.")
                first_new_ts = all_candles[-1][0] + 1 # Start one ms after last known candle
                candles_chunk = [c for c in candles_chunk if c[0] >= first_new_ts]
                if not candles_chunk: logger.debug(f"[{func_name}] Entire chunk was overlap or duplicate."); break # Break if no new candles after filtering

            logger.debug(f"[{func_name}] Fetched {len(candles_chunk)} new candles (Chunk #{request_count}). Total: {len(all_candles) + len(candles_chunk)}")
            all_candles.extend(candles_chunk)

            if len(candles_chunk) < fetch_limit: logger.debug(f"[{func_name}] Received fewer candles than requested. End of data."); break

            # Update 'since' for the next request based on the timestamp of the *last* candle received
            current_since = candles_chunk[-1][0] + 1 # Request starting *after* the last received timestamp

            # Add a small delay based on rate limit
            time.sleep(max(0.05, 1.0 / (exchange.rateLimit if exchange.rateLimit and exchange.rateLimit > 0 else 10))) # Ensure minimum delay

        return _process_ohlcv_list(all_candles, func_name, symbol, timeframe, max_total_candles)

    except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
         logger.error(f"{Fore.RED}[{func_name}] Initial setup error for OHLCV fetch ({symbol}, {timeframe}): {e}{Style.RESET_ALL}")
         return None
    except Exception as e:
        logger.critical(f"{Back.RED}[{func_name}] Unexpected critical error during OHLCV pagination setup: {e}{Style.RESET_ALL}", exc_info=True)
        return None

# --- _process_ohlcv_list ---
# ... (Function is identical to v2.8) ...
def _process_ohlcv_list(
    candle_list: List[list], parent_func_name: str, symbol: str, timeframe: str, max_candles: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """Internal helper to convert OHLCV list to validated pandas DataFrame."""
    func_name = f"{parent_func_name}._process_ohlcv_list"
    if not candle_list:
        logger.warning(f"{Fore.YELLOW}[{func_name}] No candles collected for {symbol} ({timeframe}). Returning empty DataFrame.{Style.RESET_ALL}")
        cols = ['open', 'high', 'low', 'close', 'volume']; empty_df = pd.DataFrame(columns=cols).astype({c: float for c in ['open','high','low','close','volume']})
        empty_df.index = pd.to_datetime([]).tz_localize('UTC'); empty_df.index.name = 'timestamp'
        return empty_df
    logger.debug(f"[{func_name}] Processing {len(candle_list)} raw candles for {symbol} ({timeframe})...")
    try:
        df = pd.DataFrame(candle_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce'); df.dropna(subset=['timestamp'], inplace=True)
        if df.empty: raise ValueError("All timestamp conversions failed.")
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        initial_len = len(df); df = df[~df.index.duplicated(keep='first')]
        if len(df) < initial_len: logger.debug(f"[{func_name}] Removed {initial_len - len(df)} duplicate timestamp entries.")
        nan_counts = df.isnull().sum(); total_nans = nan_counts.sum()
        if total_nans > 0:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Found {total_nans} NaNs. Ffilling... (Counts: {nan_counts.to_dict()}){Style.RESET_ALL}")
            df.ffill(inplace=True); df.dropna(inplace=True) # Drop any remaining NaNs at the start
            if df.isnull().sum().sum() > 0: logger.error(f"{Fore.RED}[{func_name}] NaNs persisted after fill!{Style.RESET_ALL}")
        df.sort_index(inplace=True)
        if max_candles and len(df) > max_candles: logger.debug(f"[{func_name}] Trimming DF to last {max_candles}."); df = df.iloc[-max_candles:]
        if df.empty: logger.error(f"{Fore.RED}[{func_name}] Processed DF is empty after cleaning.{Style.RESET_ALL}"); return df
        logger.success(f"{Fore.GREEN}[{func_name}] Processed {len(df)} valid candles for {symbol} ({timeframe}).{Style.RESET_ALL}")
        return df
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Error processing OHLCV list: {e}{Style.RESET_ALL}", exc_info=True); return None


# --- place_limit_order_tif ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=1, initial_delay=0)
def place_limit_order_tif(
    exchange: ccxt.bybit, symbol: str, side: Literal['buy', 'sell'], amount: Decimal, price: Decimal, config: 'Config',
    time_in_force: str = 'GTC', is_reduce_only: bool = False, is_post_only: bool = False, client_order_id: Optional[str] = None
) -> Optional[Dict]:
    """Places a limit order on Bybit V5 with options for Time-In-Force, Post-Only, and Reduce-Only."""
    func_name = "place_limit_order_tif"; log_prefix = f"Limit Order ({side.upper()})"
    logger.info(f"{Fore.BLUE}{log_prefix}: Init {format_amount(exchange, symbol, amount)} @ {format_price(exchange, symbol, price)} (TIF:{time_in_force}, Reduce:{is_reduce_only}, Post:{is_post_only})...{Style.RESET_ALL}")
    if amount <= config.POSITION_QTY_EPSILON or price <= Decimal("0"): logger.error(f"{Fore.RED}{log_prefix}: Invalid amount/price.{Style.RESET_ALL}"); return None
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}[{func_name}] Cannot determine category.{Style.RESET_ALL}"); return None
        amount_str = format_amount(exchange, symbol, amount); price_str = format_price(exchange, symbol, price); amount_float = float(amount_str); price_float = float(price_str)
        params: Dict[str, Any] = {'category': category}
        valid_tif = ['GTC', 'IOC', 'FOK']; tif_upper = time_in_force.upper()
        if tif_upper in valid_tif: params['timeInForce'] = tif_upper
        else: logger.warning(f"[{func_name}] Unsupported TIF '{time_in_force}'. Using GTC."); params['timeInForce'] = 'GTC'
        if is_post_only: params['postOnly'] = True
        if is_reduce_only: params['reduceOnly'] = True

        if client_order_id:
            max_coid_len = 36; original_len = len(client_order_id); valid_coid = client_order_id[:max_coid_len]
            params['clientOrderId'] = valid_coid
            if len(valid_coid) < original_len: logger.warning(f"[{func_name}] Client OID truncated: '{valid_coid}' (Orig len: {original_len})")

        logger.info(f"{Fore.CYAN}{log_prefix}: Placing -> Amt:{amount_float}, Px:{price_float}, Params:{params}{Style.RESET_ALL}")
        order = exchange.create_limit_order(symbol, side, amount_float, price_float, params=params)
        order_id = order.get('id'); client_oid_resp = order.get('clientOrderId', params.get('clientOrderId', 'N/A')); status = order.get('status', '?'); effective_tif = order.get('timeInForce', params.get('timeInForce', '?')); is_post_only_resp = order.get('postOnly', params.get('postOnly', False))
        logger.success(f"{Fore.GREEN}{log_prefix}: Limit order placed. ID:...{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, TIF:{effective_tif}, Post:{is_post_only_resp}{Style.RESET_ALL}")
        return order
    except ccxt.OrderImmediatelyFillable as e:
         if params.get('postOnly'): logger.warning(f"{Fore.YELLOW}{log_prefix}: PostOnly failed (immediate match): {e}{Style.RESET_ALL}"); return None
         else: logger.error(f"{Fore.RED}{log_prefix}: Unexpected OrderImmediatelyFillable: {e}{Style.RESET_ALL}"); raise e
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError) as e: logger.error(f"{Fore.RED}{log_prefix}: API Error: {type(e).__name__} - {e}{Style.RESET_ALL}"); send_sms_alert(f"[{symbol.split('/')[0]}] ORDER FAIL (Limit {side.upper()}): {type(e).__name__}", config); return None
    except Exception as e: logger.critical(f"{Back.RED}[{func_name}] Unexpected error: {e}{Style.RESET_ALL}", exc_info=True); send_sms_alert(f"[{symbol.split('/')[0]}] ORDER FAIL (Limit {side.upper()}): Unexpected {type(e).__name__}.", config); return None


# --- get_current_position_bybit_v5 ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=3, initial_delay=1.0)
def get_current_position_bybit_v5(exchange: ccxt.bybit, symbol: str, config: 'Config') -> Dict[str, Any]:
    """Fetches the current position details for a symbol using Bybit V5's fetchPositions logic."""
    func_name = "get_current_position_bybit_v5"; logger.debug(f"[{func_name}] Fetching position for {symbol} (V5)...")
    default_position: Dict[str, Any] = {'symbol': symbol, 'side': config.POS_NONE, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0"), 'liq_price': None, 'mark_price': None, 'pnl_unrealized': None, 'leverage': None, 'info': {}}
    try:
        market = exchange.market(symbol); market_id = market['id']; category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}[{func_name}] Not a contract symbol: {symbol}.{Style.RESET_ALL}"); return default_position
        if not exchange.has.get('fetchPositions'): logger.error(f"{Fore.RED}[{func_name}] fetchPositions not available.{Style.RESET_ALL}"); return default_position
        params = {'category': category, 'symbol': market_id}; logger.debug(f"[{func_name}] Calling fetch_positions with params: {params}")
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)
        active_position_data: Optional[Dict] = None
        for pos in fetched_positions:
            pos_info = pos.get('info', {}); pos_symbol = pos_info.get('symbol'); pos_v5_side = pos_info.get('side', 'None'); pos_size_str = pos_info.get('size'); pos_idx = int(pos_info.get('positionIdx', -1))
            if pos_symbol == market_id and pos_v5_side != 'None' and pos_idx == 0:
                pos_size = safe_decimal_conversion(pos_size_str, Decimal("0.0"))
                if pos_size is not None and abs(pos_size) > config.POSITION_QTY_EPSILON: active_position_data = pos; logger.debug(f"[{func_name}] Found active One-Way (idx 0) position."); break
        if active_position_data:
            try:
                info = active_position_data.get('info', {}); size = safe_decimal_conversion(info.get('size')); entry_price = safe_decimal_conversion(info.get('avgPrice')); liq_price = safe_decimal_conversion(info.get('liqPrice')); mark_price = safe_decimal_conversion(info.get('markPrice')); pnl = safe_decimal_conversion(info.get('unrealisedPnl')); leverage = safe_decimal_conversion(info.get('leverage'))
                pos_side_str = info.get('side'); position_side = config.POS_LONG if pos_side_str == 'Buy' else (config.POS_SHORT if pos_side_str == 'Sell' else config.POS_NONE); quantity = abs(size) if size is not None else Decimal("0.0")
                if position_side == config.POS_NONE or quantity <= config.POSITION_QTY_EPSILON: logger.info(f"[{func_name}] Pos {symbol} negligible size/side."); return default_position
                log_color = Fore.GREEN if position_side == config.POS_LONG else Fore.RED
                logger.info(f"{log_color}[{func_name}] ACTIVE {position_side} {symbol}: Qty={format_amount(exchange, symbol, quantity)}, Entry={format_price(exchange, symbol, entry_price)}, Mark={format_price(exchange, symbol, mark_price)}, Liq~{format_price(exchange, symbol, liq_price)}, uPNL={format_price(exchange, config.USDT_SYMBOL, pnl)}, Lev={leverage}x{Style.RESET_ALL}")
                return {'symbol': symbol, 'side': position_side, 'qty': quantity, 'entry_price': entry_price, 'liq_price': liq_price, 'mark_price': mark_price, 'pnl_unrealized': pnl, 'leverage': leverage, 'info': info }
            except Exception as parse_err: logger.warning(f"{Fore.YELLOW}[{func_name}] Error parsing active pos: {parse_err}. Data: {str(active_position_data)[:300]}{Style.RESET_ALL}"); return default_position
        else: logger.info(f"[{func_name}] No active One-Way position found for {symbol}."); return default_position
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching pos: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching pos: {e}{Style.RESET_ALL}", exc_info=True); return default_position


# --- close_position_reduce_only ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=2, initial_delay=1)
def close_position_reduce_only(
    exchange: ccxt.bybit, symbol: str, config: 'Config', position_to_close: Optional[Dict[str, Any]] = None, reason: str = "Signal Close"
) -> Optional[Dict[str, Any]]:
    """Closes the current position for the given symbol using a reduce-only market order."""
    func_name = "close_position_reduce_only"; market_base = symbol.split('/')[0]; log_prefix = f"Close Position ({reason})"
    logger.info(f"{Fore.YELLOW}{log_prefix}: Init for {symbol}...{Style.RESET_ALL}")
    live_position_data: Dict[str, Any]
    if position_to_close: logger.debug(f"[{func_name}] Using provided pos state."); live_position_data = position_to_close
    else: logger.debug(f"[{func_name}] Fetching current pos state..."); live_position_data = get_current_position_bybit_v5(exchange, symbol, config)
    live_side = live_position_data['side']; live_qty = live_position_data['qty']
    if live_side == config.POS_NONE or live_qty <= config.POSITION_QTY_EPSILON: logger.warning(f"{Fore.YELLOW}[{func_name}] No active position validated. Aborting close.{Style.RESET_ALL}"); return None
    close_order_side: Literal['buy', 'sell'] = config.SIDE_SELL if live_side == config.POS_LONG else config.SIDE_BUY
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: raise ValueError("Cannot determine category for close order.")
        qty_str = format_amount(exchange, symbol, live_qty); qty_float = float(qty_str); params: Dict[str, Any] = {'category': category, 'reduceOnly': True}
        bg = Back.YELLOW; fg = Fore.BLACK
        logger.warning(f"{bg}{fg}{Style.BRIGHT}[{func_name}] Attempting CLOSE {live_side} ({reason}): Exec {close_order_side.upper()} MARKET {qty_str} {symbol} (ReduceOnly)...{Style.RESET_ALL}")
        close_order = exchange.create_market_order(symbol=symbol, side=close_order_side, amount=qty_float, params=params)
        if not close_order: raise ValueError("create_market_order returned None unexpectedly.")
        fill_price = safe_decimal_conversion(close_order.get('average')); fill_qty = safe_decimal_conversion(close_order.get('filled', '0.0')); order_cost = safe_decimal_conversion(close_order.get('cost', '0.0')); order_id = format_order_id(close_order.get('id')); status = close_order.get('status', '?')
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}[{func_name}] Close Order ({reason}) submitted {symbol}. ID:...{order_id}, Status:{status}, Filled:{format_amount(exchange, symbol, fill_qty)}/{qty_str}, AvgFill:{format_price(exchange, symbol, fill_price)}, Cost:{order_cost:.4f}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] Closed {live_side} {qty_str} @ ~{format_price(exchange, symbol, fill_price)} ({reason}). ID:...{order_id}", config)
        return close_order
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e: logger.error(f"{Fore.RED}[{func_name}] Close Order Error ({reason}) for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] CLOSE FAIL ({live_side}): {type(e).__name__}", config); raise e
    except ccxt.ExchangeError as e:
        error_str = str(e).lower();
        if any(code in error_str for code in ["110025", "110045", "30086", "position is closed", "order would not reduce", "position size is zero", "qty is larger than position size"]):
            logger.warning(f"{Fore.YELLOW}[{func_name}] Close Order ({reason}): Exchange indicates already closed/zero or reduce fail: {e}. Assuming closed.{Style.RESET_ALL}"); return None
        else: logger.error(f"{Fore.RED}[{func_name}] Close Order ExchangeError ({reason}): {e}{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] CLOSE FAIL ({live_side}): ExchangeError", config); raise e
    except (ccxt.NetworkError, ValueError) as e: logger.error(f"{Fore.RED}[{func_name}] Close Order Network/Setup Error ({reason}): {e}{Style.RESET_ALL}"); raise e
    except Exception as e: logger.critical(f"{Back.RED}[{func_name}] Close Order Unexpected Error ({reason}): {e}{Style.RESET_ALL}", exc_info=True); send_sms_alert(f"[{market_base}] CLOSE FAIL ({live_side}): Unexpected Error", config); return None


# --- fetch_funding_rate ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=3, initial_delay=1.0)
def fetch_funding_rate(exchange: ccxt.bybit, symbol: str, config: 'Config') -> Optional[Dict[str, Any]]:
    """Fetches the current funding rate details for a perpetual swap symbol on Bybit V5."""
    func_name = "fetch_funding_rate"; logger.debug(f"[{func_name}] Fetching funding rate for {symbol}...")
    try:
        market = exchange.market(symbol)
        if not market.get('swap', False): logger.error(f"{Fore.RED}[{func_name}] Not a swap market: {symbol}.{Style.RESET_ALL}"); return None
        category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}[{func_name}] Invalid category '{category}' for funding rate ({symbol}).{Style.RESET_ALL}"); return None
        params = {'category': category}; logger.debug(f"[{func_name}] Calling fetch_funding_rate with params: {params}")
        funding_rate_info = exchange.fetch_funding_rate(symbol, params=params)
        processed_fr: Dict[str, Any] = { 'symbol': funding_rate_info.get('symbol'), 'fundingRate': safe_decimal_conversion(funding_rate_info.get('fundingRate')), 'fundingTimestamp': funding_rate_info.get('fundingTimestamp'), 'fundingDatetime': funding_rate_info.get('fundingDatetime'), 'markPrice': safe_decimal_conversion(funding_rate_info.get('markPrice')), 'indexPrice': safe_decimal_conversion(funding_rate_info.get('indexPrice')), 'nextFundingTime': funding_rate_info.get('nextFundingTimestamp'), 'nextFundingDatetime': None, 'info': funding_rate_info.get('info', {}) }
        if processed_fr['fundingRate'] is None: logger.warning(f"[{func_name}] Could not parse 'fundingRate' for {symbol}.")
        if processed_fr['nextFundingTime']:
            try: processed_fr['nextFundingDatetime'] = pd.to_datetime(processed_fr['nextFundingTime'], unit='ms', utc=True).strftime('%Y-%m-%d %H:%M:%S %Z')
            except Exception as dt_err: logger.warning(f"[{func_name}] Could not format next funding datetime: {dt_err}")
        rate = processed_fr.get('fundingRate'); next_dt_str = processed_fr.get('nextFundingDatetime', "N/A"); rate_str = f"{rate:.6%}" if rate is not None else "N/A"
        logger.info(f"[{func_name}] Funding Rate {symbol}: {rate_str}. Next: {next_dt_str}")
        return processed_fr
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching funding rate: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching funding rate: {e}{Style.RESET_ALL}", exc_info=True); return None

# --- set_position_mode_bybit_v5 ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=2, initial_delay=1.0)
def set_position_mode_bybit_v5(exchange: ccxt.bybit, symbol_or_category: str, mode: Literal['one-way', 'hedge'], config: 'Config') -> bool:
    """Sets the position mode (One-Way or Hedge) for a specific category (Linear/Inverse) on Bybit V5."""
    func_name = "set_position_mode_bybit_v5"; logger.info(f"{Fore.CYAN}[{func_name}] Setting mode '{mode}' for category of '{symbol_or_category}'...{Style.RESET_ALL}")
    mode_map = {'one-way': '0', 'hedge': '3'}; target_mode_code = mode_map.get(mode.lower())
    if target_mode_code is None: logger.error(f"{Fore.RED}[{func_name}] Invalid mode '{mode}'.{Style.RESET_ALL}"); return False
    target_category: Optional[Literal['linear', 'inverse']] = None
    if symbol_or_category.lower() in ['linear', 'inverse']: target_category = symbol_or_category.lower() # type: ignore
    else:
        try: market = exchange.market(symbol_or_category); target_category = _get_v5_category(market);
             if target_category not in ['linear', 'inverse']: target_category = None
        except Exception as e: logger.warning(f"[{func_name}] Could not get market/category for '{symbol_or_category}': {e}"); target_category = None
    if not target_category: logger.error(f"{Fore.RED}[{func_name}] Could not determine contract category from '{symbol_or_category}'.{Style.RESET_ALL}"); return False
    logger.debug(f"[{func_name}] Target Category: {target_category}, Mode Code: {target_mode_code} ('{mode}')")
    try:
        if not hasattr(exchange, 'private_post_v5_position_switch_mode'): logger.error(f"{Fore.RED}[{func_name}] CCXT lacks 'private_post_v5_position_switch_mode'.{Style.RESET_ALL}"); return False
        params = {'category': target_category, 'mode': target_mode_code}; logger.debug(f"[{func_name}] Calling private V5 endpoint with params: {params}")
        response = exchange.private_post_v5_position_switch_mode(params); logger.debug(f"[{func_name}] Raw V5 endpoint response: {response}")
        ret_code = response.get('retCode'); ret_msg = response.get('retMsg', '').lower()
        if ret_code == 0: logger.success(f"{Fore.GREEN}[{func_name}] Mode set '{mode}' for {target_category} via V5 endpoint.{Style.RESET_ALL}"); return True
        elif ret_code in [110021, 34036] or "not modified" in ret_msg: logger.info(f"{Fore.CYAN}[{func_name}] Mode already '{mode}' for {target_category}.{Style.RESET_ALL}"); return True
        elif ret_code == 110020 or "have position" in ret_msg or "active order" in ret_msg: logger.error(f"{Fore.RED}[{func_name}] Cannot switch mode: Active pos/orders exist. Msg: {response.get('retMsg')}{Style.RESET_ALL}"); return False
        else: raise ccxt.ExchangeError(f"Bybit API error setting mode: Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'")
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.BadSymbol) as e:
        if not (isinstance(e, ccxt.ExchangeError) and "110020" in str(e)): logger.warning(f"{Fore.YELLOW}[{func_name}] API Error setting mode: {e}{Style.RESET_ALL}")
        if isinstance(e, (ccxt.NetworkError, ccxt.AuthenticationError)): raise e
        return False
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error setting mode: {e}{Style.RESET_ALL}", exc_info=True); return False

# --- fetch_l2_order_book_validated ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=2, initial_delay=0.5)
def fetch_l2_order_book_validated(
    exchange: ccxt.bybit, symbol: str, limit: int, config: 'Config'
) -> Optional[Dict[str, List[Tuple[Decimal, Decimal]]]]:
    """Fetches the Level 2 order book for a symbol using Bybit V5 fetchOrderBook and validates the data."""
    func_name = "fetch_l2_order_book_validated"; logger.debug(f"[{func_name}] Fetching L2 OB {symbol} (Limit:{limit})...")
    if not exchange.has.get('fetchOrderBook'): logger.error(f"{Fore.RED}[{func_name}] fetchOrderBook not supported.{Style.RESET_ALL}"); return None
    try:
        market = exchange.market(symbol); category = _get_v5_category(market); params = {'category': category} if category else {}
        max_limit_map = {'spot': 50, 'linear': 200, 'inverse': 200, 'option': 25}; max_limit = max_limit_map.get(category, 50) if category else 50
        if limit > max_limit: logger.warning(f"[{func_name}] Clamping limit {limit} to {max_limit} for category '{category}'."); limit = max_limit
        logger.debug(f"[{func_name}] Calling fetchOrderBook with limit={limit}, params={params}")
        order_book = exchange.fetch_order_book(symbol, limit=limit, params=params)
        if not isinstance(order_book, dict) or 'bids' not in order_book or 'asks' not in order_book: raise ValueError("Invalid OB structure")
        raw_bids=order_book['bids']; raw_asks=order_book['asks']
        if not isinstance(raw_bids, list) or not isinstance(raw_asks, list): raise ValueError("Bids/Asks not lists")
        validated_bids: List[Tuple[Decimal, Decimal]] = []; validated_asks: List[Tuple[Decimal, Decimal]] = []; conversion_errors = 0
        for p_str, a_str in raw_bids: p = safe_decimal_conversion(p_str); a = safe_decimal_conversion(a_str); if not (p and a and p > 0 and a >= 0): conversion_errors += 1; continue; validated_bids.append((p, a))
        for p_str, a_str in raw_asks: p = safe_decimal_conversion(p_str); a = safe_decimal_conversion(a_str); if not (p and a and p > 0 and a >= 0): conversion_errors += 1; continue; validated_asks.append((p, a))
        if conversion_errors > 0: logger.warning(f"[{func_name}] Skipped {conversion_errors} invalid OB entries for {symbol}.")
        if not validated_bids or not validated_asks: logger.warning(f"[{func_name}] Empty validated bids/asks for {symbol}."); return {'bids': [], 'asks': []}
        if validated_bids[0][0] >= validated_asks[0][0]: logger.error(f"{Fore.RED}[{func_name}] OB crossed: Bid ({validated_bids[0][0]}) >= Ask ({validated_asks[0][0]}) for {symbol}.{Style.RESET_ALL}"); return {'bids': validated_bids, 'asks': validated_asks} # Return crossed OB
        logger.debug(f"[{func_name}] Processed L2 OB {symbol}. Bids:{len(validated_bids)}, Asks:{len(validated_asks)}")
        return {'bids': validated_bids, 'asks': validated_asks}
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol, ValueError) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API/Validation Error fetching L2 OB for {symbol}: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching L2 OB for {symbol}: {e}{Style.RESET_ALL}", exc_info=True); return None

# --- place_native_stop_loss ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=1, initial_delay=0)
def place_native_stop_loss(
    exchange: ccxt.bybit, symbol: str, side: Literal['buy', 'sell'], amount: Decimal, stop_price: Decimal, config: 'Config',
    trigger_by: Literal['LastPrice', 'MarkPrice', 'IndexPrice'] = 'MarkPrice', client_order_id: Optional[str] = None, position_idx: Literal[0, 1, 2] = 0
) -> Optional[Dict]:
    """Places a native Stop Market order on Bybit V5, intended as a Stop Loss (reduceOnly)."""
    func_name = "place_native_stop_loss"; log_prefix = f"Place Native SL ({side.upper()})"
    logger.info(f"{Fore.CYAN}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}, Trigger @ {format_price(exchange, symbol, stop_price)} ({trigger_by}), PosIdx:{position_idx}...{Style.RESET_ALL}")
    if amount <= config.POSITION_QTY_EPSILON or stop_price <= Decimal("0"): logger.error(f"{Fore.RED}{log_prefix}: Invalid amount/stop price.{Style.RESET_ALL}"); return None
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}[{func_name}] Not a contract symbol: {symbol}.{Style.RESET_ALL}"); return None
        amount_str = format_amount(exchange, symbol, amount); amount_float = float(amount_str); stop_price_str = format_price(exchange, symbol, stop_price); stop_price_float = float(stop_price_str)
        params: Dict[str, Any] = {
            'category': category, 'stopLoss': stop_price_str, 'slTriggerBy': trigger_by,
            'reduceOnly': True, 'positionIdx': position_idx, 'tpslMode': 'Full', 'slOrderType': 'Market'
        }
        if client_order_id:
            max_coid_len = 36; original_len = len(client_order_id); valid_coid = client_order_id[:max_coid_len]
            params['clientOrderId'] = valid_coid
            if len(valid_coid) < original_len: logger.warning(f"[{func_name}] Client OID truncated: '{valid_coid}' (Orig len: {original_len})")
        bg = Back.YELLOW; fg = Fore.BLACK
        logger.warning(f"{bg}{fg}{Style.BRIGHT}{log_prefix}: Placing NATIVE Stop Loss (Market exec) -> Qty:{amount_float}, Side:{side}, TriggerPx:{stop_price_str}, TriggerBy:{trigger_by}, Reduce:True, PosIdx:{position_idx}, Params:{params}{Style.RESET_ALL}")
        sl_order = exchange.create_order(symbol=symbol, type='market', side=side, amount=amount_float, params=params)
        order_id = sl_order.get('id'); client_oid_resp = sl_order.get('clientOrderId', params.get('clientOrderId', 'N/A')); status = sl_order.get('status', '?')
        returned_stop_price = safe_decimal_conversion(sl_order.get('info', {}).get('stopLoss', sl_order.get('stopPrice')), None)
        returned_trigger = sl_order.get('info', {}).get('slTriggerBy', trigger_by)
        logger.success(f"{Fore.GREEN}{log_prefix}: Native SL order placed. ID:...{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, Trigger:{format_price(exchange, symbol, returned_stop_price)} (by {returned_trigger}){Style.RESET_ALL}")
        return sl_order
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadSymbol) as e: logger.error(f"{Fore.RED}{log_prefix}: API Error placing SL: {type(e).__name__} - {e}{Style.RESET_ALL}"); return None
    except Exception as e: logger.critical(f"{Back.RED}[{func_name}] Unexpected error placing SL: {e}{Style.RESET_ALL}", exc_info=True); send_sms_alert(f"[{symbol.split('/')[0]}] SL PLACE FAIL ({side.upper()}): Unexpected {type(e).__name__}", config); return None


# --- fetch_open_orders_filtered ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=2, initial_delay=1.0)
def fetch_open_orders_filtered(
    exchange: ccxt.bybit, symbol: str, config: 'Config', side: Optional[Literal['buy', 'sell']] = None,
    order_type: Optional[str] = None, order_filter: Optional[Literal['Order', 'StopOrder', 'tpslOrder']] = None
) -> Optional[List[Dict]]:
    """Fetches open orders for a specific symbol on Bybit V5, with optional filtering."""
    func_name = "fetch_open_orders_filtered"; filter_log = f"(Side:{side or 'Any'}, Type:{order_type or 'Any'}, V5Filter:{order_filter or 'Default'})"
    logger.debug(f"[{func_name}] Fetching open orders {symbol} {filter_log}...")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}[{func_name}] Cannot determine category for {symbol}.{Style.RESET_ALL}"); return None
        params: Dict[str, Any] = {'category': category}
        if order_filter: params['orderFilter'] = order_filter
        elif order_type:
            norm_type = order_type.lower().replace('_', '').replace('-', '')
            if any(k in norm_type for k in ['stop', 'trigger', 'take', 'tpsl']): params['orderFilter'] = 'StopOrder'
            else: params['orderFilter'] = 'Order'
        else: params['orderFilter'] = 'Order' # Default
        logger.debug(f"[{func_name}] Calling fetch_open_orders with params: {params}")
        open_orders = exchange.fetch_open_orders(symbol=symbol, params=params)
        if not open_orders: logger.debug(f"[{func_name}] No open orders found matching {params}."); return []
        filtered = open_orders; initial_count = len(filtered)
        if side: side_lower = side.lower(); filtered = [o for o in filtered if o.get('side', '').lower() == side_lower]; logger.debug(f"[{func_name}] Filtered by side='{side}'. Count: {initial_count} -> {len(filtered)}.")
        if order_type:
            norm_type_filter = order_type.lower().replace('_', '').replace('-', ''); count_before = len(filtered)
            def check_type(o): o_type = o.get('type', '').lower().replace('_', '').replace('-', ''); info = o.get('info', {}); return o_type == norm_type_filter or info.get('orderType', '').lower() == norm_type_filter
            filtered = [o for o in filtered if check_type(o)]; logger.debug(f"[{func_name}] Filtered by type='{order_type}'. Count: {count_before} -> {len(filtered)}.")
        logger.info(f"[{func_name}] Fetched/filtered {len(filtered)} open orders for {symbol} {filter_log}.")
        return filtered
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching open orders: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching open orders: {e}{Style.RESET_ALL}", exc_info=True); return None

# --- calculate_margin_requirement ---
# ... (Function is identical to v2.8) ...
def calculate_margin_requirement(
    exchange: ccxt.bybit, symbol: str, amount: Decimal, price: Decimal, leverage: Decimal, config: 'Config',
    order_side: Literal['buy', 'sell'], is_maker: bool = False
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Calculates the estimated Initial Margin (IM) requirement for placing an order on Bybit V5."""
    func_name = "calculate_margin_requirement"; logger.debug(f"[{func_name}] Calc margin: {order_side} {format_amount(exchange, symbol, amount)} @ {format_price(exchange, symbol, price)}, Lev:{leverage}x, Maker:{is_maker}")
    if amount <= 0 or price <= 0 or leverage <= 0: logger.error(f"{Fore.RED}[{func_name}] Invalid inputs.{Style.RESET_ALL}"); return None, None
    try:
        market = exchange.market(symbol); quote_currency = market.get('quote', config.USDT_SYMBOL)
        if not market.get('contract'): logger.error(f"{Fore.RED}[{func_name}] Not a contract symbol: {symbol}.{Style.RESET_ALL}"); return None, None
        position_value = amount * price; logger.debug(f"[{func_name}] Est Order Value: {format_price(exchange, quote_currency, position_value)} {quote_currency}")
        if leverage == Decimal("0"): raise DivisionByZero("Leverage cannot be zero.")
        initial_margin_base = position_value / leverage; logger.debug(f"[{func_name}] Base IM: {format_price(exchange, quote_currency, initial_margin_base)} {quote_currency}")
        fee_rate = config.MAKER_FEE_RATE if is_maker else config.TAKER_FEE_RATE; estimated_fee = position_value * fee_rate; logger.debug(f"[{func_name}] Est Fee ({fee_rate:.4%}): {format_price(exchange, quote_currency, estimated_fee)} {quote_currency}")
        total_initial_margin_estimate = initial_margin_base + estimated_fee
        logger.info(f"[{func_name}] Est TOTAL Initial Margin Req (incl. fee): {format_price(exchange, quote_currency, total_initial_margin_estimate)} {quote_currency}")
        maintenance_margin_estimate: Optional[Decimal] = None
        try:
            mmr_keys = ['maintenanceMarginRate', 'mmr']; mmr_rate_str = None
            for key in mmr_keys: value_from_info = market.get('info', {}).get(key); value_from_root = market.get(key); mmr_rate_str = value_from_info or value_from_root; if mmr_rate_str: break
            if mmr_rate_str: mmr_rate = safe_decimal_conversion(mmr_rate_str);
                 if mmr_rate and mmr_rate > 0: maintenance_margin_estimate = position_value * mmr_rate; logger.debug(f"[{func_name}] Basic MM Estimate (Base MMR {mmr_rate:.4%}): {format_price(exchange, quote_currency, maintenance_margin_estimate)} {quote_currency}")
                 else: logger.debug(f"[{func_name}] Could not parse MMR rate '{mmr_rate_str}'.")
            else: logger.debug(f"[{func_name}] MMR key not found in market info.")
        except Exception as mm_err: logger.warning(f"[{func_name}] Could not estimate MM: {mm_err}")
        return total_initial_margin_estimate, maintenance_margin_estimate
    except (DivisionByZero, KeyError, ValueError) as e: logger.error(f"{Fore.RED}[{func_name}] Calculation error: {e}{Style.RESET_ALL}"); return None, None
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error during margin calculation: {e}{Style.RESET_ALL}", exc_info=True); return None, None


# --- fetch_ticker_validated (Fixed Timestamp/Age Logic) ---
@retry_api_call(max_retries=2, initial_delay=0.5)
def fetch_ticker_validated(
    exchange: ccxt.bybit, symbol: str, config: 'Config', max_age_seconds: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Fetches the ticker for a symbol from Bybit V5, validates its freshness and key values.
    Returns a dictionary with Decimal values, or None if validation fails or API error occurs.
    """
    func_name = "fetch_ticker_validated"; logger.debug(f"[{func_name}] Fetching/Validating ticker {symbol}...")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        params = {'category': category} if category else {}
        logger.debug(f"[{func_name}] Calling fetch_ticker with params: {params}")
        ticker = exchange.fetch_ticker(symbol, params=params)

        # --- Validation ---
        timestamp_ms = ticker.get('timestamp')
        if timestamp_ms is None:
            # Raise specific error if timestamp is missing BEFORE calculating age
            raise ValueError("Ticker data is missing timestamp.")

        current_time_ms = time.time() * 1000
        age_seconds = (current_time_ms - timestamp_ms) / 1000.0

        if age_seconds > max_age_seconds:
            # Use age_seconds in the error message now that it's valid
            raise ValueError(f"Ticker data is stale (Age: {age_seconds:.1f}s > Max: {max_age_seconds}s).")
        if age_seconds < -5: # Check for future timestamp
             raise ValueError(f"Ticker timestamp ({timestamp_ms}) seems to be in the future (Age: {age_seconds:.1f}s).")

        last_price = safe_decimal_conversion(ticker.get('last')); bid_price = safe_decimal_conversion(ticker.get('bid')); ask_price = safe_decimal_conversion(ticker.get('ask'))
        if last_price is None or last_price <= 0: raise ValueError(f"Invalid 'last' price: {ticker.get('last')}")
        if bid_price is None or bid_price <= 0: logger.warning(f"[{func_name}] Invalid/missing 'bid': {ticker.get('bid')}")
        if ask_price is None or ask_price <= 0: logger.warning(f"[{func_name}] Invalid/missing 'ask': {ticker.get('ask')}")

        spread, spread_pct = None, None
        if bid_price and ask_price:
             if bid_price >= ask_price: raise ValueError(f"Bid ({bid_price}) >= Ask ({ask_price})")
             spread = ask_price - bid_price; spread_pct = (spread / best_bid) * 100 if best_bid > 0 else Decimal("inf") # Changed to best_bid
        else: logger.warning(f"[{func_name}] Cannot calculate spread due to missing bid/ask.")

        base_volume = safe_decimal_conversion(ticker.get('baseVolume')); quote_volume = safe_decimal_conversion(ticker.get('quoteVolume'))
        if base_volume is not None and base_volume < 0: logger.warning(f"Negative baseVol: {base_volume}"); base_volume = Decimal("0.0")
        if quote_volume is not None and quote_volume < 0: logger.warning(f"Negative quoteVol: {quote_volume}"); quote_volume = Decimal("0.0")

        validated_ticker = { 'symbol': ticker.get('symbol', symbol), 'timestamp': timestamp_ms, 'datetime': ticker.get('datetime'), 'last': last_price, 'bid': bid_price, 'ask': ask_price, 'bidVolume': safe_decimal_conversion(ticker.get('bidVolume')), 'askVolume': safe_decimal_conversion(ticker.get('askVolume')), 'baseVolume': base_volume, 'quoteVolume': quote_volume, 'high': safe_decimal_conversion(ticker.get('high')), 'low': safe_decimal_conversion(ticker.get('low')), 'open': safe_decimal_conversion(ticker.get('open')), 'close': last_price, 'change': safe_decimal_conversion(ticker.get('change')), 'percentage': safe_decimal_conversion(ticker.get('percentage')), 'average': safe_decimal_conversion(ticker.get('average')), 'spread': spread, 'spread_pct': spread_pct, 'info': ticker.get('info', {}) }
        logger.debug(f"[{func_name}] Ticker OK: {symbol} Last={format_price(exchange, symbol, last_price)}, Spread={(spread_pct or Decimal('NaN')):.4f}% (Age:{age_seconds:.1f}s)")
        return validated_ticker

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        # API/Network errors should be re-raised for the decorator
        logger.warning(f"{Fore.YELLOW}[{func_name}] API/Symbol error fetching ticker for {symbol}: {type(e).__name__} - {e}{Style.RESET_ALL}")
        raise e
    except ValueError as e:
        # Validation errors (stale, bad price, missing timestamp) are logged and return None
        logger.warning(f"{Fore.YELLOW}[{func_name}] Ticker validation failed for {symbol}: {e}{Style.RESET_ALL}")
        return None
    except Exception as e:
        # Unexpected errors return None
        logger.error(f"{Fore.RED}[{func_name}] Unexpected ticker error for {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
        return None

# --- place_native_trailing_stop ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=1, initial_delay=0)
def place_native_trailing_stop(
    exchange: ccxt.bybit, symbol: str, side: Literal['buy', 'sell'], amount: Decimal, trailing_offset: Union[Decimal, str], config: 'Config',
    activation_price: Optional[Decimal] = None, trigger_by: Literal['LastPrice', 'MarkPrice', 'IndexPrice'] = 'MarkPrice',
    client_order_id: Optional[str] = None, position_idx: Literal[0, 1, 2] = 0
) -> Optional[Dict]:
    """Places a native Trailing Stop Market order on Bybit V5 (reduceOnly)."""
    func_name = "place_native_trailing_stop"; log_prefix = f"Place Native TSL ({side.upper()})"
    params: Dict[str, Any] = {}; trail_log_str = ""; is_percent_trail = False;
    try:
        if isinstance(trailing_offset, str) and trailing_offset.endswith('%'):
            percent_val = Decimal(trailing_offset.rstrip('%'))
            if not (Decimal("0.1") <= percent_val <= Decimal("10.0")): raise ValueError(f"% out of range (0.1-10): {percent_val}")
            params['trailingStop'] = str(percent_val.quantize(Decimal("0.01"))); trail_log_str = f"{percent_val}%"; is_percent_trail = True
        elif isinstance(trailing_offset, Decimal):
            if trailing_offset <= Decimal("0"): raise ValueError(f"Delta must be positive: {trailing_offset}")
            delta_str = format_price(exchange, symbol, trailing_offset); params['trailingMove'] = delta_str; trail_log_str = f"{delta_str} (abs)"
        else: raise TypeError(f"Invalid trailing_offset type: {type(trailing_offset)}")
        if activation_price is not None and activation_price <= Decimal("0"): raise ValueError("Activation price must be positive.")
        logger.info(f"{Fore.CYAN}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}, Trail:{trail_log_str}, ActPx:{format_price(exchange, symbol, activation_price) or 'Immediate'}, Trigger:{trigger_by}, PosIdx:{position_idx}{Style.RESET_ALL}")
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}[{func_name}] Not a contract symbol: {symbol}.{Style.RESET_ALL}"); return None
        amount_str = format_amount(exchange, symbol, amount); amount_float = float(amount_str); activation_price_str = format_price(exchange, symbol, activation_price) if activation_price else None
        params.update({'category': category, 'reduceOnly': True, 'positionIdx': position_idx, 'tpslMode': 'Full', 'triggerBy': trigger_by, 'tsOrderType': 'Market'})
        if activation_price_str is not None: params['activePrice'] = activation_price_str
        if client_order_id:
            max_coid_len = 36; original_len = len(client_order_id); valid_coid = client_order_id[:max_coid_len]
            params['clientOrderId'] = valid_coid
            if len(valid_coid) < original_len: logger.warning(f"[{func_name}] Client OID truncated: '{valid_coid}' (Orig len: {original_len})")
        bg = Back.YELLOW; fg = Fore.BLACK
        logger.warning(f"{bg}{fg}{Style.BRIGHT}{log_prefix}: Placing NATIVE TSL (Market exec) -> Qty:{amount_float}, Side:{side}, Trail:{trail_log_str}, ActPx:{activation_price_str or 'Immediate'}, Trigger:{trigger_by}, Reduce:True, PosIdx:{position_idx}, Params:{params}{Style.RESET_ALL}")
        tsl_order = exchange.create_order(symbol=symbol, type='market', side=side, amount=amount_float, params=params)
        order_id = tsl_order.get('id'); client_oid_resp = tsl_order.get('clientOrderId', params.get('clientOrderId', 'N/A')); status = tsl_order.get('status', '?')
        returned_trail = tsl_order.get('info', {}).get('trailingStop') or tsl_order.get('info', {}).get('trailingMove'); returned_act = safe_decimal_conversion(tsl_order.get('info', {}).get('activePrice')); returned_trigger = tsl_order.get('info', {}).get('triggerBy', trigger_by)
        logger.success(f"{Fore.GREEN}{log_prefix}: Native TSL order placed. ID:...{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, Trail:{returned_trail}, ActPx:{format_price(exchange, symbol, returned_act)}, TriggerBy:{returned_trigger}{Style.RESET_ALL}")
        return tsl_order
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadSymbol, ValueError, TypeError) as e: logger.error(f"{Fore.RED}{log_prefix}: API/Input Error placing TSL: {type(e).__name__} - {e}{Style.RESET_ALL}"); return None
    except Exception as e: logger.critical(f"{Back.RED}[{func_name}] Unexpected error placing TSL: {e}{Style.RESET_ALL}", exc_info=True); send_sms_alert(f"[{symbol.split('/')[0]}] TSL PLACE FAIL ({side.upper()}): Unexpected {type(e).__name__}", config); return None


# --- fetch_account_info_bybit_v5 ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=2, initial_delay=1.0)
def fetch_account_info_bybit_v5(exchange: ccxt.bybit, config: 'Config') -> Optional[Dict[str, Any]]:
    """Fetches general account information from Bybit V5 API (`/v5/account/info`)."""
    func_name = "fetch_account_info_bybit_v5"; logger.debug(f"[{func_name}] Fetching Bybit V5 account info...")
    try:
        if hasattr(exchange, 'private_get_v5_account_info'):
            logger.debug(f"[{func_name}] Using private_get_v5_account_info endpoint.")
            account_info_raw = exchange.private_get_v5_account_info(); logger.debug(f"[{func_name}] Raw Account Info response: {str(account_info_raw)[:400]}...")
            ret_code = account_info_raw.get('retCode'); ret_msg = account_info_raw.get('retMsg')
            if ret_code == 0 and 'result' in account_info_raw:
                result = account_info_raw['result']
                parsed_info = { 'unifiedMarginStatus': result.get('unifiedMarginStatus'), 'marginMode': result.get('marginMode'), 'dcpStatus': result.get('dcpStatus'), 'timeWindow': result.get('timeWindow'), 'smtCode': result.get('smtCode'), 'isMasterTrader': result.get('isMasterTrader'), 'updateTime': result.get('updateTime'), 'rawInfo': result }
                logger.info(f"[{func_name}] Account Info: UTA Status={parsed_info.get('unifiedMarginStatus', 'N/A')}, MarginMode={parsed_info.get('marginMode', 'N/A')}, DCP Status={parsed_info.get('dcpStatus', 'N/A')}")
                return parsed_info
            else: raise ccxt.ExchangeError(f"Failed fetch/parse account info. Code={ret_code}, Msg='{ret_msg}'")
        else:
            logger.warning(f"[{func_name}] CCXT lacks 'private_get_v5_account_info'. Using fallback fetch_accounts() (less detail).")
            accounts = exchange.fetch_accounts();
            if accounts: logger.info(f"[{func_name}] Fallback fetch_accounts(): {str(accounts[0])[:200]}..."); return accounts[0]
            else: logger.error(f"{Fore.RED}[{func_name}] Fallback fetch_accounts() returned no data.{Style.RESET_ALL}"); return None
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching account info: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching account info: {e}{Style.RESET_ALL}", exc_info=True); return None


# --- validate_market ---
# ... (Function is identical to v2.8) ...
def validate_market(
    exchange: ccxt.bybit, symbol: str, config: 'Config', expected_type: Optional[Literal['swap', 'future', 'spot', 'option']] = None,
    expected_logic: Optional[Literal['linear', 'inverse']] = None, check_active: bool = True, require_contract: bool = True
) -> Optional[Dict]:
    """Validates if a symbol exists on the exchange, is active, and optionally matches expectations."""
    func_name = "validate_market"; eff_expected_type = expected_type if expected_type is not None else config.EXPECTED_MARKET_TYPE; eff_expected_logic = expected_logic if expected_logic is not None else config.EXPECTED_MARKET_LOGIC
    logger.debug(f"[{func_name}] Validating '{symbol}'. Checks: Type='{eff_expected_type or 'Any'}', Logic='{eff_expected_logic or 'Any'}', Active={check_active}, Contract={require_contract}")
    try:
        if not exchange.markets: logger.info(f"[{func_name}] Loading markets..."); exchange.load_markets(reload=True)
        if not exchange.markets: logger.error(f"{Fore.RED}[{func_name}] Failed to load markets.{Style.RESET_ALL}"); return None
        market = exchange.market(symbol); is_active = market.get('active', False)
        if check_active and not is_active: logger.warning(f"{Fore.YELLOW}[{func_name}] Validation Warning: '{symbol}' inactive.{Style.RESET_ALL}")
        actual_type = market.get('type');
        if eff_expected_type and actual_type != eff_expected_type: logger.error(f"{Fore.RED}[{func_name}] Validation Failed: '{symbol}' type mismatch. Expected '{eff_expected_type}', Got '{actual_type}'.{Style.RESET_ALL}"); return None
        is_contract = market.get('contract', False);
        if require_contract and not is_contract: logger.error(f"{Fore.RED}[{func_name}] Validation Failed: '{symbol}' not a contract, but required.{Style.RESET_ALL}"); return None
        actual_logic_str: Optional[str] = None
        if is_contract:
            actual_logic_str = _get_v5_category(market);
            if eff_expected_logic and actual_logic_str != eff_expected_logic: logger.error(f"{Fore.RED}[{func_name}] Validation Failed: '{symbol}' logic mismatch. Expected '{eff_expected_logic}', Got '{actual_logic_str}'.{Style.RESET_ALL}"); return None
        logger.info(f"{Fore.GREEN}[{func_name}] Market OK: '{symbol}' (Type:{actual_type}, Logic:{actual_logic_str or 'N/A'}, Active:{is_active}).{Style.RESET_ALL}"); return market
    except ccxt.BadSymbol as e: logger.error(f"{Fore.RED}[{func_name}] Validation Failed: Symbol '{symbol}' not found. Error: {e}{Style.RESET_ALL}"); return None
    except ccxt.NetworkError as e: logger.error(f"{Fore.RED}[{func_name}] Network error during market validation for '{symbol}': {e}{Style.RESET_ALL}"); return None
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error validating '{symbol}': {e}{Style.RESET_ALL}", exc_info=True); return None


# --- fetch_recent_trades ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=2, initial_delay=0.5)
def fetch_recent_trades(
    exchange: ccxt.bybit, symbol: str, config: 'Config', limit: int = 100, min_size_filter: Optional[Decimal] = None
) -> Optional[List[Dict]]:
    """Fetches recent public trades for a symbol from Bybit V5, validates data."""
    func_name = "fetch_recent_trades"; filter_log = f"(MinSize:{format_amount(exchange, symbol, min_size_filter) if min_size_filter else 'N/A'})"
    logger.debug(f"[{func_name}] Fetching {limit} trades for {symbol} {filter_log}...")
    if limit > 1000: logger.warning(f"[{func_name}] Clamping limit {limit} to 1000."); limit = 1000
    if limit <= 0: logger.warning(f"[{func_name}] Invalid limit {limit}. Using 100."); limit = 100
    try:
        market = exchange.market(symbol); category = _get_v5_category(market); params = {'category': category} if category else {}
        logger.debug(f"[{func_name}] Calling fetch_trades with limit={limit}, params={params}")
        trades_raw = exchange.fetch_trades(symbol, limit=limit, params=params)
        if not trades_raw: logger.debug(f"[{func_name}] No recent trades found."); return []
        processed_trades: List[Dict] = []; conversion_errors = 0; filtered_out_count = 0
        for trade in trades_raw:
            try:
                amount = safe_decimal_conversion(trade.get('amount')); price = safe_decimal_conversion(trade.get('price'))
                if not all([trade.get('id'), trade.get('timestamp'), trade.get('side'), price, amount]) or price <= 0 or amount <= 0: conversion_errors += 1; continue
                if min_size_filter is not None and amount < min_size_filter: filtered_out_count += 1; continue
                cost = safe_decimal_conversion(trade.get('cost'))
                if cost is None or (price and amount and abs(cost - (price * amount)) > config.POSITION_QTY_EPSILON * price): cost = price * amount if price and amount else None
                processed_trades.append({'id': trade.get('id'), 'timestamp': trade.get('timestamp'), 'datetime': trade.get('datetime'), 'symbol': trade.get('symbol', symbol), 'side': trade.get('side'), 'price': price, 'amount': amount, 'cost': cost, 'takerOrMaker': trade.get('takerOrMaker'), 'info': trade.get('info', {})})
            except Exception as proc_err: conversion_errors += 1; logger.warning(f"{Fore.YELLOW}Error processing single trade: {proc_err}. Data: {trade}{Style.RESET_ALL}")
        if conversion_errors > 0: logger.warning(f"{Fore.YELLOW}Skipped {conversion_errors} trades due to processing errors.{Style.RESET_ALL}")
        if filtered_out_count > 0: logger.debug(f"[{func_name}] Filtered {filtered_out_count} trades smaller than {min_size_filter}.")
        processed_trades.sort(key=lambda x: x['timestamp'], reverse=True)
        logger.info(f"[{func_name}] Fetched/processed {len(processed_trades)} trades for {symbol} {filter_log}.")
        return processed_trades
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching trades: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching trades: {e}{Style.RESET_ALL}", exc_info=True); return None

# --- update_limit_order ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=1, initial_delay=0)
def update_limit_order(
    exchange: ccxt.bybit, symbol: str, order_id: str, config: 'Config', new_amount: Optional[Decimal] = None,
    new_price: Optional[Decimal] = None, new_client_order_id: Optional[str] = None
) -> Optional[Dict]:
    """Attempts to modify the amount and/or price of an existing open limit order on Bybit V5."""
    func_name = "update_limit_order"; log_prefix = f"Update Order ...{format_order_id(order_id)}"
    if new_amount is None and new_price is None: logger.warning(f"[{func_name}] {log_prefix}: No new amount or price provided."); return None
    if new_amount is not None and new_amount <= config.POSITION_QTY_EPSILON: logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: Invalid new amount ({new_amount})."); return None
    if new_price is not None and new_price <= Decimal("0"): logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: Invalid new price ({new_price})."); return None
    logger.info(f"{Fore.CYAN}{log_prefix}: Update {symbol} (Amt:{format_amount(exchange,symbol,new_amount) or 'NC'}, Px:{format_price(exchange,symbol,new_price) or 'NC'})...{Style.RESET_ALL}")
    try:
        if not exchange.has.get('editOrder'): logger.error(f"{Fore.RED}{log_prefix}: editOrder not supported.{Style.RESET_ALL}"); return None
        logger.debug(f"[{func_name}] Fetching current order state..."); market = exchange.market(symbol); category = _get_v5_category(market);
        if not category: raise ValueError(f"Cannot determine category for {symbol}")
        fetch_params = {'category': category}
        current_order = exchange.fetch_order(order_id, symbol, params=fetch_params)
        status = current_order.get('status'); order_type = current_order.get('type'); filled_qty = safe_decimal_conversion(current_order.get('filled', '0.0'))
        if status != 'open': raise ccxt.InvalidOrder(f"{log_prefix}: Status is '{status}' (not 'open').")
        if order_type != 'limit': raise ccxt.InvalidOrder(f"{log_prefix}: Type is '{order_type}' (not 'limit').")
        allow_partial_fill_update = False;
        if not allow_partial_fill_update and filled_qty > config.POSITION_QTY_EPSILON: logger.warning(f"{Fore.YELLOW}[{func_name}] Update aborted: partially filled ({format_amount(exchange, symbol, filled_qty)}).{Style.RESET_ALL}"); return None
        final_amount_dec = new_amount if new_amount is not None else safe_decimal_conversion(current_order.get('amount')); final_price_dec = new_price if new_price is not None else safe_decimal_conversion(current_order.get('price'))
        if final_amount_dec is None or final_price_dec is None or final_amount_dec <= config.POSITION_QTY_EPSILON or final_price_dec <= 0: raise ValueError("Invalid final amount/price.")
        edit_params: Dict[str, Any] = {'category': category}
        if new_client_order_id:
            max_coid_len = 36; original_len = len(new_client_order_id); valid_coid = new_client_order_id[:max_coid_len]
            edit_params['clientOrderId'] = valid_coid
            if len(valid_coid) < original_len: logger.warning(f"[{func_name}] New Client OID truncated: '{valid_coid}' (Orig len: {original_len})")
        final_amount_float = float(format_amount(exchange, symbol, final_amount_dec)); final_price_float = float(format_price(exchange, symbol, final_price_dec))
        logger.info(f"{Fore.CYAN}[{func_name}] Submitting update -> Amt:{final_amount_float}, Px:{final_price_float}, Side:{current_order['side']}, Params:{edit_params}{Style.RESET_ALL}")
        updated_order = exchange.edit_order(id=order_id, symbol=symbol, type='limit', side=current_order['side'], amount=final_amount_float, price=final_price_float, params=edit_params)
        if updated_order: new_id = updated_order.get('id', order_id); status_after = updated_order.get('status', '?'); new_client_oid_resp = updated_order.get('clientOrderId', edit_params.get('clientOrderId', 'N/A')); logger.success(f"{Fore.GREEN}[{func_name}] Update OK. NewID:...{format_order_id(new_id)}, Status:{status_after}, ClientOID:{new_client_oid_resp}{Style.RESET_ALL}"); return updated_order
        else: logger.warning(f"{Fore.YELLOW}[{func_name}] edit_order returned no data. Check status manually.{Style.RESET_ALL}"); return None
    except (ccxt.OrderNotFound, ccxt.InvalidOrder, ccxt.NotSupported, ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadSymbol, ValueError) as e: logger.error(f"{Fore.RED}[{func_name}] Failed update: {type(e).__name__} - {e}{Style.RESET_ALL}"); return None
    except Exception as e: logger.critical(f"{Back.RED}[{func_name}] Unexpected update error: {e}{Style.RESET_ALL}", exc_info=True); return None


# --- fetch_position_risk_bybit_v5 ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=3, initial_delay=1.0)
def fetch_position_risk_bybit_v5(exchange: ccxt.bybit, symbol: str, config: 'Config') -> Optional[Dict[str, Any]]:
    """Fetches detailed risk metrics for the current position of a specific symbol using Bybit V5 logic."""
    func_name = "fetch_position_risk_bybit_v5"; logger.debug(f"[{func_name}] Fetching position risk {symbol} (V5)...")
    default_risk = { 'symbol': symbol, 'side': config.POS_NONE, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0"), 'mark_price': None, 'liq_price': None, 'leverage': None, 'initial_margin': None, 'maint_margin': None, 'unrealized_pnl': None, 'imr': None, 'mmr': None, 'position_value': None, 'risk_limit_value': None, 'info': {} }
    try:
        market = exchange.market(symbol); market_id = market['id']; category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}[{func_name}] Not a contract symbol: {symbol}.{Style.RESET_ALL}"); return default_risk
        params = {'category': category, 'symbol': market_id}; position_data: Optional[List[Dict]] = None; fetch_method_used = "N/A"
        if exchange.has.get('fetchPositionsRisk'):
            try: logger.debug(f"[{func_name}] Using fetch_positions_risk..."); position_data = exchange.fetch_positions_risk(symbols=[symbol], params=params); fetch_method_used = "fetchPositionsRisk"
            except Exception as e: logger.warning(f"[{func_name}] fetch_positions_risk failed ({type(e).__name__}). Falling back."); position_data = None
        if position_data is None:
             if exchange.has.get('fetchPositions'): logger.debug(f"[{func_name}] Falling back to fetch_positions..."); position_data = exchange.fetch_positions(symbols=[symbol], params=params); fetch_method_used = "fetchPositions (Fallback)"
             else: logger.error(f"{Fore.RED}[{func_name}] No position fetch methods available.{Style.RESET_ALL}"); return default_risk
        if position_data is None: logger.error(f"{Fore.RED}[{func_name}] Failed fetch position data ({fetch_method_used}).{Style.RESET_ALL}"); return default_risk
        active_pos_risk: Optional[Dict] = None
        for pos in position_data:
            pos_info = pos.get('info', {}); pos_symbol = pos_info.get('symbol'); pos_v5_side = pos_info.get('side', 'None'); pos_size_str = pos_info.get('size'); pos_idx = int(pos_info.get('positionIdx', -1))
            if pos_symbol == market_id and pos_v5_side != 'None' and pos_idx == 0:
                pos_size = safe_decimal_conversion(pos_size_str, Decimal("0.0"))
                if pos_size is not None and abs(pos_size) > config.POSITION_QTY_EPSILON: active_pos_risk = pos; logger.debug(f"[{func_name}] Found active One-Way pos risk data ({fetch_method_used})."); break
        if not active_pos_risk: logger.info(f"[{func_name}] No active One-Way position found for {symbol}."); return default_risk
        try:
            info = active_pos_risk.get('info', {}); size = safe_decimal_conversion(active_pos_risk.get('contracts', info.get('size'))); entry_price = safe_decimal_conversion(active_pos_risk.get('entryPrice', info.get('avgPrice'))); mark_price = safe_decimal_conversion(active_pos_risk.get('markPrice', info.get('markPrice'))); liq_price = safe_decimal_conversion(active_pos_risk.get('liquidationPrice', info.get('liqPrice'))); leverage = safe_decimal_conversion(active_pos_risk.get('leverage', info.get('leverage'))); initial_margin = safe_decimal_conversion(active_pos_risk.get('initialMargin', info.get('positionIM'))); maint_margin = safe_decimal_conversion(active_pos_risk.get('maintenanceMargin', info.get('positionMM'))); pnl = safe_decimal_conversion(active_pos_risk.get('unrealizedPnl', info.get('unrealisedPnl'))); imr = safe_decimal_conversion(active_pos_risk.get('initialMarginPercentage', info.get('imr'))); mmr = safe_decimal_conversion(active_pos_risk.get('maintenanceMarginPercentage', info.get('mmr'))); pos_value = safe_decimal_conversion(active_pos_risk.get('contractsValue', info.get('positionValue'))); risk_limit = safe_decimal_conversion(info.get('riskLimitValue'))
            pos_side_str = info.get('side'); position_side = config.POS_LONG if pos_side_str == 'Buy' else (config.POS_SHORT if pos_side_str == 'Sell' else config.POS_NONE); quantity = abs(size) if size is not None else Decimal("0.0")
            if position_side == config.POS_NONE or quantity <= config.POSITION_QTY_EPSILON: logger.info(f"[{func_name}] Parsed pos {symbol} negligible."); return default_risk
            log_color = Fore.GREEN if position_side == config.POS_LONG else Fore.RED
            logger.info(f"{log_color}[{func_name}] Position Risk {symbol} ({position_side}):{Style.RESET_ALL}")
            logger.info(f"  Qty:{format_amount(exchange, symbol, quantity)}, Entry:{format_price(exchange, symbol, entry_price)}, Mark:{format_price(exchange, symbol, mark_price)}")
            logger.info(f"  Liq:{format_price(exchange, symbol, liq_price)}, Lev:{leverage}x, uPNL:{format_price(exchange, market['quote'], pnl)}")
            logger.info(f"  IM:{format_price(exchange, market['quote'], initial_margin)}, MM:{format_price(exchange, market['quote'], maint_margin)}")
            logger.info(f"  IMR:{imr:.4% if imr else 'N/A'}, MMR:{mmr:.4% if mmr else 'N/A'}, Value:{format_price(exchange, market['quote'], pos_value)}")
            logger.info(f"  RiskLimitValue:{risk_limit or 'N/A'}")
            return { 'symbol': symbol, 'side': position_side, 'qty': quantity, 'entry_price': entry_price, 'mark_price': mark_price, 'liq_price': liq_price, 'leverage': leverage, 'initial_margin': initial_margin, 'maint_margin': maint_margin, 'unrealized_pnl': pnl, 'imr': imr, 'mmr': mmr, 'position_value': pos_value, 'risk_limit_value': risk_limit, 'info': info }
        except Exception as parse_err: logger.warning(f"{Fore.YELLOW}[{func_name}] Error parsing pos risk: {parse_err}. Data: {str(active_pos_risk)[:300]}{Style.RESET_ALL}"); return default_risk
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching pos risk: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching pos risk: {e}{Style.RESET_ALL}", exc_info=True); return default_risk


# --- set_isolated_margin_bybit_v5 ---
# ... (Function is identical to v2.8) ...
@retry_api_call(max_retries=2, initial_delay=1.0)
def set_isolated_margin_bybit_v5(exchange: ccxt.bybit, symbol: str, leverage: int, config: 'Config') -> bool:
    """Sets margin mode to ISOLATED for a specific symbol on Bybit V5 and sets leverage for it."""
    func_name = "set_isolated_margin_bybit_v5"; logger.info(f"{Fore.CYAN}[{func_name}] Attempting ISOLATED for {symbol} with {leverage}x...{Style.RESET_ALL}")
    ret_code = -1; if leverage <= 0: logger.error(f"{Fore.RED}[{func_name}] Leverage must be positive.{Style.RESET_ALL}"); return False
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}[{func_name}] Not a contract symbol: {symbol} ({category}).{Style.RESET_ALL}"); return False
        try: # Prefer unified method
            logger.debug(f"[{func_name}] Attempting via unified exchange.set_margin_mode...")
            exchange.set_margin_mode(marginMode='isolated', symbol=symbol, params={'category': category, 'leverage': leverage})
            logger.info(f"[{func_name}] Unified set_margin_mode call OK. Verifying leverage...")
            leverage_set_success = set_leverage(exchange, symbol, leverage, config)
            if leverage_set_success: logger.success(f"{Fore.GREEN}[{func_name}] Isolated mode & leverage {leverage}x OK via unified for {symbol}.{Style.RESET_ALL}"); return True
            else: logger.error(f"{Fore.RED}[{func_name}] Failed leverage set after unified isolated attempt.{Style.RESET_ALL}"); return False
        except (ccxt.NotSupported, ccxt.ExchangeError, ccxt.ArgumentsRequired) as e_unified: logger.warning(f"[{func_name}] Unified set_margin_mode failed: {e_unified}. Trying private V5...")
        if not hasattr(exchange, 'private_post_v5_position_switch_isolated'): logger.error(f"{Fore.RED}[{func_name}] CCXT lacks 'private_post_v5_position_switch_isolated'.{Style.RESET_ALL}"); return False
        params_switch = { 'category': category, 'symbol': market['id'], 'tradeMode': 1, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage) }
        logger.debug(f"[{func_name}] Calling private_post_v5_position_switch_isolated with params: {params_switch}")
        response = exchange.private_post_v5_position_switch_isolated(params_switch); logger.debug(f"[{func_name}] Raw V5 switch response: {response}")
        ret_code = response.get('retCode'); ret_msg = response.get('retMsg', '').lower(); already_isolated_or_ok = False
        if ret_code == 0: logger.success(f"{Fore.GREEN}[{func_name}] Switched {symbol} to ISOLATED with {leverage}x leverage via V5.{Style.RESET_ALL}"); already_isolated_or_ok = True
        elif ret_code in [110026, 34036] or "margin mode is not modified" in ret_msg: logger.info(f"{Fore.CYAN}[{func_name}] {symbol} already ISOLATED via V5 check. Verifying leverage...{Style.RESET_ALL}"); already_isolated_or_ok = True
        elif ret_code == 110020 or "have position" in ret_msg or "active order" in ret_msg: logger.error(f"{Fore.RED}[{func_name}] Cannot switch {symbol} to ISOLATED: active pos/orders exist. Msg: {response.get('retMsg')}{Style.RESET_ALL}"); return False
        else: raise ccxt.ExchangeError(f"Bybit API error switching isolated mode (V5): Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'")
        if already_isolated_or_ok:
            logger.debug(f"[{func_name}] Explicitly confirming leverage {leverage}x for ISOLATED {symbol}...")
            leverage_set_success = set_leverage(exchange, symbol, leverage, config)
            if leverage_set_success: logger.success(f"{Fore.GREEN}[{func_name}] Leverage confirmed/set {leverage}x for ISOLATED {symbol}.{Style.RESET_ALL}"); return True
            else: logger.error(f"{Fore.RED}[{func_name}] Failed leverage set/confirm after ISOLATED switch/check.{Style.RESET_ALL}"); return False
        return False
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.BadSymbol, ValueError) as e:
        if not (isinstance(e, ccxt.ExchangeError) and ret_code in [110020]): logger.warning(f"{Fore.YELLOW}[{func_name}] API/Input Error setting isolated margin: {e}{Style.RESET_ALL}")
        if isinstance(e, (ccxt.NetworkError, ccxt.AuthenticationError)): raise e
        return False
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error setting isolated margin: {e}{Style.RESET_ALL}", exc_info=True); return False

# --- Example Standalone Testing Block ---
if __name__ == "__main__":
    print(f"{Fore.YELLOW}{Style.BRIGHT}--- Bybit V5 Helpers Module Standalone Execution ---{Style.RESET_ALL}")
    print("Basic syntax checks only. Depends on external Config, logger, and bybit_utils.py.")
    all_funcs = [name for name, obj in locals().items() if callable(obj) and not name.startswith('_')]
    print(f"Found {len(all_funcs)} function definitions.")
    print(f"\n{Fore.GREEN}Basic syntax check passed.{Style.RESET_ALL}")

# --- END OF FILE bybit_helpers.py ---
```

**4. `indicators.py` (v1.1 - Fixed EVT Calculation)**

```python
# --- START OF FILE indicators.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical Indicators Module (v1.1 - Fixed EVT SuperSmoother)

Provides functions to calculate various technical indicators, primarily leveraging the
`pandas_ta` library for efficiency and breadth. Includes standard indicators,
pivot points, and level calculations. Designed to work with pandas DataFrames
containing OHLCV data.

Key Features:
- Wrappers around `pandas_ta` for common indicators.
- Calculation of Standard and Fibonacci Pivot Points.
- Calculation of Support/Resistance levels based on pivots and Fibonacci retracements.
- Custom Ehlers Volumetric Trend (EVT) implementation with SuperSmoother.
- A master function (`calculate_all_indicators`) to compute indicators based on a config.
- Robust error handling and logging.
- Clear type hinting and documentation.

Assumes input DataFrame has columns: 'open', 'high', 'low', 'close', 'volume'
and a datetime index (preferably UTC).
"""

import logging
import sys
from typing import Optional, Dict, Any, Tuple, List, Union

import numpy as np
import pandas as pd
try:
    import pandas_ta as ta # type: ignore[import]
except ImportError:
    print("Error: 'pandas_ta' library not found. Please install it: pip install pandas_ta")
    sys.exit(1)


# --- Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Default level

# --- Constants ---
MIN_PERIODS_DEFAULT = 50 # Default minimum number of data points for reliable calculations

# --- Pivot Point Calculations ---

def calculate_standard_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculates standard pivot points for the *next* period based on HLC."""
    if not all(isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]):
        logger.warning("Invalid input for standard pivot points (NaN or non-numeric).")
        return {}
    if low > high: logger.warning(f"Low ({low}) > High ({high}) for pivot calc.")
    pivots = {}
    try:
        pivot = (high + low + close) / 3.0; pivots['PP'] = pivot
        pivots['S1'] = (2 * pivot) - high; pivots['R1'] = (2 * pivot) - low
        pivots['S2'] = pivot - (high - low); pivots['R2'] = pivot + (high - low)
        pivots['S3'] = low - 2 * (high - pivot); pivots['R3'] = high + 2 * (pivot - low)
    except Exception as e: logger.error(f"Error calculating standard pivots: {e}", exc_info=True); return {}
    return pivots

def calculate_fib_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculates Fibonacci pivot points for the *next* period based on HLC."""
    if not all(isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]):
        logger.warning("Invalid input for Fibonacci pivot points (NaN or non-numeric)."); return {}
    if low > high: logger.warning(f"Low ({low}) > High ({high}) for Fib pivot calc.")
    fib_pivots = {}
    try:
        pivot = (high + low + close) / 3.0; fib_range = high - low
        if abs(fib_range) < 1e-9: logger.warning("Zero range, cannot calculate Fib Pivots accurately."); fib_pivots['PP'] = pivot; return fib_pivots
        fib_pivots['PP'] = pivot
        fib_pivots['S1'] = pivot - (0.382 * fib_range); fib_pivots['R1'] = pivot + (0.382 * fib_range)
        fib_pivots['S2'] = pivot - (0.618 * fib_range); fib_pivots['R2'] = pivot + (0.618 * fib_range)
        fib_pivots['S3'] = pivot - (1.000 * fib_range); fib_pivots['R3'] = pivot + (1.000 * fib_range)
    except Exception as e: logger.error(f"Error calculating Fib pivots: {e}", exc_info=True); return {}
    return fib_pivots

# --- Support / Resistance Level Calculation ---

def calculate_levels(df_period: pd.DataFrame, current_price: Optional[float] = None) -> Dict[str, Any]:
    """Calculates various support/resistance levels based on historical data."""
    levels: Dict[str, Any] = {"support": {}, "resistance": {}, "pivot": None, "fib_retracements": {}, "standard_pivots": {}, "fib_pivots": {}}
    required_cols = ['high', 'low', 'close']
    if df_period is None or df_period.empty or not all(col in df_period.columns for col in required_cols): logger.warning("Cannot calculate levels: Invalid DataFrame."); return levels
    standard_pivots, fib_pivots = {}, {}
    if len(df_period) >= 2:
        try:
            prev_row = df_period.iloc[-2]
            if not prev_row.isnull().any():
                standard_pivots = calculate_standard_pivot_points(prev_row["high"], prev_row["low"], prev_row["close"])
                fib_pivots = calculate_fib_pivot_points(prev_row["high"], prev_row["low"], prev_row["close"])
            else: logger.warning("Previous candle data contains NaN, skipping pivot calculation.")
        except IndexError: logger.warning("IndexError calculating pivots (need >= 2 rows).")
        except Exception as e: logger.error(f"Error calculating pivots: {e}", exc_info=True)
    else: logger.warning("Cannot calculate pivots: Need >= 2 data points.")

    levels["standard_pivots"] = standard_pivots; levels["fib_pivots"] = fib_pivots
    levels["pivot"] = standard_pivots.get('PP') if standard_pivots else fib_pivots.get('PP')

    try:
        period_high = df_period["high"].max(); period_low = df_period["low"].min()
        if pd.notna(period_high) and pd.notna(period_low):
            period_diff = period_high - period_low
            if abs(period_diff) > 1e-9:
                levels["fib_retracements"] = {"High": period_high, "Fib 78.6%": period_low + period_diff*0.786, "Fib 61.8%": period_low + period_diff*0.618, "Fib 50.0%": period_low + period_diff*0.5, "Fib 38.2%": period_low + period_diff*0.382, "Fib 23.6%": period_low + period_diff*0.236, "Low": period_low}
        else: logger.warning("Could not calculate Fib retracements due to NaN in period High/Low.")
    except Exception as e: logger.error(f"Error calculating Fib retracements: {e}", exc_info=True)

    try:
        cp = float(current_price) if current_price is not None else levels.get("pivot")
        if cp is not None and isinstance(cp, (int, float)) and not np.isnan(cp):
            all_levels = {**{f"Std {k}": v for k, v in standard_pivots.items() if pd.notna(v)}, **{f"Fib {k}": v for k, v in fib_pivots.items() if k != 'PP' and pd.notna(v)}, **{k: v for k, v in levels["fib_retracements"].items() if pd.notna(v)}}
            for label, value in all_levels.items():
                if value < cp: levels["support"][label] = value
                elif value > cp: levels["resistance"][label] = value
        else: logger.debug("Cannot classify S/R relative to current price/pivot.")
    except Exception as e: logger.error(f"Error classifying S/R levels: {e}", exc_info=True)

    levels["support"] = dict(sorted(levels["support"].items(), key=lambda item: item[1], reverse=True))
    levels["resistance"] = dict(sorted(levels["resistance"].items(), key=lambda item: item[1]))
    return levels


# --- Custom Indicator Example ---

def calculate_vwma(close: pd.Series, volume: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Volume Weighted Moving Average (VWMA)."""
    if close is None or volume is None or close.empty or volume.empty or length <= 0 or len(close) < length or len(close) != len(volume):
        logger.warning(f"VWMA calculation skipped: Invalid inputs or insufficient length (need {length}).")
        return None
    if close.isnull().all() or volume.isnull().all():
        logger.warning(f"VWMA calculation skipped: Close or Volume series contains all NaNs.")
        return pd.Series(np.nan, index=close.index, name=f"VWMA_{length}")
    try:
        pv = (close * volume).fillna(0)
        cumulative_pv = pv.rolling(window=length, min_periods=length).sum()
        cumulative_vol = volume.rolling(window=length, min_periods=length).sum()
        vwma = np.where(cumulative_vol != 0, cumulative_pv / cumulative_vol, np.nan)
        return pd.Series(vwma, index=close.index, name=f"VWMA_{length}")
    except Exception as e:
        logger.error(f"Error calculating VWMA(length={length}): {e}", exc_info=True)
        return None

def ehlers_volumetric_trend(df: pd.DataFrame, length: int, multiplier: Union[float, int]) -> pd.DataFrame:
    """
    Calculate Ehlers Volumetric Trend using VWMA and SuperSmoother filter.
    Adds columns: 'vwma_X', 'smooth_vwma_X', 'evt_trend_X', 'evt_buy_X', 'evt_sell_X'.
    """
    required_cols = ['close', 'volume']
    if not all(col in df.columns for col in required_cols) or df.empty or length <= 1 or multiplier <= 0:
        logger.warning(f"EVT calculation skipped: Missing columns, empty df, or invalid params (len={length}, mult={multiplier}).")
        return df

    df_out = df.copy()
    vwma_col = f'vwma_{length}'
    smooth_col = f'smooth_vwma_{length}'
    trend_col = f'evt_trend_{length}'
    buy_col = f'evt_buy_{length}'
    sell_col = f'evt_sell_{length}'

    try:
        vwma = calculate_vwma(df_out['close'], df_out['volume'], length=length)
        if vwma is None or vwma.isnull().all():
            raise ValueError(f"VWMA calculation failed for EVT (length={length})")
        df_out[vwma_col] = vwma

        # SuperSmoother Filter Calculation
        if length <= 0: raise ValueError("SuperSmoother length must be positive")
        a = np.exp(-np.sqrt(2.0) * np.pi / length)
        b = 2.0 * a * np.cos(np.sqrt(2.0) * np.pi / length)
        c2 = b; c3 = -a * a; c1 = 1.0 - c2 - c3

        # Initialize smoothed series & apply filter iteratively
        smoothed = pd.Series(np.nan, index=df_out.index, dtype=float)
        vwma_valid = df_out[vwma_col].copy().fillna(method='ffill').fillna(0) # Handle NaNs for filter init

        # Prime the first two values for the iterative filter
        if len(df_out) > 0: smoothed.iloc[0] = vwma_valid.iloc[0]
        if len(df_out) > 1: smoothed.iloc[1] = vwma_valid.iloc[1] # Simple init, filter starts effectively at index 2

        for i in range(2, len(df_out)):
            smoothed.iloc[i] = c1 * vwma_valid.iloc[i] + c2 * smoothed.iloc[i-1] + c3 * smoothed.iloc[i-2]

        df_out[smooth_col] = smoothed

        # Trend Determination
        mult_h = 1.0 + float(multiplier) / 100.0
        mult_l = 1.0 - float(multiplier) / 100.0
        shifted_smooth = df_out[smooth_col].shift(1)

        # Conditions - ensure comparison happens only where both values are valid
        valid_comparison = pd.notna(df_out[smooth_col]) & pd.notna(shifted_smooth)
        up_trend_cond = valid_comparison & (df_out[smooth_col] > shifted_smooth * mult_h)
        down_trend_cond = valid_comparison & (df_out[smooth_col] < shifted_smooth * mult_l)

        # Vectorized trend calculation using forward fill
        trend = pd.Series(np.nan, index=df_out.index, dtype=float) # Start with NaN
        trend[up_trend_cond] = 1.0
        trend[down_trend_cond] = -1.0
        df_out[trend_col] = trend.ffill().fillna(0).astype(int) # Forward fill trend, fill initial NaNs with 0

        # Buy/Sell Signal Generation (Trend Initiation)
        trend_shifted = df_out[trend_col].shift(1).fillna(0) # Fill NaN for first comparison
        df_out[buy_col] = (df_out[trend_col] == 1) & (trend_shifted != 1)
        df_out[sell_col] = (df_out[trend_col] == -1) & (trend_shifted != -1)

        logger.debug(f"Ehlers Volumetric Trend (len={length}, mult={multiplier}) calculated.")
        return df_out

    except Exception as e:
        logger.error(f"Error in EVT(len={length}, mult={multiplier}): {e}", exc_info=True)
        # Add NaN columns to original df before returning to signal failure
        df[vwma_col] = np.nan; df[smooth_col] = np.nan; df[trend_col] = np.nan
        df[buy_col] = np.nan; df[sell_col] = np.nan
        return df

# --- Master Indicator Calculation Function ---

def calculate_all_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Calculates enabled technical indicators using pandas_ta and custom functions."""
    if df is None or df.empty: logger.error("Input DataFrame empty."); return pd.DataFrame()
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]; logger.error(f"Input DataFrame missing: {missing}."); return df.copy()
    df_out = df.copy(); settings = config.get("indicator_settings", {}); flags = config.get("analysis_flags", {})
    min_rows = settings.get("min_data_periods", MIN_PERIODS_DEFAULT)
    if len(df_out.dropna(subset=required_cols)) < min_rows: logger.warning(f"Insufficient valid rows ({len(df_out.dropna(subset=required_cols))}) < {min_rows}. Results may be NaN.")

    def get_param(name: str, default: Any = None) -> Any: return settings.get(name, default)

    # --- Calculate Standard Indicators ---
    # Example: Calculate ATR if needed by strategy or other indicators
    atr_len = get_param('atr_period', 14)
    if flags.get("use_atr", False) and atr_len > 0:
        try: df_out.ta.atr(length=atr_len, append=True)
        except Exception as e: logger.error(f"Error calculating ATR({atr_len}): {e}", exc_info=False)

    # Add other pandas_ta indicators based on flags here...
    # Example: EMA
    if flags.get("use_ema"):
        try:
             ema_s = get_param('ema_short_period', 12); ema_l = get_param('ema_long_period', 26)
             if ema_s > 0: df_out.ta.ema(length=ema_s, append=True)
             if ema_l > 0: df_out.ta.ema(length=ema_l, append=True)
        except Exception as e: logger.error(f"Error calculating EMA: {e}", exc_info=False)

    # --- Calculate Custom Strategy Indicators ---
    strategy_config = config.get('strategy_params', {}).get(config.get('strategy', {}).get('name', '').lower(), {})
    # Ehlers Volumetric Trend (Primary)
    if flags.get("use_evt"): # Generic flag or strategy-specific check
        try:
            evt_len = strategy_config.get('evt_length', get_param('evt_length', 7))
            evt_mult = strategy_config.get('evt_multiplier', get_param('evt_multiplier', 2.5))
            if evt_len > 0 and evt_mult > 0:
                df_out = ehlers_volumetric_trend(df_out, evt_len, float(evt_mult))
            else: logger.warning("Invalid parameters for EVT, skipping.")
        except Exception as e: logger.error(f"Error calculating EVT: {e}", exc_info=True)

    # Example: Dual EVT Strategy specific logic (if needed)
    if config.get('strategy',{}).get('name','').lower() == "dual_ehlers_volumetric":
         # Rename primary columns if necessary
         # ... rename logic ...
         # Calculate confirmation EVT
         try:
            conf_evt_len = strategy_config.get('confirm_evt_length', get_param('confirm_evt_length', 5))
            conf_evt_mult = strategy_config.get('confirm_evt_multiplier', get_param('confirm_evt_multiplier', 2.0))
            if conf_evt_len > 0 and conf_evt_mult > 0:
                df_out = ehlers_volumetric_trend(df_out, conf_evt_len, float(conf_evt_mult))
                # Rename confirmation columns
                # ... rename logic ...
            else: logger.warning("Invalid parameters for Confirmation EVT, skipping.")
         except Exception as e: logger.error(f"Error calculating Confirmation EVT: {e}", exc_info=True)

    logger.debug(f"Finished calculating indicators. Final DataFrame shape: {df_out.shape}")
    return df_out


# --- Example Standalone Usage ---
if __name__ == "__main__":
    print("-" * 60); print("--- Indicator Module Demo (v1.1) ---"); print("-" * 60)
    logger.setLevel(logging.DEBUG)
    # ... (Rest of the example usage remains the same) ...
    print("\n--- Example 1: Basic Setup ---")
    test_config = {"indicator_settings": {"atr_period": 14, "evt_length": 7, "evt_multiplier": 2.5},"analysis_flags": {"use_atr": True, "use_evt": True}}
    # Create dummy data (same as before)
    periods=200; prices=55.0*np.exp(np.cumsum(np.random.normal(0.0001,0.01,periods))); data={'timestamp':pd.date_range(start='2023-01-01',periods=periods-1,freq='H',tz='UTC'),'open':prices[:-2],'close':prices[1:-1]}; df_test=pd.DataFrame(data).set_index('timestamp'); df_test['high']=df_test[['open','close']].max(axis=1)*(1+np.random.uniform(0,0.01,periods-1)); df_test['low']=df_test[['open','close']].min(axis=1)*(1-np.random.uniform(0,0.01,periods-1)); df_test['high']=np.maximum.reduce([df_test['open'],df_test['close'],df_test['high']]); df_test['low']=np.minimum.reduce([df_test['open'],df_test['close'],df_test['low']]); df_test['volume']=np.random.uniform(100,2000,periods-1);
    print(f"Input shape: {df_test.shape}"); print(f"Input head:\n{df_test.head()}")
    df_results = calculate_all_indicators(df_test, test_config)
    print("-" * 60); print(f"Output shape: {df_results.shape}"); print(f"Output tail:\n{df_results.tail()}"); print("-" * 60)
    print(f"Output columns ({len(df_results.columns)}): {df_results.columns.tolist()}"); print("-" * 60)
    last_row_nans = df_results.iloc[-1].isnull().sum(); print(f"NaNs in last row: {last_row_nans}");

# --- END OF FILE indicators.py ---
```

**5. `ehlers_volumatic_straregy.py` (v1.3 - Class Structure, TP, Order Mgmt)**

```python
# --- START OF FILE ehlers_volumatic_straregy.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ehlers Volumetric Trend Strategy for Bybit V5 (v1.3 - Class, TP, Order Mgmt)

This script implements a trading strategy based on the Ehlers Volumetric Trend
indicator using the Bybit V5 API via CCXT. It leverages custom helper modules
for exchange interaction, indicator calculation, logging, and utilities.

Strategy Logic:
- Uses Ehlers Volumetric Trend (EVT) for primary entry signals.
- Enters LONG on EVT bullish trend initiation.
- Enters SHORT on EVT bearish trend initiation.
- Exits positions when the EVT trend reverses.
- Uses ATR-based stop-loss and take-profit orders (placed as reduce-only limit orders).
- Manages position size based on risk percentage.
- Includes error handling, retries, and rate limit awareness via helper modules.
- Encapsulated within an EhlersStrategy class.
"""

import os
import sys
import time
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP # Import ROUND_UP for TP
from typing import Optional, Dict, Tuple, Any

# Third-party libraries
import ccxt
import pandas as pd
from dotenv import load_dotenv
# --- Import Colorama for main script logging ---
try:
    from colorama import Fore, Style, Back, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    print("Warning: 'colorama' library not found. Main script logs will not be colored.", file=sys.stderr)
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor()


# --- Import Custom Modules ---
try:
    from neon_logger import setup_logger
    import bybit_helpers # Import the module itself
    import indicators as ind
    from bybit_utils import (
        safe_decimal_conversion, format_price, format_amount,
        format_order_id, send_sms_alert, retry_api_call
    )
except ImportError as e:
    print(f"FATAL: Error importing helper modules: {e}")
    print("Ensure all .py files (bybit_helpers, indicators, neon_logger, bybit_utils) are present.")
    sys.exit(1)

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration Class ---
class Config:
    def __init__(self):
        # Exchange & API
        self.EXCHANGE_ID: str = "bybit"
        self.API_KEY: Optional[str] = os.getenv("BYBIT_API_KEY")
        self.API_SECRET: Optional[str] = os.getenv("BYBIT_API_SECRET")
        self.TESTNET_MODE: bool = os.getenv("BYBIT_TESTNET_MODE", "true").lower() == "true"
        self.DEFAULT_RECV_WINDOW: int = int(os.getenv("DEFAULT_RECV_WINDOW", 10000))

        # Symbol & Market
        self.SYMBOL: str = os.getenv("SYMBOL", "BTC/USDT:USDT") # Example: BTC/USDT Perpetual
        self.USDT_SYMBOL: str = "USDT"
        self.EXPECTED_MARKET_TYPE: str = 'swap'
        self.EXPECTED_MARKET_LOGIC: str = 'linear'
        self.TIMEFRAME: str = os.getenv("TIMEFRAME", "5m")
        self.OHLCV_LIMIT: int = int(os.getenv("OHLCV_LIMIT", 200)) # Candles for indicators

        # Account & Position Settings
        self.DEFAULT_LEVERAGE: int = int(os.getenv("LEVERAGE", 10))
        self.DEFAULT_MARGIN_MODE: str = 'cross' # Or 'isolated' (Note: requires UTA Pro upgrade on Bybit usually)
        self.DEFAULT_POSITION_MODE: str = 'one-way' # Or 'hedge'
        self.RISK_PER_TRADE: Decimal = Decimal(os.getenv("RISK_PER_TRADE", "0.01")) # 1% risk

        # Order Settings
        self.DEFAULT_SLIPPAGE_PCT: Decimal = Decimal(os.getenv("DEFAULT_SLIPPAGE_PCT", "0.005")) # 0.5%
        self.ORDER_BOOK_FETCH_LIMIT: int = 25
        self.SHALLOW_OB_FETCH_DEPTH: int = 5
        self.PLACE_TPSL_AS_LIMIT: bool = True # Place TP/SL as reduce-only Limit orders (vs native stop market)

        # Fees
        self.TAKER_FEE_RATE: Decimal = Decimal(os.getenv("BYBIT_TAKER_FEE", "0.00055"))
        self.MAKER_FEE_RATE: Decimal = Decimal(os.getenv("BYBIT_MAKER_FEE", "0.0002"))

        # Strategy Parameters (Ehlers Volumetric Trend)
        self.EVT_ENABLED: bool = True # Master switch for the indicator calc
        self.EVT_LENGTH: int = int(os.getenv("EVT_LENGTH", 7))
        self.EVT_MULTIPLIER: float = float(os.getenv("EVT_MULTIPLIER", 2.5))
        self.STOP_LOSS_ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", 14))
        self.STOP_LOSS_ATR_MULTIPLIER: Decimal = Decimal(os.getenv("ATR_MULTIPLIER", "2.0"))
        # **NEW**: Take Profit Parameter
        self.TAKE_PROFIT_ATR_MULTIPLIER: Decimal = Decimal(os.getenv("TAKE_PROFIT_ATR_MULTIPLIER", "3.0"))

        # Retry & Timing
        self.RETRY_COUNT: int = int(os.getenv("RETRY_COUNT", 3))
        self.RETRY_DELAY_SECONDS: float = float(os.getenv("RETRY_DELAY", 2.0))
        self.LOOP_DELAY_SECONDS: int = int(os.getenv("LOOP_DELAY", 60)) # Wait time between cycles

        # Logging & Alerts
        self.LOG_CONSOLE_LEVEL: str = os.getenv("LOG_CONSOLE_LEVEL", "INFO")
        self.LOG_FILE_LEVEL: str = os.getenv("LOG_FILE_LEVEL", "DEBUG")
        self.LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "ehlers_strategy.log")
        self.ENABLE_SMS_ALERTS: bool = os.getenv("ENABLE_SMS_ALERTS", "false").lower() == "true"
        self.SMS_RECIPIENT_NUMBER: Optional[str] = os.getenv("SMS_RECIPIENT_NUMBER")
        self.SMS_TIMEOUT_SECONDS: int = 30

        # Constants
        self.SIDE_BUY: str = "buy"
        self.SIDE_SELL: str = "sell"
        self.POS_LONG: str = "LONG"
        self.POS_SHORT: str = "SHORT"
        self.POS_NONE: str = "NONE"
        self.POSITION_QTY_EPSILON: Decimal = Decimal("1e-9") # Small value to compare quantities

        # --- Derived/Helper Attributes ---
        self.indicator_settings = {
            "atr_period": self.STOP_LOSS_ATR_PERIOD,
             "evt_length": self.EVT_LENGTH,
             "evt_multiplier": self.EVT_MULTIPLIER,
        }
        self.analysis_flags = {"use_atr": True, "use_evt": self.EVT_ENABLED}
        self.strategy_params = {'ehlers_volumetric': {'evt_length': self.EVT_LENGTH, 'evt_multiplier': self.EVT_MULTIPLIER}}
        self.strategy = {'name': 'ehlers_volumetric'}


# --- Global Logger ---
# Logger configured in main block
logger: logging.Logger = None


# --- Strategy Class ---
class EhlersStrategy:
    """Encapsulates the Ehlers Volumetric Trend trading strategy logic."""

    def __init__(self, config: Config):
        self.config = config
        self.symbol = config.SYMBOL
        self.timeframe = config.TIMEFRAME
        self.exchange: Optional[ccxt.bybit] = None # Will be initialized
        self.bybit_helpers = bybit_helpers # Store module for access
        self.is_initialized = False
        self.is_running = False

        # Position State
        self.current_side: str = self.config.POS_NONE
        self.current_qty: Decimal = Decimal("0.0")
        self.entry_price: Optional[Decimal] = None
        self.sl_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None

        # Market details
        self.min_qty: Optional[Decimal] = None
        self.qty_step: Optional[Decimal] = None
        self.price_tick: Optional[Decimal] = None

        logger.info(f"EhlersStrategy initialized for {self.symbol} on {self.timeframe}.")

    def _initialize(self) -> bool:
        """Connects to the exchange, validates market, sets config, fetches initial state."""
        logger.info(f"{Fore.CYAN}--- Strategy Initialization Phase ---{Style.RESET_ALL}")
        try:
            self.exchange = self.bybit_helpers.initialize_bybit(self.config)
            if not self.exchange: return False

            market_details = self.bybit_helpers.validate_market(self.exchange, self.symbol, self.config)
            if not market_details: return False
            self._extract_market_details(market_details)

            logger.info(f"Setting leverage for {self.symbol} to {self.config.DEFAULT_LEVERAGE}x...")
            if not self.bybit_helpers.set_leverage(self.exchange, self.symbol, self.config.DEFAULT_LEVERAGE, self.config):
                 logger.critical(f"{Back.RED}Failed to set leverage.{Style.RESET_ALL}")
                 return False
            logger.success("Leverage set/confirmed.")

            # Set Position Mode (One-Way) - Optional but recommended for clarity
            logger.info(f"Attempting to set position mode to '{self.config.DEFAULT_POSITION_MODE}'...")
            mode_set = self.bybit_helpers.set_position_mode_bybit_v5(self.exchange, self.symbol, self.config.DEFAULT_POSITION_MODE, self.config)
            if not mode_set:
                 logger.warning(f"{Fore.YELLOW}Could not explicitly set position mode to '{self.config.DEFAULT_POSITION_MODE}'. Ensure it's set correctly in Bybit UI.{Style.RESET_ALL}")
            else:
                 logger.info(f"Position mode confirmed/set to '{self.config.DEFAULT_POSITION_MODE}'.")


            logger.info("Fetching initial account state (position, orders, balance)...")
            if not self._update_state():
                 logger.error("Failed to fetch initial state.")
                 # Decide if this is critical - maybe allow continuing if balance fetch worked?
                 # For now, let's require position state.
                 return False

            logger.info(f"Initial Position: Side={self.current_side}, Qty={self.current_qty}")
            # Initial balance log moved to state update

            logger.info("Performing initial cleanup: cancelling existing orders...")
            if not self._cancel_open_orders("Initialization Cleanup"):
                 logger.warning("Initial order cancellation failed or encountered issues.")
                 # Continue anyway? Or stop? Let's continue but log warning.

            self.is_initialized = True
            logger.success("--- Strategy Initialization Complete ---")
            return True

        except Exception as e:
            logger.critical(f"{Back.RED}Critical error during strategy initialization: {e}{Style.RESET_ALL}", exc_info=True)
            return False

    def _extract_market_details(self, market: Dict):
        """Extracts and stores relevant market limits and precision."""
        self.min_qty = safe_decimal_conversion(market.get('limits', {}).get('amount', {}).get('min'))
        amount_precision = market.get('precision', {}).get('amount')
        self.qty_step = (Decimal('1') / (Decimal('10') ** amount_precision)) if amount_precision is not None else None
        price_precision = market.get('precision', {}).get('price')
        self.price_tick = (Decimal('1') / (Decimal('10') ** price_precision)) if price_precision is not None else None
        logger.info(f"Market Details Set: Min Qty={self.min_qty}, Qty Step={self.qty_step}, Price Tick={self.price_tick}")

    def _update_state(self) -> bool:
        """Fetches and updates the current position, balance, and open orders."""
        logger.debug("Updating strategy state...")
        try:
            # Fetch Position
            pos_data = self.bybit_helpers.get_current_position_bybit_v5(self.exchange, self.symbol, self.config)
            if pos_data is None:
                logger.error("Failed to fetch position data during state update.")
                return False # Treat as critical failure

            self.current_side = pos_data['side']
            self.current_qty = pos_data['qty']
            self.entry_price = pos_data.get('entry_price') # Can be None if no position

            # Fetch Balance
            _, available_balance = self.bybit_helpers.fetch_usdt_balance(self.exchange, self.config)
            if available_balance is None:
                logger.error("Failed to fetch available balance during state update.")
                return False # Treat as critical failure
            logger.info(f"Available Balance: {available_balance:.4f} {self.config.USDT_SYMBOL}")

            # We don't fetch all open orders here anymore, only check SL/TP status if needed
            # If not in position, reset tracked orders
            if self.current_side == self.config.POS_NONE:
                self.sl_order_id = None
                self.tp_order_id = None

            logger.debug("State update complete.")
            return True

        except Exception as e:
            logger.error(f"Unexpected error during state update: {e}", exc_info=True)
            return False

    def _fetch_data(self) -> Tuple[Optional[pd.DataFrame], Optional[Decimal]]:
        """Fetches OHLCV data and the latest ticker price."""
        logger.debug("Fetching market data...")
        ohlcv_df = self.bybit_helpers.fetch_ohlcv_paginated(
            self.exchange, self.symbol, self.timeframe,
            limit_per_req=1000, max_total_candles=self.config.OHLCV_LIMIT, config=self.config
        )
        if ohlcv_df is None or ohlcv_df.empty:
            logger.warning("Could not fetch sufficient OHLCV data.")
            return None, None

        ticker = self.bybit_helpers.fetch_ticker_validated(self.exchange, self.symbol, self.config)
        if ticker is None:
            logger.warning("Could not fetch valid ticker data.")
            return ohlcv_df, None # Return OHLCV if available, but no price

        current_price = ticker.get('last')
        if current_price is None:
            logger.warning("Ticker data retrieved but missing 'last' price.")
            return ohlcv_df, None

        logger.debug(f"Data fetched: {len(ohlcv_df)} candles, Last Price: {current_price}")
        return ohlcv_df, current_price

    def _calculate_indicators(self, ohlcv_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates indicators based on the config."""
        if ohlcv_df is None or ohlcv_df.empty: return None
        logger.debug("Calculating indicators...")
        # Prepare config for indicator module
        indicator_config = {
            "indicator_settings": self.config.indicator_settings,
            "analysis_flags": self.config.analysis_flags,
            "strategy_params": self.config.strategy_params,
            "strategy": self.config.strategy
        }
        df_with_indicators = ind.calculate_all_indicators(ohlcv_df, indicator_config)

        # Validate necessary columns exist
        evt_trend_col = f'evt_trend_{self.config.EVT_LENGTH}'
        atr_col = f'ATRr_{self.config.STOP_LOSS_ATR_PERIOD}'
        if df_with_indicators is None or evt_trend_col not in df_with_indicators.columns:
            logger.error(f"Required EVT trend column '{evt_trend_col}' not found after calculation.")
            return None
        if atr_col not in df_with_indicators.columns:
             logger.error(f"Required ATR column '{atr_col}' not found after calculation.")
             return None # ATR is essential for SL/TP

        logger.debug("Indicators calculated successfully.")
        return df_with_indicators

    def _generate_signals(self, df_ind: pd.DataFrame) -> str | None:
        """Generates trading signals based on the last indicator data point."""
        if df_ind is None or df_ind.empty: return None
        logger.debug("Generating trading signals...")
        try:
            latest = df_ind.iloc[-1]
            trend_col = f'evt_trend_{self.config.EVT_LENGTH}'
            buy_col = f'evt_buy_{self.config.EVT_LENGTH}'
            sell_col = f'evt_sell_{self.config.EVT_LENGTH}'

            if not all(col in latest.index and pd.notna(latest[col]) for col in [trend_col, buy_col, sell_col]):
                 logger.warning(f"EVT signal columns missing or NaN in latest data: {latest[[trend_col, buy_col, sell_col]].to_dict()}")
                 return None

            buy_signal = latest[buy_col]
            sell_signal = latest[sell_col]

            if buy_signal:
                logger.info(f"{Fore.GREEN}BUY signal generated based on EVT Buy flag.{Style.RESET_ALL}")
                return self.config.SIDE_BUY
            elif sell_signal:
                logger.info(f"{Fore.RED}SELL signal generated based on EVT Sell flag.{Style.RESET_ALL}")
                return self.config.SIDE_SELL
            else:
                logger.debug("No new entry signal generated.")
                return None

        except IndexError:
            logger.warning("IndexError generating signals (DataFrame likely too short).")
            return None
        except Exception as e:
            logger.error(f"Error generating signals: {e}", exc_info=True)
            return None

    def _calculate_sl_tp(self, df_ind: pd.DataFrame, side: str, entry_price: Decimal) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculates initial stop-loss and take-profit prices."""
        if df_ind is None or df_ind.empty: return None, None
        logger.debug(f"Calculating SL/TP for {side} entry at {entry_price}...")
        try:
            atr_col = f'ATRr_{self.config.STOP_LOSS_ATR_PERIOD}'
            if atr_col not in df_ind.columns: logger.error(f"ATR column '{atr_col}' not found."); return None, None

            latest_atr = safe_decimal_conversion(df_ind.iloc[-1][atr_col])
            if latest_atr is None or latest_atr <= Decimal(0):
                logger.warning("Invalid ATR value for SL/TP calculation. Cannot proceed.")
                return None, None # Require valid ATR

            # Stop Loss Calculation
            sl_offset = latest_atr * self.config.STOP_LOSS_ATR_MULTIPLIER
            stop_loss_price = (entry_price - sl_offset) if side == self.config.SIDE_BUY else (entry_price + sl_offset)
            # Basic validation
            if side == self.config.SIDE_BUY and stop_loss_price >= entry_price: stop_loss_price = entry_price * (Decimal(1) - Decimal("0.001"))
            if side == self.config.SIDE_SELL and stop_loss_price <= entry_price: stop_loss_price = entry_price * (Decimal(1) + Decimal("0.001"))
            # Format to price tick precision
            sl_formatted = self.bybit_helpers.format_price(self.exchange, self.symbol, stop_loss_price)
            sl_price_precise = safe_decimal_conversion(sl_formatted)
            if sl_price_precise is None: logger.error("Failed to format SL price precisely."); return None, None

            # Take Profit Calculation
            tp_offset = latest_atr * self.config.TAKE_PROFIT_ATR_MULTIPLIER
            take_profit_price = (entry_price + tp_offset) if side == self.config.SIDE_BUY else (entry_price - tp_offset)
            # Ensure TP is logical relative to entry
            if side == self.config.SIDE_BUY and take_profit_price <= entry_price: logger.warning("Calculated Buy TP below entry. Skipping TP."); return sl_price_precise, None
            if side == self.config.SIDE_SELL and take_profit_price >= entry_price: logger.warning("Calculated Sell TP above entry. Skipping TP."); return sl_price_precise, None
             # Format to price tick precision
            tp_formatted = self.bybit_helpers.format_price(self.exchange, self.symbol, take_profit_price)
            tp_price_precise = safe_decimal_conversion(tp_formatted)
            if tp_price_precise is None: logger.error("Failed to format TP price precisely."); return sl_price_precise, None

            logger.info(f"Calculated SL: {format_price(self.exchange, self.symbol, sl_price_precise)}, "
                        f"TP: {format_price(self.exchange, self.symbol, tp_price_precise)} (ATR: {latest_atr:.4f})")
            return sl_price_precise, tp_price_precise

        except IndexError: logger.warning("IndexError calculating SL/TP."); return None, None
        except Exception as e: logger.error(f"Error calculating SL/TP: {e}", exc_info=True); return None, None

    def _calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal) -> Optional[Decimal]:
        """Calculates position size based on risk percentage and stop-loss distance."""
        logger.debug("Calculating position size...")
        try:
            _, available_balance = self.bybit_helpers.fetch_usdt_balance(self.exchange, self.config)
            if available_balance is None or available_balance <= Decimal("0"):
                logger.error("Cannot calculate position size: Zero or invalid available balance.")
                return None

            risk_amount_usd = available_balance * self.config.RISK_PER_TRADE
            price_diff = abs(entry_price - stop_loss_price)

            if price_diff <= Decimal("0"):
                logger.error(f"Cannot calculate size: Entry price ({entry_price}) and SL price ({stop_loss_price}) invalid.")
                return None

            position_size_base = risk_amount_usd / price_diff

            # Apply market precision/step size constraints
            if self.qty_step is None:
                 logger.warning("Quantity step size unknown, cannot adjust precisely.")
                 position_size_adjusted = position_size_base # Use raw value
            else:
                 # Round down to the nearest step size increment
                 position_size_adjusted = (position_size_base // self.qty_step) * self.qty_step

            if position_size_adjusted <= Decimal(0):
                 logger.warning(f"Adjusted position size is zero. Step: {self.qty_step}, Orig: {position_size_base}")
                 return None

            if self.min_qty is not None and position_size_adjusted < self.min_qty:
                logger.warning(f"Calculated size ({position_size_adjusted}) < Min Qty ({self.min_qty}). Cannot trade.")
                return None

            logger.info(f"Calculated position size: {format_amount(self.exchange, self.symbol, position_size_adjusted)} "
                        f"(Risk: {risk_amount_usd:.2f} USDT, Balance: {available_balance:.2f} USDT)")
            return position_size_adjusted

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return None

    def _cancel_open_orders(self, reason: str = "Strategy Action") -> bool:
        """Cancels tracked SL and TP orders."""
        cancelled_sl, cancelled_tp = True, True # Assume success if no ID tracked

        if self.sl_order_id:
            logger.info(f"Cancelling existing SL order {format_order_id(self.sl_order_id)} ({reason})...")
            try:
                cancelled_sl = self.bybit_helpers.cancel_order(self.exchange, self.symbol, self.sl_order_id, self.config)
                if cancelled_sl: logger.info("SL order cancelled successfully or already gone.")
                else: logger.warning("Failed to cancel SL order.")
            except Exception as e:
                 logger.error(f"Error cancelling SL order {self.sl_order_id}: {e}", exc_info=True)
                 cancelled_sl = False # Mark as failed on exception
            finally:
                 self.sl_order_id = None # Always clear tracked ID after attempt

        if self.tp_order_id:
             logger.info(f"Cancelling existing TP order {format_order_id(self.tp_order_id)} ({reason})...")
             try:
                 cancelled_tp = self.bybit_helpers.cancel_order(self.exchange, self.symbol, self.tp_order_id, self.config)
                 if cancelled_tp: logger.info("TP order cancelled successfully or already gone.")
                 else: logger.warning("Failed to cancel TP order.")
             except Exception as e:
                  logger.error(f"Error cancelling TP order {self.tp_order_id}: {e}", exc_info=True)
                  cancelled_tp = False # Mark as failed on exception
             finally:
                  self.tp_order_id = None # Always clear tracked ID after attempt

        return cancelled_sl and cancelled_tp # Return overall success

    def _handle_exit(self, df_ind: pd.DataFrame) -> bool:
        """Checks exit conditions and closes the position if necessary."""
        if self.current_side == self.config.POS_NONE: return False # Not in position

        logger.debug("Checking exit conditions...")
        should_exit = False
        exit_reason = ""

        try:
            latest_trend = df_ind.iloc[-1].get(f'evt_trend_{self.config.EVT_LENGTH}')

            if latest_trend is not None:
                if self.current_side == self.config.POS_LONG and latest_trend == -1:
                    should_exit = True; exit_reason = "EVT Trend flipped Short"
                elif self.current_side == self.config.POS_SHORT and latest_trend == 1:
                    should_exit = True; exit_reason = "EVT Trend flipped Long"
            else:
                logger.warning("Cannot determine latest EVT trend for exit check.")

            # --- Add other potential exit reasons ---
            # Example: Check if SL or TP orders were filled (requires fetching order status)
            # if self.sl_order_id:
            #    sl_status = self.bybit_helpers.fetch_order(self.exchange, self.symbol, self.sl_order_id, self.config)
            #    if sl_status and sl_status.get('status') == 'closed': # Adjust based on actual filled status
            #        should_exit = True; exit_reason = "Stop Loss Hit"
            #        logger.warning(f"{Back.RED}{Fore.WHITE}STOP LOSS HIT!{Style.RESET_ALL} Order: {self.sl_order_id}")
            #        self.sl_order_id = None # Clear SL ID as it's closed
            #        # Need to cancel TP if SL hit
            #        if self.tp_order_id: self._cancel_open_orders("SL Hit - Cancelling TP")

            # if not should_exit and self.tp_order_id: # Only check TP if not already exiting for SL/Trend
            #    tp_status = self.bybit_helpers.fetch_order(self.exchange, self.symbol, self.tp_order_id, self.config)
            #    if tp_status and tp_status.get('status') == 'closed':
            #        should_exit = True; exit_reason = "Take Profit Hit"
            #        logger.success(f"{Back.GREEN}{Fore.WHITE}TAKE PROFIT HIT!{Style.RESET_ALL} Order: {self.tp_order_id}")
            #        self.tp_order_id = None # Clear TP ID
            #        # Need to cancel SL if TP hit
            #        if self.sl_order_id: self._cancel_open_orders("TP Hit - Cancelling SL")

            # --- Execute Exit ---
            if should_exit:
                logger.warning(f"{Fore.YELLOW}Exit condition met for {self.current_side} position: {exit_reason}. Attempting to close.{Style.RESET_ALL}")

                # Cancel any remaining SL/TP orders *before* sending close order
                if not self._cancel_open_orders(f"Pre-Close ({exit_reason})"):
                     logger.error("Failed to cancel open SL/TP orders before closing position! Potential dangling orders.")
                     # Decide whether to proceed with close or halt

                # Send reduce-only market order to close
                close_order = self.bybit_helpers.close_position_reduce_only(
                    self.exchange, self.symbol, self.config,
                    position_to_close={'side': self.current_side, 'qty': self.current_qty}, # Provide current state
                    reason=exit_reason
                )

                if close_order:
                    logger.success(f"{Fore.GREEN}Position closed successfully based on: {exit_reason}.{Style.RESET_ALL}")
                    if self.config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{self.symbol}] {self.current_side} Position Closed: {exit_reason}", self.config)
                    # Reset position state after successful close
                    self.current_side = self.config.POS_NONE
                    self.current_qty = Decimal("0.0")
                    self.entry_price = None
                    self.sl_order_id = None # Already cleared by _cancel_open_orders
                    self.tp_order_id = None # Already cleared by _cancel_open_orders
                    time.sleep(5) # Brief pause after closing
                    return True # Indicates an exit occurred
                else:
                    logger.error(f"{Fore.RED}Failed to close position for exit signal! Manual intervention required.{Style.RESET_ALL}")
                    if self.config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{self.symbol}] URGENT: Failed to close {self.current_side} position on exit signal!", self.config)
                    # Don't reset state if close failed
                    return False # Indicates exit failed

        except IndexError:
             logger.warning("IndexError checking exit conditions.")
             return False
        except Exception as e:
            logger.error(f"Error handling exit: {e}", exc_info=True)
            return False

        return False # No exit condition met


    def _handle_entry(self, signal: str, df_ind: pd.DataFrame, current_price: Decimal) -> bool:
        """Handles the process of entering a new position."""
        if self.current_side != self.config.POS_NONE:
             logger.warning(f"Attempted entry signal '{signal}' while already in {self.current_side} position. Skipping.")
             return False
        if not signal: return False

        logger.info(f"{Fore.CYAN}Attempting to enter {signal.upper()} position based on signal...{Style.RESET_ALL}")

        # 1. Pre-entry Cleanup
        logger.info("Running pre-entry order cleanup...")
        self._cancel_open_orders("Pre-Entry Cleanup") # Cancel any lingering tracked orders
        # Cancel any other potential orders just in case
        self.bybit_helpers.cancel_all_orders(self.exchange, self.symbol, self.config, reason="Pre-Entry Global Cleanup", order_filter='Order')
        self.bybit_helpers.cancel_all_orders(self.exchange, self.symbol, self.config, reason="Pre-Entry Global Stop Cleanup", order_filter='StopOrder')


        # 2. Calculate SL/TP and Size
        stop_loss_price, take_profit_price = self._calculate_sl_tp(df_ind, signal, current_price)
        if stop_loss_price is None:
             logger.error("Could not calculate stop-loss. Aborting entry."); return False

        position_size = self._calculate_position_size(current_price, stop_loss_price)
        if position_size is None:
             logger.error("Could not calculate position size. Aborting entry."); return False

        # 3. Place Entry Market Order
        entry_order = self.bybit_helpers.place_market_order_slippage_check(
            self.exchange, self.symbol, signal, position_size, self.config
        )

        if not (entry_order and entry_order.get('id')):
            logger.error(f"{Fore.RED}Entry market order placement failed.{Style.RESET_ALL}"); return False

        logger.success(f"Entry market order submitted: ID ...{format_order_id(entry_order['id'])}")
        time.sleep(5) # Wait for potential fill

        # 4. Verify Position and Get Actual Entry Price/Qty
        if not self._update_state(): # Update internal state after order
            logger.error("Failed to update state after entry order submission. Cannot place SL/TP."); return False

        # Use updated state to confirm entry and get details
        if self.current_side != signal.upper():
             logger.error(f"Position side ({self.current_side}) does not match entry signal ({signal.upper()}) after order! Manual check required.")
             if self.config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{self.symbol}] URGENT: Position side mismatch after entry order {entry_order.get('id')}!", self.config)
             # Attempt to close if we entered the wrong side
             if self.current_side != self.config.POS_NONE:
                  self.bybit_helpers.close_position_reduce_only(self.exchange, self.symbol, self.config, reason="Incorrect Entry Side")
             return False

        if self.current_qty <= self.config.POSITION_QTY_EPSILON:
             logger.error(f"Position quantity is zero or negligible ({self.current_qty}) after entry order. Cannot place SL/TP.")
             if self.config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{self.symbol}] URGENT: Zero quantity after entry order {entry_order.get('id')}!", self.config)
             return False

        # Position confirmed! Log entry details
        actual_entry_price = self.entry_price or current_price # Use actual entry if available, else market price at time of order
        actual_qty = self.current_qty
        logger.success(f"POSITION ENTERED: {self.current_side} {format_amount(self.exchange, self.symbol, actual_qty)} @ avg ~{format_price(self.exchange, self.symbol, actual_entry_price)}")


        # 5. Place SL and TP Orders
        sl_success, tp_success = self._place_sl_tp_orders(signal, actual_qty, stop_loss_price, take_profit_price)

        if not sl_success:
             logger.error(f"{Fore.RED}Failed to place stop-loss order after entry! Attempting immediate close.{Style.RESET_ALL}")
             if self.config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{self.symbol}] URGENT: Failed SL place after {signal.upper()} entry! Closing.", self.config)
             self._cancel_open_orders("Failed SL - Cancelling TP") # Cancel TP if it was placed
             close_success = self.bybit_helpers.close_position_reduce_only(self.exchange, self.symbol, self.config, reason="Failed SL Placement")
             if close_success: logger.warning("Position closed due to failed SL placement.")
             else: logger.critical("CRITICAL: FAILED TO CLOSE POSITION AFTER FAILED SL PLACEMENT!")
             self._update_state() # Update state after attempted close
             return False # Entry failed overall

        if not tp_success and take_profit_price is not None:
            logger.warning(f"{Fore.YELLOW}Failed to place take-profit order, but Stop Loss was placed. Proceeding without TP.{Style.RESET_ALL}")
            # Optionally send SMS alert about missing TP

        # SMS Alert for successful entry with SL (and TP if placed)
        tp_msg_part = f", TP @ {format_price(self.exchange, self.symbol, take_profit_price)}" if tp_success and take_profit_price else " (TP Failed or Skipped)"
        if self.config.ENABLE_SMS_ALERTS: send_sms_alert(f"[{self.symbol}] Entered {self.current_side} {format_amount(self.exchange, self.symbol, actual_qty)} @ ~{format_price(self.exchange, self.symbol, actual_entry_price)}. SL @ {format_price(self.exchange, self.symbol, stop_loss_price)}{tp_msg_part}", self.config)

        return True # Entry successful

    def _place_sl_tp_orders(self, position_side_signal: str, qty: Decimal, sl_price: Decimal, tp_price: Optional[Decimal]) -> Tuple[bool, bool]:
        """Places Stop Loss and Take Profit orders."""
        sl_placed = False
        tp_placed = False

        # Place Stop Loss
        sl_side = self.config.SIDE_SELL if position_side_signal == self.config.SIDE_BUY else self.config.SIDE_BUY
        sl_client_oid = f"sl_{self.symbol.replace('/','').replace(':','')}_{int(time.time())}"[-36:]

        if self.config.PLACE_TPSL_AS_LIMIT:
            sl_order = self.bybit_helpers.place_limit_order_tif(
                self.exchange, self.symbol, sl_side, qty, sl_price, self.config,
                is_reduce_only=True, is_post_only=False, # SL should execute, not be post-only
                client_order_id=sl_client_oid, time_in_force='GTC' # Make SL GTC limit
            )
        else: # Place as native stop market
             sl_order = self.bybit_helpers.place_native_stop_loss(
                 self.exchange, self.symbol, sl_side, qty, sl_price, self.config,
                 client_order_id=sl_client_oid
             )

        if sl_order and sl_order.get('id'):
            logger.success(f"Stop Loss order placed successfully. Type: {'Limit' if self.config.PLACE_TPSL_AS_LIMIT else 'Native Stop Market'}, ID: ...{format_order_id(sl_order['id'])}")
            self.sl_order_id = sl_order['id']
            sl_placed = True
        else:
            logger.error(f"{Fore.RED}Failed to place Stop Loss order!{Style.RESET_ALL}")

        # Place Take Profit (only if SL was successful and TP price is valid)
        if sl_placed and tp_price is not None:
            tp_side = self.config.SIDE_SELL if position_side_signal == self.config.SIDE_BUY else self.config.SIDE_BUY
            tp_client_oid = f"tp_{self.symbol.replace('/','').replace(':','')}_{int(time.time())}"[-36:]
            # Always place TP as a reduce-only limit order
            tp_order = self.bybit_helpers.place_limit_order_tif(
                self.exchange, self.symbol, tp_side, qty, tp_price, self.config,
                is_reduce_only=True, is_post_only=False, # Allow TP to execute immediately if price is favorable
                client_order_id=tp_client_oid, time_in_force='GTC'
            )
            if tp_order and tp_order.get('id'):
                logger.success(f"Take Profit limit order placed successfully. ID: ...{format_order_id(tp_order['id'])}")
                self.tp_order_id = tp_order['id']
                tp_placed = True
            else:
                logger.warning(f"{Fore.YELLOW}Failed to place Take Profit order.{Style.RESET_ALL}")
        elif tp_price is None:
             logger.info("Take Profit price was not valid or calculated, skipping TP order.")
             tp_placed = True # Consider success if TP wasn't desired

        return sl_placed, tp_placed


    def run(self):
        """Main strategy execution loop."""
        if not self._initialize():
            logger.critical("Strategy initialization failed. Exiting.")
            return

        self.is_running = True
        logger.success(f"{Fore.GREEN}--- Strategy Running for {self.symbol} ---{Style.RESET_ALL}")

        while self.is_running:
            try:
                # --- 1. Fetch Data ---
                logger.info(f"{Fore.BLUE}{'-'*20} Cycle Start: {pd.Timestamp.now(tz='UTC').isoformat()} {'-'*20}{Style.RESET_ALL}")
                ohlcv_df, current_price = self._fetch_data()
                if ohlcv_df is None or current_price is None:
                    logger.warning("Failed to fetch data for this cycle.")
                    time.sleep(self.config.LOOP_DELAY_SECONDS); continue

                # --- 2. Calculate Indicators ---
                df_indicators = self._calculate_indicators(ohlcv_df)
                if df_indicators is None:
                    logger.warning("Failed to calculate indicators for this cycle.")
                    time.sleep(self.config.LOOP_DELAY_SECONDS); continue

                # --- 3. Update State (Position Check) ---
                if not self._update_state():
                     logger.error("Failed to update account state. Skipping trading logic for safety.")
                     time.sleep(self.config.LOOP_DELAY_SECONDS); continue

                # --- 4. Handle Exits ---
                if self.current_side != self.config.POS_NONE:
                    if self._handle_exit(df_indicators):
                         # If exit occurred, skip entry logic for this cycle
                         time.sleep(self.config.LOOP_DELAY_SECONDS); continue

                # --- 5. Generate Entry Signal ---
                signal = self._generate_signals(df_indicators)

                # --- 6. Handle Entries ---
                if


Okay, here is the enhanced introductory text and the complete, improved bash script.

---

### **Enhanced Project Setup Script for Bybit EVT Strategy**

This document provides an enhanced bash script designed to streamline the setup of a Python-based trading bot project for the Bybit exchange (V5 API). The bot utilizes an Ehlers Volumetric Trend (EVT) strategy as its foundation.

The script automates the following:

1.  **Directory Structure:** Creates a clean project folder (`bybit_evt_strategy`).
2.  **File Population:** Generates the necessary Python files (`config_models.py`, `neon_logger.py`, `bybit_utils.py`, `bybit_helper_functions.py`, `indicators.py`, `ehlers_volumetric_strategy.py`, `main.py`) populated with the provided code.
3.  **Configuration Files:** Creates essential configuration files (`.env`, `.gitignore`, `requirements.txt`).
4.  **Documentation:** Generates a basic `README.md`.
5.  **Version Control:** Initializes a new Git repository and creates an initial commit.

**Prerequisites:**

*   A Unix-like environment (Linux, macOS, WSL on Windows) with Bash shell.
*   Git installed.
*   Python 3.x installed (for running the bot later, not the script itself).

**Key Features of the Generated Project:**

*   **Robust Configuration:** Uses Pydantic for validating settings loaded from environment variables and a `.env` file.
*   **Modular Design:** Code is separated into logical modules (config, logging, utilities, helpers, indicators, strategy, main entry point).
*   **Strategy Foundation:** Provides a working base for the Ehlers Volumetric Trend strategy.
*   **Exchange Interaction:** Includes helper functions for interacting with the Bybit V5 API via the CCXT library.
*   **Git Ready:** The project is immediately ready for version control.

**Crucial Steps & Warnings:**

1.  <0xF0><0x9F><0x9A><0xA7> **Run Safely:** Execute this script **only** in the parent directory where you want the new `bybit_evt_strategy` project folder to be created. It operates within the sub-directory it creates.
2.  <0xE2><0x9A><0xA0><0xEF><0xB8><0x8F> **API Keys Security:** The generated `.env` file contains **PLACEHOLDER** API keys. **You ABSOLUTELY MUST edit `.env` and replace these placeholders with your actual Bybit API key and secret** before running the Python bot.
    *   Generate keys via the Bybit website (Testnet or Mainnet).
    *   Ensure keys have permissions for `Orders` and `Positions` (Read & Write) under the Unified Trading Account scope.
    *   **Never commit your actual API keys to version control.** The included `.gitignore` prevents committing `.env` by default.
3.  <0xF0><0x9F><0x9A><0xAB> **Overwriting Prevention:** The script checks if the target project directory (`bybit_evt_strategy`) already exists and will exit if it does to prevent accidental data loss. Remove or rename any existing directory with that name first.
4.  <0xF0><0x9F><0xAA><0xB0> **Virtual Environment:** It is **highly recommended** to create and activate a Python virtual environment within the generated project directory *before* installing dependencies. This isolates project packages. Instructions are provided in the script's output and the README.
5.  <0xF0><0x9F><0x93><0x9D> **Git Configuration:** The script initializes a local Git repository. It includes commented-out commands in its final output showing how to set your Git user name and email specifically for this repository if needed (useful if it differs from your global Git config).
6.  <0xE2><0x9A><0xA1><0xEF><0xB8><0x8F> **Remote Repository:** The script does *not* automatically create or link to a remote repository (like on GitHub or GitLab). You will need to create one manually on your preferred platform and then follow the example commands (provided in the script output) to link your local repository and push the initial commit.

---

### **The Bash Script (`create_project.sh`)**

```bash
#!/bin/bash
# Script to create the directory structure, populate files for the Bybit EVT strategy bot,
# and initialize a new Git repository.

# --- Safety Settings ---
# Exit immediately if a command exits with a non-zero status. Crucial for preventing partial setup on error.
set -e
# Treat unset variables as an error when substituting.
# set -u # Be cautious with set -u, especially if sourcing other scripts or relying on potentially unset env vars.

# --- Configuration ---
PROJECT_DIR="bybit_evt_strategy"
# These GIT variables are primarily used for generating instructions in the final output.
GIT_USER_NAME="YourGitHubUsername" # Replace with your GitHub/GitLab username
GIT_USER_EMAIL="your_email@example.com" # Replace with your actual email used for Git

# --- ANSI Color Codes for Output ---
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_INFO='\033[36m'    # Cyan
C_SUCCESS='\033[32m' # Green
C_WARN='\033[33m'    # Yellow
C_ERROR='\033[91m'   # Bright Red

# --- Pre-flight Checks ---
echo -e "${C_INFO}${C_BOLD}Starting Project Setup: ${PROJECT_DIR}${C_RESET}"

# Check if Git is installed
if ! command -v git &> /dev/null; then
  echo -e "${C_ERROR}Error: 'git' command not found. Please install Git.${C_RESET}"
  exit 1
fi

# Safety Check: Prevent overwriting existing directory
if [ -d "$PROJECT_DIR" ]; then
  echo -e "${C_ERROR}Error: Directory '${PROJECT_DIR}' already exists in the current location.${C_RESET}"
  echo -e "${C_WARN}Please remove or rename the existing directory before running this script.${C_RESET}"
  exit 1
fi

# --- Directory Creation ---
echo -e "${C_INFO}Creating project directory: ${PROJECT_DIR}${C_RESET}"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR" # Change into the project directory for subsequent file creation

echo -e "${C_INFO}Creating Python source files...${C_RESET}"

# --- Create config_models.py ---
echo -e "${C_INFO} -> Generating config_models.py${C_RESET}"
cat << 'EOF' > config_models.py
# config_models.py
"""
Pydantic Models for Application Configuration using pydantic-settings.

Loads configuration from environment variables and/or a .env file.
Provides type validation and default values for the trading bot.
"""

import logging
import os # For environment variable access during load
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    PositiveInt,
    PositiveFloat,
    NonNegativeInt,
    FilePath, # Consider if needed later
    DirectoryPath, # Consider if needed later
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Import Enums for type safety ---
# Attempt to import from bybit_helpers, provide fallback Literals if needed during setup
# This ensures the script can generate the file even if dependencies aren't installed yet.
try:
    # Ensure bybit_helper_functions.py exists and is importable when this runs
    # If running this script standalone before dependencies are set up, this might fail.
    from bybit_helper_functions import (
        PositionIdx,
        Category,
        OrderFilter,
        Side,
        TimeInForce,
        TriggerBy,
        TriggerDirection,
    )
    ENUM_IMPORT_SUCCESS = True
except ImportError:
    print(
        "Warning [config_models]: Could not import Enums from bybit_helper_functions. Using basic types/Literals as fallback."
    )
    # Define basic types or Literals as placeholders - MUST match the expected values
    PositionIdx = Literal[0, 1, 2] # 0: One-Way, 1: Hedge Buy, 2: Hedge Sell
    Category = Literal["linear", "inverse", "spot", "option"]
    OrderFilter = Literal["Order", "StopOrder", "tpslOrder", "TakeProfit", "StopLoss"] # Add others as needed
    Side = Literal["Buy", "Sell"] # Bybit V5 often uses Capitalized Buy/Sell in responses/some params
    TimeInForce = Literal["GTC", "IOC", "FOK", "PostOnly"]
    TriggerBy = Literal["LastPrice", "MarkPrice", "IndexPrice"]
    TriggerDirection = Literal[1, 2] # 1: Rise, 2: Fall
    ENUM_IMPORT_SUCCESS = False

class APIConfig(BaseModel):
    """Configuration for Bybit API Connection and Market Defaults."""
    exchange_id: Literal["bybit"] = Field("bybit", description="CCXT Exchange ID")
    api_key: Optional[str] = Field(None, description="Bybit API Key (Loaded from env/file)")
    api_secret: Optional[str] = Field(None, description="Bybit API Secret (Loaded from env/file)")
    testnet_mode: bool = Field(True, description="Use Bybit Testnet environment")
    default_recv_window: PositiveInt = Field(
        10000, ge=100, le=60000, description="API request validity window in milliseconds (100-60000)"
    )

    # Market & Symbol Defaults
    symbol: str = Field(..., description="Primary trading symbol (e.g., BTC/USDT:USDT)")
    usdt_symbol: str = Field("USDT", description="Quote currency for balance reporting (usually USDT)")
    expected_market_type: Literal["swap", "spot", "option", "future"] = Field(
        "swap", description="Expected CCXT market type for validation (e.g., 'swap')"
    )
    expected_market_logic: Literal["linear", "inverse"] = Field(
        "linear", description="Expected market logic for derivative validation ('linear' or 'inverse')"
    )

    # Retry & Rate Limit Defaults
    retry_count: NonNegativeInt = Field(
        3, description="Default number of retries for API calls (0 disables retries)"
    )
    retry_delay_seconds: PositiveFloat = Field(
        2.0, gt=0, description="Default base delay (seconds) for API retries"
    )

    # Fee Rates (Important for accurate calculations, check Bybit for your tier)
    maker_fee_rate: Decimal = Field(
        Decimal("0.0002"), ge=0, description="Maker fee rate (e.g., 0.0002 for 0.02%)"
    )
    taker_fee_rate: Decimal = Field(
        Decimal("0.00055"), ge=0, description="Taker fee rate (e.g., 0.00055 for 0.055%)"
    )

    # Order Execution Defaults & Helpers
    default_slippage_pct: Decimal = Field(
        Decimal("0.005"), # 0.5%
        gt=0,
        le=Decimal("0.1"), # Max 10% sanity check
        description="Default max allowable slippage % for market order checks vs OB (e.g., 0.005 for 0.5%)",
    )
    position_qty_epsilon: Decimal = Field(
        Decimal("1e-9"), # Small value for comparing position sizes near zero
        gt=0,
        description="Small value for floating point quantity comparisons (e.g., treating size < epsilon as zero)",
    )
    shallow_ob_fetch_depth: PositiveInt = Field(
        5, ge=1, le=50, description="Order book depth for quick spread/slippage check (e.g., 5)"
    )
    order_book_fetch_limit: PositiveInt = Field(
        25, ge=1, le=1000, description="Default depth for fetching L2 order book (e.g., 25, 50)"
    )

    # Position/Side Constants (Internal use, ensures consistency)
    pos_none: Literal["NONE"] = "NONE"
    pos_long: Literal["LONG"] = "LONG"
    pos_short: Literal["SHORT"] = "SHORT"
    side_buy: Literal["Buy"] = "Buy" # Match Bybit V5 standard Side
    side_sell: Literal["Sell"] = "Sell" # Match Bybit V5 standard Side

    @field_validator('api_key', 'api_secret', mode='before') # mode='before' to catch env var directly
    @classmethod
    def check_not_placeholder(cls, v: Optional[str], info) -> Optional[str]:
        """Warns if API key/secret looks like a placeholder."""
        if v and isinstance(v, str) and "PLACEHOLDER" in v.upper():
             # Use print for early warnings before logger might be set up
             print(f"\033[93mWarning [APIConfig]: Field '{info.field_name}' appears to be a placeholder: '{v[:15]}...'\033[0m")
        return v

    @field_validator('symbol', mode='before')
    @classmethod
    def check_and_format_symbol(cls, v: Any) -> str:
        """Validates and standardizes the symbol format."""
        if not isinstance(v, str) or not v:
             raise ValueError("Symbol must be a non-empty string")
        # Basic check for common derivative format (e.g., BTC/USDT:USDT) or spot (BTC/USDT)
        if ":" not in v and "/" not in v:
            raise ValueError(f"Invalid symbol format: '{v}'. Expected format like 'BTC/USDT:USDT' or 'BTC/USDT'.")
        # CCXT typically uses uppercase symbols
        return v.strip().upper()

    @model_validator(mode='after')
    def check_api_keys_presence(self) -> 'APIConfig':
        """Checks if API keys are present if not in testnet mode (optional check)."""
        # This check is optional, as some operations might be public-only.
        # Enable if private endpoints are always required for your bot's core function.
        # if not self.testnet_mode and (not self.api_key or not self.api_secret):
        #     print("\033[91mCRITICAL [APIConfig]: API Key and Secret are required for mainnet operation.\033[0m")
        #     # Consider raising ValueError here if keys are strictly mandatory
        return self


class IndicatorSettings(BaseModel):
    """Parameters for Technical Indicator Calculations."""
    min_data_periods: PositiveInt = Field(
        100, ge=20, description="Minimum historical candles needed for reliable indicator calculations (e.g., 100)"
    )
    # Ehlers Volumetric specific
    evt_length: PositiveInt = Field(
        7, gt=1, description="Period length for EVT indicator (must be > 1)"
    )
    evt_multiplier: PositiveFloat = Field(
        2.5, gt=0, description="Multiplier for EVT bands calculation (must be > 0)"
    )
    # ATR specific (often used for stop loss)
    atr_period: PositiveInt = Field(
        14, gt=0, description="Period length for ATR indicator (must be > 0)"
    )
    # Add other indicator parameters here if needed
    # rsi_period: PositiveInt = Field(14, ...)
    # macd_fast: PositiveInt = Field(12, ...)


class AnalysisFlags(BaseModel):
    """Flags to Enable/Disable Specific Indicator Calculations or Features."""
    use_evt: bool = Field(True, description="Enable Ehlers Volumetric Trend calculation and signaling")
    use_atr: bool = Field(True, description="Enable ATR calculation (primarily for stop loss)")
    # Add other flags here
    # use_rsi: bool = Field(False, ...)
    # use_macd: bool = Field(False, ...)


class StrategyConfig(BaseModel):
    """Core Strategy Behavior and Parameters."""
    name: str = Field(..., description="Name of the strategy instance (used in logs, potentially broker IDs)")
    timeframe: str = Field("15m", pattern=r"^\d+[mhdMy]$", description="Candlestick timeframe (e.g., '1m', '5m', '1h', '4h', '1d')")
    polling_interval_seconds: PositiveInt = Field(
        60, ge=5, description="Frequency (seconds) to fetch data and check signals (min 5s)"
    )
    leverage: PositiveInt = Field(
        5, ge=1, description="Desired leverage for the symbol (check exchange limits, 1 means no leverage)"
    )
    # Use type annotation directly if possible, fallback to Literal if import failed
    position_idx: Union[PositionIdx, Literal[0, 1, 2]] = Field(
        0, # Default to One-Way (0)
        description="Position mode (0: One-Way, 1: Hedge Buy, 2: Hedge Sell)"
    )
    risk_per_trade: Decimal = Field(
        Decimal("0.01"), # 1%
        gt=0,
        le=Decimal("0.1"), # Max 10% risk sanity check
        description="Fraction of available balance to risk per trade (e.g., 0.01 for 1%)",
    )
    stop_loss_atr_multiplier: Decimal = Field(
        Decimal("2.0"),
        gt=0,
        description="ATR multiplier for stop loss distance (must be > 0 if ATR is used)"
    )
    indicator_settings: IndicatorSettings = Field(default_factory=IndicatorSettings)
    analysis_flags: AnalysisFlags = Field(default_factory=AnalysisFlags)

    # Strategy specific flag (can be redundant if analysis_flags.use_evt is the primary control)
    EVT_ENABLED: bool = Field(
        True, description="Confirms EVT logic is the core driver (should match analysis_flags.use_evt)"
    )

    @field_validator('timeframe')
    @classmethod
    def check_timeframe_format(cls, v: str) -> str:
        # Basic validation, CCXT handles more complex cases
        import re
        if not re.match(r"^\d+[mhdMy]$", v):
             raise ValueError(f"Invalid timeframe format: '{v}'. Use formats like '1m', '15m', '1h', '4h', '1d'.")
        return v

    @model_validator(mode='after')
    def check_feature_consistency(self) -> 'StrategyConfig':
        """Ensures related configuration flags are consistent."""
        if self.EVT_ENABLED != self.analysis_flags.use_evt:
             print(f"\033[93mWarning [StrategyConfig]: Mismatch between 'EVT_ENABLED' ({self.EVT_ENABLED}) and 'analysis_flags.use_evt' ({self.analysis_flags.use_evt}). Ensure consistency.\033[0m")
             # Optionally raise ValueError if they MUST match
             # raise ValueError("'EVT_ENABLED' must match 'analysis_flags.use_evt'")

        if self.stop_loss_atr_multiplier > 0 and not self.analysis_flags.use_atr:
            raise ValueError("'stop_loss_atr_multiplier' > 0 requires 'analysis_flags.use_atr' to be True.")

        # Ensure position_idx is valid even if enum import failed
        if not ENUM_IMPORT_SUCCESS and self.position_idx not in [0, 1, 2]:
             raise ValueError(f"Invalid position_idx: {self.position_idx}. Must be 0, 1, or 2.")

        return self


class LoggingConfig(BaseModel):
    """Configuration for the Logger Setup."""
    logger_name: str = Field("TradingBot", description="Name for the logger instance")
    log_file: Optional[str] = Field("trading_bot.log", description="Path to the log file (relative or absolute). Set to None or empty string to disable file logging.")
    # Use standard logging level names
    console_level_str: Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="Logging level for console output"
    )
    file_level_str: Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        "DEBUG", description="Logging level for file output (if enabled)"
    )
    log_rotation_bytes: NonNegativeInt = Field(
        5 * 1024 * 1024, # 5 MB
        description="Max log file size in bytes before rotating (0 disables rotation)"
    )
    log_backup_count: NonNegativeInt = Field(
        5, description="Number of backup log files to keep (requires rotation enabled)"
    )
    third_party_log_level_str: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "WARNING", description="Log level for noisy third-party libraries (e.g., ccxt, websockets)"
    )

    @field_validator('log_file', mode='before')
    @classmethod
    def validate_log_file(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v.strip() == "":
            return None # Explicitly return None if empty or None
        # Basic check for invalid characters (OS dependent, this is a simple example)
        if any(char in v for char in ['<', '>', ':', '"', '|', '?', '*']):
             raise ValueError(f"Log file path '{v}' contains invalid characters.")
        return v.strip()


class SMSConfig(BaseModel):
    """Configuration for SMS Alerting (e.g., via Termux or Twilio)."""
    enable_sms_alerts: bool = Field(False, description="Globally enable/disable SMS alerts")

    # Termux Specific
    use_termux_api: bool = Field(False, description="Use Termux:API for sending SMS (requires Termux app setup)")
    sms_recipient_number: Optional[str] = Field(None, pattern=r"^\+?[1-9]\d{1,14}$", description="Recipient phone number (E.164 format recommended, e.g., +11234567890)")
    sms_timeout_seconds: PositiveInt = Field(30, ge=5, le=120, description="Timeout for Termux API call (seconds)")

    # Add Twilio fields here if implementing Twilio support
    # use_twilio_api: bool = Field(False, ...)
    # twilio_account_sid: Optional[str] = Field(None, ...)
    # twilio_auth_token: Optional[str] = Field(None, ...)
    # twilio_from_number: Optional[str] = Field(None, ...)

    @model_validator(mode='after')
    def check_sms_provider_details(self) -> 'SMSConfig':
        """Validates that if SMS is enabled, a provider and necessary details are set."""
        if self.enable_sms_alerts:
            provider_configured = False
            if self.use_termux_api:
                if not self.sms_recipient_number:
                    raise ValueError("Termux SMS enabled, but 'sms_recipient_number' is missing.")
                provider_configured = True
            # --- Add check for Twilio if implemented ---
            # elif self.use_twilio_api:
            #     if not all([self.twilio_account_sid, self.twilio_auth_token, self.twilio_from_number, self.sms_recipient_number]):
            #         raise ValueError("Twilio SMS enabled, but required fields (SID, Token, From, Recipient) are missing.")
            #     provider_configured = True

            if not provider_configured:
                raise ValueError("SMS alerts enabled, but no provider (Termux/Twilio) is configured or required details are missing.")
        return self


class AppConfig(BaseSettings):
    """Master Configuration Model integrating all sub-configurations."""
    # Configure Pydantic-Settings behavior
    model_config = SettingsConfigDict(
        env_file='.env',                # Load from .env file in the current working directory
        env_nested_delimiter='__',      # Use double underscore for nested env vars (e.g., BOT_API_CONFIG__SYMBOL)
        env_prefix='BOT_',              # Prefix for environment variables (e.g., BOT_API_CONFIG__API_KEY)
        case_sensitive=False,           # Environment variables are case-insensitive
        extra='ignore',                 # Ignore extra fields not defined in the models
        validate_default=True,          # Validate default values
    )

    # Define the nested configuration models
    api_config: APIConfig = Field(default_factory=APIConfig)
    strategy_config: StrategyConfig = Field(default_factory=StrategyConfig)
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)
    sms_config: SMSConfig = Field(default_factory=SMSConfig)

    # Add top-level settings if needed, e.g.:
    # app_version: str = "1.0.0"


def load_config() -> AppConfig:
    """
    Loads the application configuration from environment variables and the .env file.

    Handles validation errors and provides informative messages. Exits on critical failure.

    Returns:
        AppConfig: The validated application configuration object.

    Raises:
        SystemExit: If configuration validation fails or a fatal error occurs.
    """
    try:
        print(f"\033[36mLoading configuration...\033[0m")
        # Determine the path to the .env file relative to this script file
        # This makes loading more robust regardless of where the script is run from.
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # env_file_path = os.path.join(script_dir, '.env')
        # If running main.py, cwd might be better if .env is placed there
        env_file_path = os.path.join(os.getcwd(), '.env')

        if os.path.exists(env_file_path):
            print(f"Attempting to load from: {env_file_path}")
            config = AppConfig(_env_file=env_file_path)
        else:
            print(f"'.env' file not found at {env_file_path}. Loading from environment variables only.")
            config = AppConfig() # Still loads from env vars even if file missing

        # Post-load checks/logging (using print as logger might not be ready)
        if config.api_config.api_key and "PLACEHOLDER" in config.api_config.api_key.upper():
             print("\033[91m\033[1mCRITICAL WARNING: API Key appears to be a placeholder. Bot will likely fail authentication.\033[0m")
        if config.api_config.api_secret and "PLACEHOLDER" in config.api_config.api_secret.upper():
            print("\033[91m\033[1mCRITICAL WARNING: API Secret appears to be a placeholder. Bot will likely fail authentication.\033[0m")
        if not config.api_config.testnet_mode:
             print("\033[93m\033[1mWARNING: Testnet mode is DISABLED. Bot will attempt LIVE trading.\033[0m")

        print(f"\033[32mConfiguration loaded successfully.\033[0m")
        # Optional: Print loaded config for debugging (be careful with secrets)
        # print("--- Loaded Config (Partial) ---")
        # print(f"  Symbol: {config.api_config.symbol}")
        # print(f"  Timeframe: {config.strategy_config.timeframe}")
        # print(f"  Testnet: {config.api_config.testnet_mode}")
        # print("-----------------------------")
        return config

    except ValidationError as e:
        print(f"\n{'-'*20}\033[91m CONFIGURATION VALIDATION FAILED \033[0m{'-'*20}")
        for error in e.errors():
            # Construct a user-friendly path to the error location
            loc_list = [str(item) for item in error['loc']]
            # Prepend BOT_ and convert to uppercase for env var name suggestion
            env_var_suggestion = "BOT_" + "__".join(loc_list).upper()
            loc_path = " -> ".join(loc_list) if loc_list else 'AppConfig'
            print(f"  \033[91mField:\033[0m {loc_path}")
            print(f"  \033[91mError:\033[0m {error['msg']}")
            err_input = error.get('input')
            if err_input is not None:
                # Avoid printing secrets if possible
                is_secret = any(secret_part in loc_path.lower() for secret_part in ['key', 'secret', 'token'])
                val_display = "*****" if is_secret and isinstance(err_input, str) else repr(err_input)
                print(f"  \033[91mValue:\033[0m {val_display}")
            print(f"  \033[93mSuggestion:\033[0m Check env var '{env_var_suggestion}' or the corresponding field in '.env'.")
            print("-" * 25)
        print(f"{'-'*60}\n")
        raise SystemExit("\033[91mConfiguration validation failed. Please check your '.env' file or environment variables.\033[0m")

    except Exception as e:
        print(f"\033[91m\033[1mFATAL: Unexpected error loading configuration: {e}\033[0m")
        import traceback
        traceback.print_exc()
        raise SystemExit("\033[91mFailed to load configuration due to an unexpected error.\033[0m")

# Example of how to load config in the main script:
if __name__ == "__main__":
    # This block executes only when config_models.py is run directly
    # Useful for testing the configuration loading independently
    print("Running config_models.py directly for testing...")
    try:
        app_settings = load_config()
        print("\n\033[1mLoaded Config (JSON Representation):\033[0m")
        # Use model_dump_json for Pydantic v2
        # Be cautious about printing secrets - might need custom serialization to exclude them
        print(app_settings.model_dump_json(indent=2))
        print("\n\033[32mConfiguration test successful.\033[0m")
    except SystemExit as e:
         print(f"\n\033[91mExiting due to configuration error during test: {e}\033[0m")
    except Exception as e:
         print(f"\n\033[91mAn unexpected error occurred during the configuration test: {e}\033[0m")
         import traceback
         traceback.print_exc()
EOF

# --- Create neon_logger.py ---
echo -e "${C_INFO} -> Generating neon_logger.py${C_RESET}"
cat << 'EOF' > neon_logger.py
#!/usr/bin/env python
"""Neon Logger Setup (v1.4) - Enhanced Robustness & Pydantic Integration

Provides a function `setup_logger` to configure a Python logger instance with:
- Colorized console output using a "neon" theme via colorama (TTY only).
- Uses a custom Formatter for cleaner color handling and padding.
- Clean, non-colorized file output.
- Optional log file rotation (size-based).
- Comprehensive log formatting (timestamp, level, name, function, line, thread).
- Custom SUCCESS log level (25).
- Configuration driven by a Pydantic `LoggingConfig` model.
- Option to control verbosity of common third-party libraries.
- Improved error handling during setup.
"""

import logging
import logging.handlers
import os
import sys
from typing import Any, Literal, Optional

# --- Import Pydantic model for config type hinting ---
try:
    from config_models import LoggingConfig
except ImportError:
    print("FATAL [neon_logger]: Could not import LoggingConfig from config_models.py.", file=sys.stderr)
    # Define a fallback simple structure if needed for basic operation, though setup will likely fail later.
    class LoggingConfig: # type: ignore
        logger_name: str = "FallbackLogger"
        log_file: Optional[str] = "fallback.log"
        console_level_str: str = "INFO"
        file_level_str: str = "DEBUG"
        log_rotation_bytes: int = 0
        log_backup_count: int = 0
        third_party_log_level_str: str = "WARNING"
    # sys.exit(1) # Or exit if LoggingConfig is absolutely essential

# --- Attempt to import colorama ---
try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init

    # Initialize colorama (autoreset=True ensures colors reset after each print)
    # On Windows, init() is necessary; on Linux/macOS, it might not be strictly required
    # but doesn't hurt. strip=False prevents stripping codes if output is redirected.
    colorama_init(autoreset=True, strip=False)
    COLORAMA_AVAILABLE = True
except ImportError:
    # Define dummy color objects if colorama is not installed
    class DummyColor:
        def __getattr__(self, name: str) -> str:
            return ""  # Return empty string for any attribute access

    Fore = DummyColor()
    Back = DummyColor()
    Style = DummyColor()
    COLORAMA_AVAILABLE = False
    # Warning printed by setup_logger if colors are expected but unavailable

# --- Custom Log Level ---
SUCCESS_LEVEL_NUM = 25  # Between INFO (20) and WARNING (30)
SUCCESS_LEVEL_NAME = "SUCCESS"
logging.addLevelName(SUCCESS_LEVEL_NUM, SUCCESS_LEVEL_NAME)

# Type hint for the logger method we are adding
if sys.version_info >= (3, 8):
    from typing import Protocol
    class LoggerWithSuccess(Protocol):
        def success(self, message: str, *args: Any, **kwargs: Any) -> None: ...
else:
    LoggerWithSuccess = logging.Logger # Fallback for older Python

def log_success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Adds a custom 'success' log method to the Logger instance."""
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        # pylint: disable=protected-access
        self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)

# Add the method to the Logger class dynamically if it doesn't exist
# This avoids potential issues if the script is run multiple times in the same process
if not hasattr(logging.Logger, SUCCESS_LEVEL_NAME.lower()):
    setattr(logging.Logger, SUCCESS_LEVEL_NAME.lower(), log_success)


# --- Neon Color Theme Mapping ---
# Ensure all standard levels and the custom SUCCESS level are included
LOG_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.BLUE + Style.BRIGHT,
    SUCCESS_LEVEL_NUM: Fore.MAGENTA + Style.BRIGHT, # Make success stand out
    logging.WARNING: Fore.YELLOW + Style.BRIGHT,
    logging.ERROR: Fore.RED + Style.BRIGHT,
    logging.CRITICAL: f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}",
}
DEFAULT_COLOR = Fore.WHITE # Default for levels not explicitly mapped


# --- Custom Formatter for Colored Console Output ---
class ColoredConsoleFormatter(logging.Formatter):
    """A custom logging formatter that adds colors to console output based on log level,
    only if colorama is available and output is detected as a TTY (terminal).
    Handles level name padding correctly within color codes.
    """
    # Define format string components for easier modification
    LOG_FORMAT_BASE = "%(asctime)s - %(name)s - {levelname_placeholder} [%(threadName)s:%(funcName)s:%(lineno)d] - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LEVELNAME_WIDTH = 9 # Width for the padded level name (e.g., "WARNING  ")

    def __init__(
        self,
        *, # Force keyword arguments for clarity
        use_colors: Optional[bool] = None, # Allow overriding color detection
        **kwargs: Any
    ):
        # Determine if colors should be used
        if use_colors is None:
            self.use_colors = COLORAMA_AVAILABLE and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        else:
            self.use_colors = use_colors and COLORAMA_AVAILABLE

        # Dynamically create the format string with or without color placeholders
        levelname_fmt = "%(levelname)s" # Placeholder to be replaced in format()
        fmt = self.LOG_FORMAT_BASE.format(levelname_placeholder=levelname_fmt)

        # Initialize the parent Formatter
        super().__init__(fmt=fmt, datefmt=self.LOG_DATE_FORMAT, style='%', **kwargs) # type: ignore # Pylint/MyPy confusion on style

        if not COLORAMA_AVAILABLE and self.use_colors:
             # This case should ideally not happen if use_colors=None (default)
             print("\033[93mWarning [Logger]: Colorama not found, but color usage was requested. Console logs will be monochrome.\033[0m", file=sys.stderr)
             self.use_colors = False
        elif not self.use_colors and COLORAMA_AVAILABLE:
             # Inform user if colors are available but disabled (e.g., redirected output)
             if use_colors is None: # Only print if auto-detected off
                print("\033[94mInfo [Logger]: Console output is not a TTY or colorama disabled. Logs will be monochrome.\033[0m", file=sys.stderr)

    def format(self, record: logging.LogRecord) -> str:
        """Formats the record, applying colors and padding to the level name."""
        # Get the color for the record's level, default if not found
        level_color = LOG_LEVEL_COLORS.get(record.levelno, DEFAULT_COLOR)

        # Store original levelname, apply padding and color (if enabled)
        original_levelname = record.levelname
        padded_levelname = f"{original_levelname:<{self.LEVELNAME_WIDTH}}" # Pad to fixed width

        if self.use_colors:
            # Apply color codes around the padded level name
            record.levelname = f"{level_color}{padded_levelname}{Style.RESET_ALL}"
        else:
            # Use the padded level name without color codes
            record.levelname = padded_levelname

        # Let the parent class handle the rest of the formatting
        formatted_message = super().format(record)

        # Restore the original levelname on the record object in case it's used elsewhere
        record.levelname = original_levelname

        return formatted_message


# --- Log Format Strings (for File Handler) ---
# Use a standard format without color codes for files
FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)-9s [%(threadName)s:%(funcName)s:%(lineno)d] - %(message)s" # Pad levelname
FILE_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- Create Formatters (instantiate only once for efficiency) ---
# Use the custom formatter for the console
console_formatter = ColoredConsoleFormatter()
# Use a standard formatter for the file
file_formatter = logging.Formatter(FILE_LOG_FORMAT, datefmt=FILE_LOG_DATE_FORMAT)


# --- Main Setup Function ---
def setup_logger(
    config: LoggingConfig,
    propagate: bool = False # Whether to allow messages to propagate to the root logger
) -> LoggerWithSuccess: # Return type hint includes the custom .success() method
    """
    Sets up and configures a logger instance based on the provided LoggingConfig.

    Args:
        config: A validated LoggingConfig Pydantic model instance.
        propagate: If True, messages logged to this logger will also be passed to
                   handlers of ancestor loggers (usually the root logger). Default is False.

    Returns:
        The configured logging.Logger instance (typed as LoggerWithSuccess).
    """
    # --- Validate Log Levels from Config ---
    try:
        console_level = logging.getLevelName(config.console_level_str.upper())
        file_level = logging.getLevelName(config.file_level_str.upper())
        third_party_level = logging.getLevelName(config.third_party_log_level_str.upper())

        # Ensure getLevelName returned valid integer levels
        if not isinstance(console_level, int):
            print(f"\033[93mWarning [Logger]: Invalid console log level '{config.console_level_str}'. Defaulting to INFO.\033[0m", file=sys.stderr)
            console_level = logging.INFO
        if not isinstance(file_level, int):
            print(f"\033[93mWarning [Logger]: Invalid file log level '{config.file_level_str}'. Defaulting to DEBUG.\033[0m", file=sys.stderr)
            file_level = logging.DEBUG
        if not isinstance(third_party_level, int):
            print(f"\033[93mWarning [Logger]: Invalid third-party log level '{config.third_party_log_level_str}'. Defaulting to WARNING.\033[0m", file=sys.stderr)
            third_party_level = logging.WARNING

    except Exception as e:
        # Fallback in case of unexpected errors during level processing
        print(f"\033[91mFATAL [Logger]: Error processing log levels from config: {e}. Using defaults (INFO, DEBUG, WARNING).\033[0m", file=sys.stderr)
        console_level, file_level, third_party_level = logging.INFO, logging.DEBUG, logging.WARNING

    # --- Get Logger Instance ---
    logger = logging.getLogger(config.logger_name)
    # Set the logger's effective level to the lowest of its handlers (DEBUG allows all messages to reach handlers)
    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate

    # --- Clear Existing Handlers (optional but recommended for reconfiguration) ---
    if logger.hasHandlers():
        print(f"\033[94mInfo [Logger]: Logger '{config.logger_name}' already has handlers. Clearing them to reconfigure.\033[0m", file=sys.stderr)
        for handler in logger.handlers[:]: # Iterate over a copy
            try:
                handler.close()
                logger.removeHandler(handler)
            except Exception as e_close:
                print(f"\033[93mWarning [Logger]: Error removing/closing handler {handler}: {e_close}\033[0m", file=sys.stderr)

    # --- Console Handler ---
    try:
        console_h = logging.StreamHandler(sys.stdout)
        console_h.setLevel(console_level)
        console_h.setFormatter(console_formatter)
        logger.addHandler(console_h)
        print(f"\033[94m[Logger] Console logging active: Level=[{logging.getLevelName(console_level)}]\033[0m")
    except Exception as e_console:
        print(f"\033[91mError [Logger]: Failed to set up console handler: {e_console}\033[0m", file=sys.stderr)

    # --- File Handler (Optional) ---
    if config.log_file:
        try:
            # Ensure log directory exists
            log_file_path = os.path.abspath(config.log_file)
            log_dir = os.path.dirname(log_file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True) # exist_ok=True prevents error if dir exists

            # Choose between RotatingFileHandler and FileHandler based on config
            if config.log_rotation_bytes > 0 and config.log_backup_count >= 0:
                file_h = logging.handlers.RotatingFileHandler(
                    filename=log_file_path,
                    maxBytes=config.log_rotation_bytes,
                    backupCount=config.log_backup_count,
                    encoding="utf-8"
                )
                log_type = "Rotating"
                log_details = f"(Max: {config.log_rotation_bytes / (1024*1024):.1f} MB, Backups: {config.log_backup_count})"
            else:
                file_h = logging.FileHandler(log_file_path, mode="a", encoding="utf-8") # Append mode
                log_type = "Basic"
                log_details = "(Rotation disabled)"

            file_h.setLevel(file_level)
            file_h.setFormatter(file_formatter) # Use the non-colored file formatter
            logger.addHandler(file_h)
            print(f"\033[94m[Logger] {log_type} file logging active: Level=[{logging.getLevelName(file_level)}] File='{log_file_path}' {log_details}\033[0m")

        except OSError as e_os:
            # Handle file system errors (e.g., permission denied)
            print(f"\033[91mFATAL [Logger]: OS Error configuring log file '{config.log_file}': {e_os}. File logging disabled.\033[0m", file=sys.stderr)
        except Exception as e_file:
            # Handle other unexpected errors during file handler setup
            print(f"\033[91mError [Logger]: Unexpected error setting up file logging: {e_file}. File logging disabled.\033[0m", file=sys.stderr)
    else:
        print("\033[94m[Logger] File logging disabled by configuration.\033[0m")

    # --- Configure Third-Party Log Levels ---
    # List of common noisy libraries
    noisy_libs = [
        "ccxt", "ccxt.base", "ccxt.async_support", # CCXT core and async
        "urllib3", "requests", # Underlying HTTP libraries
        "asyncio", # Can be verbose in debug mode
        "websockets", # WebSocket library
    ]
    print(f"\033[94m[Logger] Setting third-party library log level: [{logging.getLevelName(third_party_level)}]\033[0m")
    for lib_name in noisy_libs:
        try:
            lib_logger = logging.getLogger(lib_name)
            if lib_logger:
                lib_logger.setLevel(third_party_level)
                lib_logger.propagate = False # Prevent noisy libs from logging to root/our handlers
        except Exception as e_lib:
            # Non-critical if setting a specific library level fails
            print(f"\033[93mWarning [Logger]: Could not set log level for library '{lib_name}': {e_lib}\033[0m", file=sys.stderr)

    # Cast the logger to the type hint that includes the .success method
    return logger # type: ignore

# Example Usage (within main script)
# if __name__ == "__main__":
#     # Create a dummy config for testing
#     test_config = LoggingConfig(
#         logger_name="TestLogger",
#         log_file="test_logger.log",
#         console_level_str="DEBUG",
#         file_level_str="DEBUG",
#         third_party_log_level_str="ERROR"
#     )
#     try:
#         test_logger = setup_logger(test_config)
#         test_logger.debug("This is a debug message.")
#         test_logger.info("This is an info message.")
#         test_logger.success("This is a success message!") # Use the custom method
#         test_logger.warning("This is a warning message.")
#         test_logger.error("This is an error message.")
#         test_logger.critical("This is a critical message!")
#         print(f"Test log file created at: {os.path.abspath('test_logger.log')}")
#     except Exception as e:
#         print(f"Error during logger test: {e}")
EOF

# --- Create bybit_utils.py ---
echo -e "${C_INFO} -> Generating bybit_utils.py${C_RESET}"
cat << 'EOF' > bybit_utils.py
# bybit_utils.py
"""
Utility functions supporting the Bybit trading bot framework.

Includes:
- Safe data conversions (e.g., to Decimal).
- Formatting helpers for prices, amounts, order IDs.
- SMS alerting functionality (currently Termux-based).
- Asynchronous retry decorator for robust API calls.
"""

import functools
import logging
import subprocess  # For Termux API call
import time
import asyncio
import sys
from collections.abc import Callable, Coroutine
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN, getcontext
from typing import Any, TypeVar, Optional, Union

# --- Import Pydantic models for type hinting ---
try:
    from config_models import AppConfig, SMSConfig, APIConfig # Use the unified config
except ImportError:
    print("FATAL ERROR [bybit_utils]: Could not import config_models. Check file presence.", file=sys.stderr)
    # Define fallback structures or exit if config is critical
    class DummyConfig: pass
    AppConfig = SMSConfig = APIConfig = DummyConfig # type: ignore
    # sys.exit(1)

# --- Attempt to import CCXT ---
# Define common CCXT exceptions even if import fails, so retry decorator can compile
class DummyExchangeError(Exception): pass
class DummyNetworkError(DummyExchangeError): pass
class DummyRateLimitExceeded(DummyExchangeError): pass
class DummyExchangeNotAvailable(DummyNetworkError): pass
class DummyRequestTimeout(DummyNetworkError): pass
class DummyAuthenticationError(DummyExchangeError): pass

try:
    import ccxt
    import ccxt.async_support as ccxt_async # Alias for async usage if needed
    from ccxt.base.errors import (
        ExchangeError, NetworkError, RateLimitExceeded, ExchangeNotAvailable,
        RequestTimeout, AuthenticationError
        # Import others like OrderNotFound, InvalidOrder etc. as needed
    )
    CCXT_AVAILABLE = True
except ImportError:
    print("\033[91mFATAL ERROR [bybit_utils]: CCXT library not found. Install with 'pip install ccxt'\033[0m", file=sys.stderr)
    ccxt = None # Set to None to allow checking later
    ccxt_async = None
    # Assign dummy classes to names expected by retry decorator
    ExchangeError = DummyExchangeError # type: ignore
    NetworkError = DummyNetworkError # type: ignore
    RateLimitExceeded = DummyRateLimitExceeded # type: ignore
    ExchangeNotAvailable = DummyExchangeNotAvailable # type: ignore
    RequestTimeout = DummyRequestTimeout # type: ignore
    AuthenticationError = DummyAuthenticationError # type: ignore
    CCXT_AVAILABLE = False
    # Consider sys.exit(1) if CCXT is absolutely essential for this module

# --- Attempt to import colorama ---
try:
    from colorama import Back, Fore, Style
    from colorama import init as colorama_init
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor() # type: ignore
    COLORAMA_AVAILABLE = False
    # No warning here, handled by logger setup if needed

# --- Logger Setup ---
# Get logger configured in the main script.
# Ensures consistency in logging format and handlers.
logger = logging.getLogger(__name__)

# --- Decimal Precision ---
# Set precision for Decimal context (adjust as needed)
getcontext().prec = 28

# --- Utility Functions ---

def safe_decimal_conversion(
    value: Any, default: Optional[Decimal] = None, context: str = ""
) -> Optional[Decimal]:
    """
    Safely convert various inputs (string, int, float, Decimal) to Decimal.

    Args:
        value: The value to convert.
        default: The value to return if conversion fails or input is None. Defaults to None.
        context: Optional string describing the source of the value for logging.

    Returns:
        The converted Decimal, or the default value. Logs a warning on failure.
    """
    if value is None:
        return default
    try:
        # Convert float to string first to avoid potential precision issues
        if isinstance(value, float):
            value = str(value)
        d = Decimal(value)
        # Check for NaN (Not a Number) or Infinity, which are valid Decimal states but often unwanted
        if d.is_nan() or d.is_infinite():
            logger.warning(f"[safe_decimal] Converted '{value}' to {d}{' in '+context if context else ''}. Returning default.")
            return default
        return d
    except (ValueError, TypeError, InvalidOperation) as e:
        logger.warning(f"[safe_decimal] Failed to convert '{value}' (type: {type(value).__name__}) to Decimal{' in '+context if context else ''}: {e}. Returning default.")
        return default
    except Exception as e_unexpected:
        logger.error(f"[safe_decimal] Unexpected error converting '{value}'{' in '+context if context else ''}: {e_unexpected}", exc_info=True)
        return default

def format_price(exchange: Optional[ccxt.Exchange], symbol: str, price: Any) -> Optional[str]:
    """
    Format a price value according to the market's precision rules using CCXT.

    Args:
        exchange: The CCXT exchange instance (must be initialized).
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        price: The price value to format (can be str, int, float, Decimal).

    Returns:
        The formatted price string, or None if input price is None, or "Error" on failure.
    """
    if price is None:
        return None
    if not exchange or not CCXT_AVAILABLE:
        logger.warning("[format_price] CCXT Exchange instance unavailable. Returning raw string.")
        return str(price)

    price_dec = safe_decimal_conversion(price, context=f"format_price for {symbol}")
    if price_dec is None:
        logger.error(f"[format_price] Invalid price value '{price}' for {symbol}. Cannot format.")
        return "Error" # Indicate failure clearly

    try:
        # Use CCXT's built-in method
        return exchange.price_to_precision(symbol, float(price_dec))
    except (AttributeError, KeyError, ccxt.ExchangeError) as e_ccxt:
        # Fallback if price_to_precision fails or market data incomplete
        logger.debug(f"[format_price] CCXT price_to_precision failed for '{symbol}': {e_ccxt}. Attempting manual fallback.")
        try:
            market = exchange.market(symbol) # Throws BadSymbol if not found
            if market and 'precision' in market and 'price' in market['precision']:
                tick_size = Decimal(str(market['precision']['price']))
                if tick_size <= 0: raise ValueError("Tick size must be positive")
                # Quantize to the tick size using appropriate rounding for prices (often ROUND_HALF_UP)
                formatted = price_dec.quantize(tick_size, rounding=ROUND_HALF_UP)
                # Determine number of decimal places from tick size exponent for string formatting
                decimal_places = abs(tick_size.normalize().as_tuple().exponent)
                return f"{formatted:.{decimal_places}f}"
            else:
                logger.warning(f"[format_price] Market precision data missing for '{symbol}'. Using default 8dp formatting.")
                return f"{price_dec:.8f}" # Default fallback formatting
        except (ccxt.BadSymbol, ValueError, InvalidOperation, Exception) as format_err:
            logger.error(f"[format_price] Fallback formatting failed for {symbol}: {format_err}", exc_info=False)
            return f"{price_dec:.8f}" # Last resort formatting
    except Exception as e:
        logger.error(f"[format_price] Unexpected error formatting price '{price}' for {symbol}: {e}", exc_info=True)
        return "Error"

def format_amount(exchange: Optional[ccxt.Exchange], symbol: str, amount: Any) -> Optional[str]:
    """
    Format an amount (quantity) value according to the market's precision rules using CCXT.

    Args:
        exchange: The CCXT exchange instance (must be initialized).
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        amount: The amount value to format (can be str, int, float, Decimal).

    Returns:
        The formatted amount string, or None if input amount is None, or "Error" on failure.
    """
    if amount is None:
        return None
    if not exchange or not CCXT_AVAILABLE:
        logger.warning("[format_amount] CCXT Exchange instance unavailable. Returning raw string.")
        return str(amount)

    amount_dec = safe_decimal_conversion(amount, context=f"format_amount for {symbol}")
    if amount_dec is None:
        logger.error(f"[format_amount] Invalid amount value '{amount}' for {symbol}. Cannot format.")
        return "Error"

    try:
        # Use CCXT's built-in method
        return exchange.amount_to_precision(symbol, float(amount_dec))
    except (AttributeError, KeyError, ccxt.ExchangeError) as e_ccxt:
        # Fallback if amount_to_precision fails or market data incomplete
        logger.debug(f"[format_amount] CCXT amount_to_precision failed for '{symbol}': {e_ccxt}. Attempting manual fallback.")
        try:
            market = exchange.market(symbol) # Throws BadSymbol if not found
            if market and 'precision' in market and 'amount' in market['precision']:
                 step_size = Decimal(str(market['precision']['amount']))
                 if step_size <= 0: raise ValueError("Step size must be positive")
                 # Quantize to the step size using ROUND_DOWN for amounts (common requirement)
                 formatted = amount_dec.quantize(step_size, rounding=ROUND_DOWN)
                 decimal_places = abs(step_size.normalize().as_tuple().exponent)
                 return f"{formatted:.{decimal_places}f}"
            else:
                 logger.warning(f"[format_amount] Market precision data missing for '{symbol}'. Using default 8dp formatting.")
                 return f"{amount_dec:.8f}" # Default fallback formatting
        except (ccxt.BadSymbol, ValueError, InvalidOperation, Exception) as format_err:
            logger.error(f"[format_amount] Fallback formatting failed for {symbol}: {format_err}", exc_info=False)
            return f"{amount_dec:.8f}" # Last resort formatting
    except Exception as e:
        logger.error(f"[format_amount] Unexpected error formatting amount '{amount}' for {symbol}: {e}", exc_info=True)
        return "Error"

def format_order_id(order_id: Optional[Union[str, int]]) -> str:
    """
    Format an order ID for concise logging (shows first/last parts).

    Args:
        order_id: The order ID (string or integer).

    Returns:
        A formatted string (e.g., "1234...5678") or "N/A" or "UNKNOWN".
    """
    if order_id is None:
        return "N/A"
    try:
        id_str = str(order_id).strip()
        if not id_str: return "N/A"
        # Adjust length thresholds as needed
        if len(id_str) <= 10:
            return id_str
        else:
            # Show first 4 and last 4 characters for longer IDs
            return f"{id_str[:4]}...{id_str[-4:]}"
    except Exception as e:
        logger.error(f"Error formatting order ID '{order_id}': {e}")
        return "UNKNOWN"

def send_sms_alert(message: str, sms_config: SMSConfig) -> bool:
    """
    Send an SMS alert using the configured method (currently Termux).

    Args:
        message: The text message content.
        sms_config: The validated SMSConfig object containing settings.

    Returns:
        True if the alert was sent successfully (or if alerts are disabled),
        False if sending failed or configuration was invalid.
    """
    # Check if SMS alerts are globally disabled
    if not sms_config.enable_sms_alerts:
        logger.debug(f"SMS suppressed (globally disabled): {message[:80]}...")
        return True # Return True as no action was required/failed

    recipient = sms_config.sms_recipient_number
    if not recipient:
        logger.warning("SMS alert requested, but no recipient number configured in SMSConfig.")
        return False

    # --- Termux API Method ---
    if sms_config.use_termux_api:
        timeout = sms_config.sms_timeout_seconds
        try:
            logger.info(f"Attempting Termux SMS to {recipient} (Timeout: {timeout}s)...")
            # Ensure message is treated as a single argument, handle potential quotes/special chars if necessary
            command = ["termux-sms-send", "-n", recipient, message]

            # Using subprocess.run for simplicity (can be blocking if called from async code without run_in_executor)
            # If calling from an async context, wrap this in loop.run_in_executor
            result = subprocess.run(
                command,
                timeout=timeout,
                check=True,          # Raise CalledProcessError on non-zero exit code
                capture_output=True, # Capture stdout/stderr
                text=True,           # Decode stdout/stderr as text
                encoding='utf-8'     # Explicitly set encoding
            )
            # Log success with output (if any)
            success_msg = f"{Fore.GREEN}Termux SMS Sent OK to {recipient}.{Style.RESET_ALL}"
            output_log = result.stdout.strip()
            if output_log: success_msg += f" Output: {output_log}"
            logger.info(success_msg)
            return True
        except FileNotFoundError:
            logger.error(f"{Fore.RED}Termux command 'termux-sms-send' not found. Is Termux:API installed and accessible?{Style.RESET_ALL}")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"{Fore.RED}Termux SMS timed out after {timeout} seconds for recipient {recipient}.{Style.RESET_ALL}")
            return False
        except subprocess.CalledProcessError as e:
            # Log detailed error including exit code and stderr
            stderr_output = e.stderr.strip() if e.stderr else "(No stderr)"
            logger.error(f"{Fore.RED}Termux SMS failed (Exit Code: {e.returncode}) for recipient {recipient}. Error: {stderr_output}{Style.RESET_ALL}")
            return False
        except Exception as e:
            # Catch any other unexpected errors during subprocess execution
            logger.critical(f"{Fore.RED}Unexpected error during Termux SMS execution: {e}{Style.RESET_ALL}", exc_info=True)
            return False

    # --- Twilio API Method (Placeholder) ---
    # elif sms_config.use_twilio_api:
    #     logger.warning("Twilio SMS sending is not implemented in this version.")
    #     # Add Twilio client logic here if implemented
    #     # from twilio.rest import Client
    #     # try:
    #     #     client = Client(sms_config.twilio_account_sid, sms_config.twilio_auth_token)
    #     #     message = client.messages.create(
    #     #         body=message,
    #     #         from_=sms_config.twilio_from_number,
    #     #         to=recipient
    #     #     )
    #     #     logger.info(f"Twilio SMS Sent OK (SID: {message.sid})")
    #     #     return True
    #     # except Exception as e:
    #     #     logger.error(f"Twilio SMS failed: {e}")
    #     #     return False
    #     return False # Return False until implemented

    else:
        # This case should ideally be caught by SMSConfig validation, but double-check
        logger.error("SMS alerts enabled, but no valid provider (Termux/Twilio) is configured or active.")
        return False

# --- Asynchronous Retry Decorator Factory ---
# Type variable for the decorated function's return type
T = TypeVar("T")

# Default exceptions to handle (only if CCXT is available)
_DEFAULT_HANDLED_EXCEPTIONS = (
    RateLimitExceeded,
    NetworkError, # Includes RequestTimeout, ExchangeNotAvailable, etc.
    # Add other potentially transient CCXT errors if needed
) if CCXT_AVAILABLE else () # Empty tuple if CCXT failed import

def retry_api_call(
    max_retries_override: Optional[int] = None,
    initial_delay_override: Optional[float] = None,
    handled_exceptions: tuple = _DEFAULT_HANDLED_EXCEPTIONS,
    error_message_prefix: str = "API Call Failed",
    # Add specific exception-delay multipliers if needed:
    # delay_multipliers: Optional[Dict[Type[Exception], float]] = None
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """
    Decorator factory to automatically retry ASYNCHRONOUS API calls with exponential backoff.

    Requires the decorated async function (or its caller) to pass an `AppConfig` instance
    either as a positional argument or a keyword argument named `app_config`.

    Args:
        max_retries_override: Specific number of retries for this call, overriding config.
        initial_delay_override: Specific initial delay, overriding config.
        handled_exceptions: Tuple of exception types to catch and retry. Defaults to common
                            CCXT network/rate limit errors.
        error_message_prefix: String to prefix log messages on failure/retry.

    Returns:
        A decorator for async functions.
    """
    if not handled_exceptions:
        # If CCXT failed or no exceptions provided, log a warning - decorator won't retry.
        logger.warning("[retry_api_call] No handled_exceptions defined. Decorator will not retry on errors.")

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # --- Find AppConfig ---
            app_config: Optional[AppConfig] = kwargs.get("app_config")
            if not isinstance(app_config, AppConfig):
                # Search positional arguments if not found in kwargs
                app_config = next((arg for arg in args if isinstance(arg, AppConfig)), None)

            # Check if AppConfig was found
            func_name_log = func.__name__
            if not isinstance(app_config, AppConfig):
                # Critical failure if config is missing
                logger.critical(f"{Back.RED}{Fore.WHITE}FATAL: @retry_api_call in '{func_name_log}' requires AppConfig instance in args or kwargs.{Style.RESET_ALL}")
                raise ValueError(f"AppConfig instance not provided to decorated function {func_name_log}")

            api_conf: APIConfig = app_config.api_config # Extract API config for easier access

            # Determine effective retry parameters
            effective_max_retries = max_retries_override if max_retries_override is not None else api_conf.retry_count
            effective_base_delay = initial_delay_override if initial_delay_override is not None else api_conf.retry_delay_seconds

            last_exception: Optional[Exception] = None

            # Retry loop: initial call + number of retries
            for attempt in range(effective_max_retries + 1):
                try:
                    # Log retry attempts
                    if attempt > 0:
                        logger.debug(f"Retrying {func_name_log} (Attempt {attempt + 1}/{effective_max_retries + 1})")

                    # Execute the wrapped asynchronous function
                    result = await func(*args, **kwargs)
                    return result # Return successfully

                # --- Catch exceptions specified for retry ---
                except handled_exceptions as e:
                    last_exception = e
                    # Check if this was the last attempt
                    if attempt == effective_max_retries:
                        logger.error(f"{Fore.RED}{error_message_prefix}: Max retries ({effective_max_retries + 1}) reached for {func_name_log}. Last error: {type(e).__name__} - {e}{Style.RESET_ALL}")
                        # Trigger alert on final failure
                        asyncio.create_task(send_sms_alert_async(f"ALERT: Max retries failed for {func_name_log} ({type(e).__name__})", app_config.sms_config))
                        raise e # Re-raise the last exception

                    # Calculate delay with exponential backoff + jitter
                    delay = (effective_base_delay * (2 ** attempt)) + (effective_base_delay * random.uniform(0.1, 0.5))

                    # Log specific error types differently
                    log_level, log_color = logging.WARNING, Fore.YELLOW
                    if isinstance(e, RateLimitExceeded):
                        log_color = Fore.YELLOW + Style.BRIGHT
                        # CCXT might provide a 'retry_after' value in seconds
                        retry_after_sec = getattr(e, 'retry_after', None)
                        if retry_after_sec:
                             suggested_delay = float(retry_after_sec) + 1.0 # Add a small buffer
                             delay = max(delay, suggested_delay) # Use the longer delay
                             logger.warning(f"{log_color}Rate limit hit in {func_name_log}. API suggests retry after {retry_after_sec}s. Retrying in {delay:.2f}s... Error: {e}{Style.RESET_ALL}")
                        else:
                             logger.warning(f"{log_color}Rate limit hit in {func_name_log}. Retrying in {delay:.2f}s... Error: {e}{Style.RESET_ALL}")
                    elif isinstance(e, (NetworkError, RequestTimeout, ExchangeNotAvailable)):
                        log_level, log_color = logging.ERROR, Fore.RED
                        delay = max(delay, 5.0) # Ensure a minimum delay for network issues
                        logger.log(log_level, f"{log_color}Network/Timeout/Unavailable in {func_name_log}. Retrying in {delay:.2f}s... Error: {e}{Style.RESET_ALL}")
                    else:
                        # Generic handled exception
                        logger.log(log_level, f"{log_color}Handled exception {type(e).__name__} in {func_name_log}. Retrying in {delay:.2f}s... Error: {e}{Style.RESET_ALL}")

                    # Wait asynchronously before the next attempt
                    await asyncio.sleep(delay)

                # --- Catch other unexpected exceptions ---
                except Exception as e_unhandled:
                    # Log critical errors and re-raise immediately - do not retry unhandled exceptions.
                    logger.critical(f"{Back.RED}{Fore.WHITE}UNEXPECTED error in {func_name_log}: {type(e_unhandled).__name__} - {e_unhandled}{Style.RESET_ALL}", exc_info=True)
                    # Trigger alert for critical errors
                    asyncio.create_task(send_sms_alert_async(f"CRITICAL ERROR in {func_name_log}: {type(e_unhandled).__name__}", app_config.sms_config))
                    raise e_unhandled # Re-raise the unhandled exception

            # This point should theoretically not be reached if the loop logic is correct
            # If it is reached, it means the loop finished without returning or raising.
            if last_exception:
                 logger.error(f"Retry loop for {func_name_log} finished unexpectedly after error: {last_exception}")
                 raise last_exception
            else:
                 # Should be impossible if max_retries >= 0
                 msg = f"Retry loop for {func_name_log} finished unexpectedly without success or error."
                 logger.critical(msg)
                 raise RuntimeError(msg)

        return wrapper
    return decorator

async def send_sms_alert_async(message: str, sms_config: SMSConfig):
    """Asynchronous wrapper for send_sms_alert using run_in_executor."""
    if not sms_config.enable_sms_alerts:
        return # Don't bother scheduling if disabled
    try:
        loop = asyncio.get_running_loop()
        # Run the potentially blocking subprocess call in a separate thread
        await loop.run_in_executor(
            None, # Use default executor (ThreadPoolExecutor)
            functools.partial(send_sms_alert, message, sms_config)
        )
    except RuntimeError as e:
         logger.error(f"Failed to get running loop for async SMS: {e}")
    except Exception as e:
        logger.error(f"Error dispatching async SMS alert: {e}", exc_info=True)

# Example usage of the decorator:
# @retry_api_call(max_retries_override=2, error_message_prefix="Fetch Balance Failed")
# async def fetch_balance_with_retry(exchange: ccxt.Exchange, app_config: AppConfig):
#     # Must accept app_config
#     # ... implementation ...
#     balance = await exchange.fetch_balance()
#     return balance
EOF

# --- Create indicators.py ---
echo -e "${C_INFO} -> Generating indicators.py${C_RESET}"
cat << 'EOF' > indicators.py
#!/usr/bin/env python
"""Technical Indicators Module (v1.2)

Provides functions to calculate various technical indicators using pandas DataFrames
containing OHLCV data. Leverages the `pandas_ta` library for common indicators
and includes custom implementations like Ehlers Volumetric Trend (EVT).

Designed to be driven by configuration passed via an `AppConfig` object.
"""

import logging
import sys
from typing import Any, Dict, Optional, Tuple # Removed List, Tuple as return types simplified

import numpy as np
import pandas as pd

# --- Import pandas_ta ---
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    print(
        "\033[91mFATAL ERROR [indicators]: 'pandas_ta' library not found.\033[0m"
        "\033[93mPlease install it: pip install pandas_ta\033[0m"
    )
    # Set flag and allow module to load, but calculations requiring it will fail.
    PANDAS_TA_AVAILABLE = False
    # Consider sys.exit(1) if pandas_ta is absolutely essential for any operation.

# --- Import Pydantic models for config type hinting ---
try:
    from config_models import AppConfig, IndicatorSettings, AnalysisFlags
except ImportError:
    print("FATAL [indicators]: Could not import from config_models.py.", file=sys.stderr)
    # Define fallback structures or exit
    class DummyConfig: pass
    AppConfig = IndicatorSettings = AnalysisFlags = DummyConfig # type: ignore
    # sys.exit(1)


# --- Setup ---
logger = logging.getLogger(__name__) # Get logger configured in main script

# --- Constants ---
# Define standard column names expected in input DataFrames
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"
COL_VOLUME = "volume"
REQUIRED_OHLCV_COLS = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]


# --- Helper Functions ---
# (Numpy and pandas_ta handle most float operations efficiently)

def _validate_dataframe(df: pd.DataFrame, min_rows: int, required_cols: list[str]) -> bool:
    """Helper to validate DataFrame input for indicator calculations."""
    func_name = sys._getframe(1).f_code.co_name # Get caller function name
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.error(f"[{func_name}] Input DataFrame is None or empty.")
        return False
    if not all(col in df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        logger.error(f"[{func_name}] Missing required columns: {missing}.")
        return False
    # Check for sufficient non-NaN rows in required columns
    valid_rows = len(df.dropna(subset=required_cols))
    if valid_rows < min_rows:
        logger.warning(f"[{func_name}] Insufficient valid data rows ({valid_rows}) for calculation (minimum required: {min_rows}). Results may be inaccurate or NaN.")
        # Allow calculation to proceed but warn the user. Strict mode could return False here.
    return True


# --- Pivot Point Calculations (Standard & Fibonacci) ---
# Note: These typically use the *previous* period's OHLC to calculate levels for the *current* or *next* period.
# The main indicator function applies them to the entire DataFrame history if needed.

def calculate_standard_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculates standard pivot points based on H, L, C values."""
    # Basic validation
    if not all(isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]):
        logger.debug("Standard Pivots skipped: Invalid H/L/C input.")
        return {}
    if low > high:
        logger.warning(f"Standard Pivots: Low ({low}) > High ({high}). Check input data.")
        # Optional: Swap them: low, high = high, low
        # Or return empty: return {}

    pivots = {}
    try:
        pivot = (high + low + close) / 3.0
        pivots["PP"] = pivot # Pivot Point
        pivots["S1"] = (2 * pivot) - high # Support 1
        pivots["R1"] = (2 * pivot) - low  # Resistance 1
        pivots["S2"] = pivot - (high - low) # Support 2
        pivots["R2"] = pivot + (high - low) # Resistance 2
        pivots["S3"] = low - 2 * (high - pivot) # Support 3
        pivots["R3"] = high + 2 * (pivot - low) # Resistance 3
    except Exception as e:
        logger.error(f"Error calculating standard pivots: {e}", exc_info=False)
        return {}
    return pivots

def calculate_fib_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculates Fibonacci pivot points based on H, L, C values."""
    # Basic validation
    if not all(isinstance(v, (int, float)) and not np.isnan(v) for v in [high, low, close]):
        logger.debug("Fibonacci Pivots skipped: Invalid H/L/C input.")
        return {}
    if low > high:
        logger.warning(f"Fibonacci Pivots: Low ({low}) > High ({high}). Check input data.")
        # Optional: Swap or return empty

    fib_pivots = {}
    try:
        pivot = (high + low + close) / 3.0
        fib_range = high - low

        fib_pivots["PP"] = pivot # Pivot Point (same as standard)
        # Handle potential zero range
        if abs(fib_range) < 1e-9: # Use tolerance for float comparison
            logger.debug("Fibonacci Pivots: Range is near zero. Only PP calculated.")
            return fib_pivots

        # Fibonacci Levels (Common Ratios: 0.382, 0.618, 1.000)
        fib_pivots["S1"] = pivot - (0.382 * fib_range) # Fib Support 1
        fib_pivots["R1"] = pivot + (0.382 * fib_range) # Fib Resistance 1
        fib_pivots["S2"] = pivot - (0.618 * fib_range) # Fib Support 2
        fib_pivots["R2"] = pivot + (0.618 * fib_range) # Fib Resistance 2
        fib_pivots["S3"] = pivot - (1.000 * fib_range) # Fib Support 3 (Often low - range)
        fib_pivots["R3"] = pivot + (1.000 * fib_range) # Fib Resistance 3 (Often high + range)
    except Exception as e:
        logger.error(f"Error calculating Fibonacci pivots: {e}", exc_info=False)
        return {}
    return fib_pivots

# --- Support / Resistance Level Calculation (Example) ---
# This is a simplified example; robust S/R often involves more complex analysis (e.g., peak/trough detection).
def calculate_levels(df_period: pd.DataFrame, current_price: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculates various potential support/resistance levels based on historical data (e.g., pivots, fib retracements).
    Note: This is a basic example; robust S/R requires more advanced techniques.
    """
    levels: Dict[str, Any] = {
        "support": {}, "resistance": {}, "pivot": None,
        "fib_retracements": {}, "standard_pivots": {}, "fib_pivots": {}
    }
    if not _validate_dataframe(df_period, min_rows=2, required_cols=[COL_HIGH, COL_LOW, COL_CLOSE]):
        logger.debug("Levels calculation skipped: Invalid DataFrame or insufficient rows.")
        return levels

    try:
        # --- Calculate Pivots for the *next* period based on the *last* period in the df ---
        # Use iloc[-1] for the most recent completed period's data
        last_high = df_period[COL_HIGH].iloc[-1]
        last_low = df_period[COL_LOW].iloc[-1]
        last_close = df_period[COL_CLOSE].iloc[-1]

        standard_pivots = calculate_standard_pivot_points(last_high, last_low, last_close)
        if standard_pivots:
            levels["standard_pivots"] = standard_pivots
            levels["pivot"] = standard_pivots.get("PP") # Assign main pivot

        fib_pivots = calculate_fib_pivot_points(last_high, last_low, last_close)
        if fib_pivots:
            levels["fib_pivots"] = fib_pivots
            if levels["pivot"] is None: # Use Fib Pivot if standard wasn't calculated
                levels["pivot"] = fib_pivots.get("PP")

        # --- Calculate Fibonacci Retracements over the *entire* df_period range ---
        period_high = df_period[COL_HIGH].max()
        period_low = df_period[COL_LOW].min()
        period_diff = period_high - period_low

        if abs(period_diff) > 1e-9: # Avoid division by zero or meaningless retracements
            levels["fib_retracements"] = {
                "High": period_high,
                "Fib 78.6%": period_low + period_diff * 0.786, # Common Fib level
                "Fib 61.8%": period_low + period_diff * 0.618, # Golden Ratio conjugate
                "Fib 50.0%": period_low + period_diff * 0.5,   # Midpoint
                "Fib 38.2%": period_low + period_diff * 0.382, # Golden Ratio conjugate
                "Fib 23.6%": period_low + period_diff * 0.236, # Common Fib level
                "Low": period_low,
            }
        else:
            logger.debug("Fib Retracements skipped: Period range is near zero.")

        # --- Classify Levels as Support/Resistance based on Current Price ---
        if current_price is not None and isinstance(current_price, (int, float)):
            # Combine all calculated levels into one dictionary for easier iteration
            all_potential_levels = {
                **{f"Std {k}": v for k, v in levels["standard_pivots"].items()},
                **{f"FibPiv {k}": v for k, v in levels["fib_pivots"].items() if k != "PP"}, # Avoid duplicate PP
                **levels["fib_retracements"]
            }
            for label, value in all_potential_levels.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if value < current_price:
                        levels["support"][label] = value
                    elif value > current_price:
                        levels["resistance"][label] = value
                    # else: level is very close to current price, could be either or ignore

    except IndexError:
        logger.warning("IndexError calculating levels (likely insufficient data rows). Some levels might be missing.")
    except Exception as e:
        logger.error(f"Error calculating S/R levels: {e}", exc_info=True)

    # Sort S/R levels for easier reading/use
    levels["support"] = dict(sorted(levels["support"].items(), key=lambda item: item[1], reverse=True)) # Highest first
    levels["resistance"] = dict(sorted(levels["resistance"].items(), key=lambda item: item[1])) # Lowest first
    return levels


# --- Custom Indicator Implementations ---

def calculate_vwma(close: pd.Series, volume: pd.Series, length: int) -> Optional[pd.Series]:
    """Calculates Volume Weighted Moving Average (VWMA)."""
    if not isinstance(close, pd.Series) or not isinstance(volume, pd.Series): return None
    if close.empty or volume.empty: return None
    if len(close) != len(volume): logger.error("VWMA Error: Close and Volume series lengths differ."); return None
    if length <= 0: logger.error(f"VWMA Error: Invalid length ({length})."); return None
    if len(close) < length: logger.debug(f"VWMA Debug: Data length ({len(close)}) < period ({length}). Result will have NaNs."); # Allow calculation

    try:
        pv = close * volume
        cumulative_pv = pv.rolling(window=length, min_periods=length).sum() # Ensure full window sum
        cumulative_vol = volume.rolling(window=length, min_periods=length).sum()

        # Avoid division by zero or NaN volumes
        # Replace 0 volume with NaN to prevent division errors and propagate NaN result
        vwma = cumulative_pv / cumulative_vol.replace(0, np.nan)
        vwma.name = f"VWMA_{length}"
        return vwma
    except Exception as e:
        logger.error(f"Error calculating VWMA(len={length}): {e}", exc_info=True)
        return None

def ehlers_volumetric_trend(df: pd.DataFrame, length: int, multiplier: float) -> pd.DataFrame:
    """
    Calculates the Ehlers Volumetric Trend (EVT) indicator.

    This indicator uses a Volume Weighted Moving Average (VWMA) smoothed with a
    SuperSmoother filter. Trend direction is determined by comparing the smoothed
    VWMA to its previous value multiplied by a factor related to the multiplier.

    Adds columns:
        - `vwma_{length}`: Raw VWMA.
        - `smooth_vwma_{length}`: SuperSmoother applied to VWMA.
        - `evt_trend_{length}`: Trend direction (1 for up, -1 for down, 0 for neutral/transition).
        - `evt_buy_{length}`: Boolean signal, True when trend flips from non-1 to 1.
        - `evt_sell_{length}`: Boolean signal, True when trend flips from non_-1 to -1.

    Args:
        df: DataFrame with 'close' and 'volume' columns.
        length: The period length for VWMA and SuperSmoother (e.g., 7).
        multiplier: Multiplier for trend bands (e.g., 2.5 for 2.5%).

    Returns:
        The original DataFrame with EVT columns added. Returns original df on failure.
    """
    func_name = "ehlers_volumetric_trend"
    if not _validate_dataframe(df, min_rows=length + 2, required_cols=[COL_CLOSE, COL_VOLUME]): # Need length + 2 for smoother calc
        logger.warning(f"[{func_name}] Input validation failed. Skipping calculation.")
        return df # Return original df

    if length <= 1: logger.error(f"[{func_name}] Invalid length ({length}). Must be > 1."); return df
    if multiplier <= 0: logger.error(f"[{func_name}] Invalid multiplier ({multiplier}). Must be > 0."); return df

    df_out = df.copy()
    vwma_col = f"vwma_{length}"
    smooth_col = f"smooth_vwma_{length}"
    trend_col = f"evt_trend_{length}"
    buy_col = f"evt_buy_{length}"
    sell_col = f"evt_sell_{length}"

    try:
        # 1. Calculate VWMA
        vwma = calculate_vwma(df_out[COL_CLOSE], df_out[COL_VOLUME], length=length)
        if vwma is None or vwma.isnull().all():
            raise ValueError(f"VWMA calculation failed or resulted in all NaNs for length {length}.")
        df_out[vwma_col] = vwma

        # Fill initial NaNs in VWMA if needed for smoother start (optional, consider implications)
        # df_out[vwma_col] = df_out[vwma_col].fillna(method='bfill') # Or some other strategy

        # 2. Apply SuperSmoother Filter to VWMA
        # SuperSmoother constants (derived from Ehlers' work)
        a = np.exp(-1.414 * np.pi / length)
        b = 2 * a * np.cos(1.414 * np.pi / length)
        c2 = b
        c3 = -a * a
        c1 = 1 - c2 - c3

        # Initialize smoothed series with NaNs
        smoothed = pd.Series(np.nan, index=df_out.index, dtype=float)
        vwma_vals = df_out[vwma_col].values # Access numpy array for potential speedup

        # Iterate to calculate smoothed values (requires previous 2 values)
        for i in range(2, len(df_out)):
            # Check if current VWMA and previous two smoothed values (or VWMA as fallback) are valid
            if pd.notna(vwma_vals[i]):
                 # Use previous smoothed value if available, otherwise fallback to previous VWMA
                 sm1 = smoothed.iloc[i-1] if pd.notna(smoothed.iloc[i-1]) else vwma_vals[i-1]
                 sm2 = smoothed.iloc[i-2] if pd.notna(smoothed.iloc[i-2]) else vwma_vals[i-2]
                 # Ensure fallbacks are valid numbers
                 if pd.notna(sm1) and pd.notna(sm2):
                     smoothed.iloc[i] = c1 * vwma_vals[i] + c2 * sm1 + c3 * sm2

        df_out[smooth_col] = smoothed

        # 3. Determine Trend based on smoothed VWMA changes
        mult_factor_high = 1.0 + (multiplier / 100.0)
        mult_factor_low = 1.0 - (multiplier / 100.0)

        shifted_smooth = df_out[smooth_col].shift(1)

        # Conditions for trend change
        trend_up_condition = (df_out[smooth_col] > shifted_smooth * mult_factor_high)
        trend_down_condition = (df_out[smooth_col] < shifted_smooth * mult_factor_low)

        # Initialize trend series (0 = neutral/no trend)
        trend = pd.Series(0, index=df_out.index, dtype=int)
        trend[trend_up_condition] = 1   # Mark potential uptrend start
        trend[trend_down_condition] = -1 # Mark potential downtrend start

        # Forward fill the trend signal (once a trend starts, it persists until a counter-signal)
        # Replace 0s with NaN before ffill so only actual 1/-1 signals propagate
        trend = trend.replace(0, np.nan).ffill().fillna(0).astype(int) # Fill remaining NaNs (at start) with 0
        df_out[trend_col] = trend

        # 4. Generate Buy/Sell Signals based on Trend *Changes*
        trend_shifted = df_out[trend_col].shift(1, fill_value=0) # Previous period's trend

        # Buy signal: Trend flips from non-up (0 or -1) to up (1)
        df_out[buy_col] = (df_out[trend_col] == 1) & (trend_shifted != 1)
        # Sell signal: Trend flips from non-down (0 or 1) to down (-1)
        df_out[sell_col] = (df_out[trend_col] == -1) & (trend_shifted != -1)

        logger.debug(f"[{func_name}] Calculation successful for length={length}, multiplier={multiplier}.")
        return df_out

    except Exception as e:
        logger.error(f"Error during {func_name}(len={length}, mult={multiplier}): {e}", exc_info=True)
        # Add NaN columns to indicate failure but maintain structure
        for col in [vwma_col, smooth_col, trend_col, buy_col, sell_col]:
            if col not in df_out.columns: df_out[col] = np.nan
        return df # Return the (partially) modified DataFrame


# --- Master Indicator Calculation Function ---
def calculate_all_indicators(df: pd.DataFrame, app_config: AppConfig) -> pd.DataFrame:
    """
    Calculates all enabled technical indicators based on the AppConfig.

    Args:
        df: Input DataFrame with OHLCV data. Index should be DatetimeIndex.
        app_config: The validated AppConfig object containing strategy and indicator settings.

    Returns:
        A DataFrame with the original data and calculated indicator columns added.
        Returns an empty DataFrame or the original DataFrame on critical failure.
    """
    func_name = "calculate_all_indicators"
    if df is None or df.empty:
        logger.error(f"[{func_name}] Input DataFrame is empty or None. Cannot calculate indicators.")
        return pd.DataFrame() # Return empty DF on critical input error

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning(f"[{func_name}] DataFrame index is not a DatetimeIndex. Indicator calculations might be affected or fail.")
        # Optionally attempt conversion: try: df.index = pd.to_datetime(df.index) except Exception: pass

    # Extract config components
    try:
        settings: IndicatorSettings = app_config.strategy_config.indicator_settings
        flags: AnalysisFlags = app_config.strategy_config.analysis_flags
        min_rows_needed = settings.min_data_periods
    except AttributeError as e:
        logger.critical(f"[{func_name}] Failed to access configuration from AppConfig: {e}. Cannot proceed.")
        return df # Return original df as config is missing

    # Validate DataFrame content and length
    if not _validate_dataframe(df, min_rows=min_rows_needed, required_cols=REQUIRED_OHLCV_COLS):
        logger.error(f"[{func_name}] Input DataFrame validation failed. Indicator calculation aborted.")
        # Depending on strictness, could return df or empty df
        return df # Return original df, allowing caller to handle potentially missing indicators

    df_out = df.copy() # Work on a copy to avoid modifying the original DataFrame

    logger.debug(f"[{func_name}] Calculating indicators. Flags: {flags.model_dump()}, Settings: {settings.model_dump()}")

    try:
        # --- Use pandas_ta for standard indicators ---
        if PANDAS_TA_AVAILABLE:
            if flags.use_atr:
                if settings.atr_period > 0:
                    logger.debug(f"Calculating ATR(length={settings.atr_period})")
                    # pandas_ta automatically appends column named like 'ATRr_14'
                    df_out.ta.atr(length=settings.atr_period, append=True)
                else:
                    logger.warning("ATR calculation skipped: atr_period <= 0 in settings.")

            # Add other pandas_ta indicators based on flags:
            # if flags.use_rsi:
            #     if settings.rsi_period > 0:
            #         logger.debug(f"Calculating RSI(length={settings.rsi_period})")
            #         df_out.ta.rsi(length=settings.rsi_period, append=True) # Appends 'RSI_14'
            #     else: logger.warning("RSI skipped: rsi_period <= 0")
            #
            # if flags.use_macd:
            #      # MACD requires fast, slow, signal periods
            #      if all(p > 0 for p in [settings.macd_fast, settings.macd_slow, settings.macd_signal]):
            #           logger.debug(f"Calculating MACD(fast={settings.macd_fast}, slow={settings.macd_slow}, signal={settings.macd_signal})")
            #           df_out.ta.macd(fast=settings.macd_fast, slow=settings.macd_slow, signal=settings.macd_signal, append=True) # Appends MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            #      else: logger.warning("MACD skipped: Invalid periods.")

        else:
            logger.warning(f"[{func_name}] pandas_ta library not available. Skipping standard indicator calculations (ATR, RSI, MACD, etc.).")

        # --- Calculate Custom Indicators ---
        if flags.use_evt:
            logger.debug(f"Calculating Ehlers Volumetric Trend (len={settings.evt_length}, mult={settings.evt_multiplier})")
            # Ensure multiplier is float for calculation
            df_out = ehlers_volumetric_trend(df_out, settings.evt_length, float(settings.evt_multiplier))

        # --- Post-Calculation Processing ---
        # Remove potential duplicate columns if calculations appended existing names (less common with pandas_ta)
        df_out = df_out.loc[:, ~df_out.columns.duplicated()]

        # Optional: Log NaN count in final indicator columns for monitoring
        final_cols = df_out.columns.difference(df.columns) # Get newly added columns
        nan_counts = df_out[final_cols].isnull().sum()
        nan_summary = nan_counts[nan_counts > 0]
        if not nan_summary.empty: logger.debug(f"[{func_name}] NaN counts in new indicator columns:\n{nan_summary.to_string()}")

    except Exception as e:
        logger.error(f"[{func_name}] Unexpected error during indicator calculation: {e}", exc_info=True)
        # Return the DataFrame as it was before the error, possibly partially calculated
        return df_out

    logger.debug(f"[{func_name}] Indicator calculation finished. DataFrame shape: {df_out.shape}")
    return df_out
EOF

# --- Create bybit_helper_functions.py ---
echo -e "${C_INFO} -> Generating bybit_helper_functions.py${C_RESET}"
cat << 'EOF' > bybit_helper_functions.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bybit V5 CCXT Helper Functions (v3.5 - Pydantic Config & Enhanced)

Collection of asynchronous helper functions for interacting with the Bybit V5 API
using the CCXT library. Integrates tightly with Pydantic models defined in
`config_models.py` for configuration and validation. Provides robust error
handling, retries, and commonly needed operations like fetching data, placing orders,
managing positions, and handling WebSocket connections (optional).
"""

# Standard Library Imports
import asyncio
import json
import logging
import math
import os
import random
import sys
import time
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, DivisionByZero, InvalidOperation, getcontext
from enum import Enum
from typing import (Any, Coroutine, Dict, List, Literal, Optional, Sequence,
                    Tuple, TypeVar, Union, Callable) # Removed TypedDict for simplicity

# --- Import Pydantic models first ---
try:
    from config_models import APIConfig, AppConfig # Use the unified Pydantic config
except ImportError:
    print("FATAL [bybit_helpers]: Could not import from config_models.py.", file=sys.stderr)
    class DummyConfig: pass
    AppConfig = APIConfig = DummyConfig # type: ignore
    # sys.exit(1)

# Third-party Libraries
try:
    import ccxt
    import ccxt.async_support as ccxt_async # Use async version
    from ccxt.base.errors import (
        ArgumentsRequired, AuthenticationError, BadSymbol, DDoSProtection,
        ExchangeError, ExchangeNotAvailable, InsufficientFunds, InvalidNonce,
        InvalidOrder, NetworkError, NotSupported, OrderImmediatelyFillable,
        OrderNotFound, RateLimitExceeded, RequestTimeout
    )
    # For precise rounding in specific cases (less common now with built-in methods)
    from ccxt.base.decimal_to_precision import ROUND_UP, ROUND_DOWN as CCXT_ROUND_DOWN

    CCXT_AVAILABLE = True
except ImportError:
    print("\033[91mFATAL ERROR [bybit_helpers]: CCXT library not found. Install with 'pip install ccxt'\033[0m", file=sys.stderr)
    # Define dummy exceptions and classes if CCXT is missing
    class DummyExchangeError(Exception): pass
    class DummyNetworkError(DummyExchangeError): pass
    class DummyRateLimitExceeded(DummyExchangeError): pass
    class DummyAuthenticationError(DummyExchangeError): pass
    class DummyOrderNotFound(DummyExchangeError): pass
    class DummyInvalidOrder(DummyExchangeError): pass
    class DummyInsufficientFunds(DummyExchangeError): pass
    class DummyExchangeNotAvailable(DummyNetworkError): pass
    class DummyRequestTimeout(DummyNetworkError): pass
    class DummyNotSupported(DummyExchangeError): pass
    class DummyOrderImmediatelyFillable(DummyInvalidOrder): pass
    class DummyBadSymbol(DummyExchangeError): pass
    class DummyArgumentsRequired(DummyExchangeError): pass
    class DummyDDoSProtection(DummyExchangeError): pass
    class DummyInvalidNonce(DummyAuthenticationError): pass

    ccxt = None; ccxt_async = None # type: ignore
    # Assign dummy exceptions to the names used in the code
    ExchangeError = DummyExchangeError; NetworkError = DummyNetworkError # type: ignore
    RateLimitExceeded = DummyRateLimitExceeded; AuthenticationError = DummyAuthenticationError # type: ignore
    OrderNotFound = DummyOrderNotFound; InvalidOrder = DummyInvalidOrder # type: ignore
    InsufficientFunds = DummyInsufficientFunds; ExchangeNotAvailable = DummyExchangeNotAvailable # type: ignore
    RequestTimeout = DummyRequestTimeout; NotSupported = DummyNotSupported # type: ignore
    OrderImmediatelyFillable = DummyOrderImmediatelyFillable; BadSymbol = DummyBadSymbol # type: ignore
    ArgumentsRequired = DummyArgumentsRequired; DDoSProtection = DummyDDoSProtection # type: ignore
    InvalidNonce = DummyInvalidNonce; ROUND_UP = 'ROUND_UP'; CCXT_ROUND_DOWN = 'ROUND_DOWN' # type: ignore

    CCXT_AVAILABLE = False
    # sys.exit(1) # Exit if CCXT is absolutely required

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    # print("Warning [bybit_helpers]: pandas library not found. OHLCV data will be returned as lists.", file=sys.stderr)
    pd = None # type: ignore
    PANDAS_AVAILABLE = False

try:
    from colorama import Fore, Style, Back, init as colorama_init
    # Initialize colorama (required on Windows, safe on others)
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    class DummyColor:
        def __getattr__(self, name: str) -> str: return ""
    Fore = Style = Back = DummyColor() # type: ignore
    COLORAMA_AVAILABLE = False

# Optional WebSocket support
WEBSOCKETS_AVAILABLE = False
try:
    import websockets
    from websockets.exceptions import (WebSocketException, ConnectionClosed, ConnectionClosedOK,
                                       ConnectionClosedError, InvalidHandshake, InvalidURI,
                                       PayloadTooBig, ProtocolError)
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    # print("Warning [bybit_helpers]: websockets library not found. WebSocket features disabled.", file=sys.stderr)
    websockets = None # type: ignore
    # Define dummy exceptions if websockets are needed conditionally
    class DummyWebSocketException(Exception): pass
    WebSocketException = ConnectionClosed = ConnectionClosedOK = ConnectionClosedError = InvalidURI = InvalidHandshake = PayloadTooBig = ProtocolError = DummyWebSocketException # type: ignore


# --- Configuration & Constants ---
getcontext().prec = 30 # Set global Decimal precision

# Enums for type safety (Imported from config_models or defined as fallback there)
# If the import fails in config_models, these will be Literal types
try:
    from config_models import (
        PositionIdx, Category, OrderFilter, Side, TimeInForce, TriggerBy, TriggerDirection
    )
except ImportError:
    # Fallback Literals must match those in config_models
    PositionIdx = Literal[0, 1, 2] # type: ignore
    Category = Literal["linear", "inverse", "spot", "option"] # type: ignore
    OrderFilter = Literal["Order", "StopOrder", "tpslOrder", "TakeProfit", "StopLoss"] # type: ignore
    Side = Literal["Buy", "Sell"] # type: ignore
    TimeInForce = Literal["GTC", "IOC", "FOK", "PostOnly"] # type: ignore
    TriggerBy = Literal["LastPrice", "MarkPrice", "IndexPrice"] # type: ignore
    TriggerDirection = Literal[1, 2] # type: ignore

# Define other common Literal types used in Bybit V5
OrderType = Literal['Limit', 'Market']
StopLossTakeProfitMode = Literal['Full', 'Partial'] # For position TP/SL mode
ConditionalOrderType = Literal['StopLoss', 'TakeProfit'] # For stopOrderType param
AccountType = Literal['UNIFIED', 'CONTRACT', 'SPOT'] # Common account types


# --- Logger Setup ---
logger = logging.getLogger(__name__) # Get logger instance configured in main script

# --- Global Market Cache ---
class MarketCache:
    """Simple asynchronous cache for CCXT market data."""
    def __init__(self):
        self._markets: Dict[str, Dict[str, Any]] = {}
        self._categories: Dict[str, Optional[str]] = {} # Cache category string
        self._lock = asyncio.Lock()
        self._last_load_time: float = 0.0
        self._cache_duration_seconds: int = 3600 # Cache markets for 1 hour by default

    async def load_markets(self, exchange: ccxt_async.Exchange, reload: bool = False) -> bool:
        """Loads or reloads market data into the cache."""
        current_time = time.monotonic()
        if not reload and self._markets and (current_time - self._last_load_time < self._cache_duration_seconds):
            logger.debug("[MarketCache] Using cached markets (within validity period).")
            return True

        async with self._lock:
            # Double-check after acquiring lock
            if not reload and self._markets and (current_time - self._last_load_time < self._cache_duration_seconds):
                return True

            action = 'Reloading' if self._markets else 'Loading'
            logger.info(f"{Fore.BLUE}[MarketCache] {action} markets for {exchange.id}...{Style.RESET_ALL}")
            try:
                all_markets = await exchange.load_markets(reload=True) # Force reload from exchange
                if not all_markets:
                    logger.critical(f"{Back.RED}FATAL [MarketCache]: Failed to load markets - received empty response from exchange.{Style.RESET_ALL}")
                    self._markets = {}
                    self._categories = {}
                    return False

                self._markets = all_markets
                self._categories.clear() # Clear derived category cache
                self._last_load_time = time.monotonic()
                logger.success(f"{Fore.GREEN}[MarketCache] Loaded {len(self._markets)} markets successfully.{Style.RESET_ALL}")
                return True
            except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
                logger.error(f"{Fore.RED}[MarketCache] Network/Exchange error loading markets: {type(e).__name__} - {e}{Style.RESET_ALL}")
                return False # Indicate failure, cache remains stale or empty
            except ExchangeError as e:
                logger.error(f"{Fore.RED}[MarketCache] Exchange specific error loading markets: {e}{Style.RESET_ALL}", exc_info=False)
                return False
            except Exception as e:
                logger.critical(f"{Back.RED}[MarketCache] CRITICAL unexpected error loading markets: {e}{Style.RESET_ALL}", exc_info=True)
                # Consider clearing cache on critical failure
                self._markets = {}
                self._categories = {}
                return False

    def get_market(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves market data for a symbol from the cache."""
        market_data = self._markets.get(symbol)
        if not market_data:
            logger.debug(f"[MarketCache] Market '{symbol}' not found in cache.")
        return market_data

    def get_category(self, symbol: str) -> Optional[str]:
        """Retrieves or derives the V5 category (linear, inverse, spot, option) for a symbol."""
        if symbol in self._categories:
            return self._categories[symbol]

        market = self.get_market(symbol)
        category_str: Optional[str] = None
        if market:
            category_str = _get_v5_category(market) # Use helper function
            if category_str is None:
                logger.warning(f"[MarketCache] Could not determine V5 category for symbol '{symbol}'. Market data: {market.get('type', 'N/A')}, Info: {market.get('info', {}).get('category', 'N/A')}")

        # Cache the result (even if None) to avoid recalculating
        self._categories[symbol] = category_str
        return category_str

    def get_all_symbols(self) -> List[str]:
        """Returns a list of all symbols currently in the cache."""
        return list(self._markets.keys())

# Instantiate the global cache
market_cache = MarketCache()


# --- Utility Function Imports ---
# Import utility functions AFTER defining logger and market_cache
try:
    from bybit_utils import (safe_decimal_conversion, format_price, format_amount,
                             format_order_id, send_sms_alert_async, retry_api_call)
except ImportError:
    print("FATAL [bybit_helpers]: Could not import from bybit_utils.py.", file=sys.stderr)
    # Define dummy functions or exit if utils are critical
    def _dummy_func(*args, **kwargs): logger.error("Util function missing!"); return None
    def _dummy_decorator_factory(*args_dec, **kwargs_dec):
        def decorator(func): return func
        return decorator
    safe_decimal_conversion = format_price = format_amount = format_order_id = _dummy_func # type: ignore
    send_sms_alert_async = _dummy_func # type: ignore
    retry_api_call = _dummy_decorator_factory # type: ignore
    # sys.exit(1)


# --- Helper to Determine Bybit V5 Category ---
def _get_v5_category(market: Dict[str, Any]) -> Optional[str]:
    """
    Attempts to determine the Bybit V5 category ('linear', 'inverse', 'spot', 'option')
    from CCXT market data.
    """
    if not market: return None

    # Priority 1: Check 'info' field for explicit category (most reliable)
    info = market.get('info', {})
    category_from_info = info.get('category')
    if category_from_info in [Category.LINEAR.value, Category.INVERSE.value, Category.SPOT.value, Category.OPTION.value]:
        return category_from_info

    # Priority 2: Use CCXT standard market type fields
    if market.get('spot', False): return Category.SPOT.value
    if market.get('option', False): return Category.OPTION.value
    if market.get('linear', False): return Category.LINEAR.value # CCXT standard linear flag
    if market.get('inverse', False): return Category.INVERSE.value # CCXT standard inverse flag

    # Priority 3: Infer from 'type' and contract details (less reliable, more guesswork)
    market_type = market.get('type') # e.g., 'spot', 'swap', 'future'
    symbol = market.get('symbol', 'N/A')

    if market_type == Category.SPOT.value: return Category.SPOT.value
    if market_type == Category.OPTION.value: return Category.OPTION.value

    if market_type in ['swap', 'future']:
        # For derivatives, check contract type and settle currency
        contract_type = str(info.get('contractType', '')).lower() # 'linear', 'inverse'
        settle_coin = market.get('settle', '').upper() # e.g., 'USDT', 'BTC'

        if contract_type == Category.LINEAR.value: return Category.LINEAR.value
        if contract_type == Category.INVERSE.value: return Category.INVERSE.value

        # If contractType missing, guess based on settle coin
        if settle_coin in ['USDT', 'USDC']: return Category.LINEAR.value # Common stablecoin collateral
        if settle_coin and settle_coin == market.get('base', '').upper(): return Category.INVERSE.value # Settle = Base -> Inverse

        # If still unsure, make a default assumption (e.g., linear is more common)
        logger.debug(f"[_get_v5_category] Ambiguous derivative {symbol}. Assuming '{Category.LINEAR.value}' based on common usage.")
        return Category.LINEAR.value

    logger.warning(f"[_get_v5_category] Could not determine V5 category for market {symbol} with type '{market_type}'.")
    return None


# --- Exchange Initialization & Configuration ---
# Apply retry logic directly to the initialization function
@retry_api_call(max_retries_override=2, initial_delay_override=5.0, error_message_prefix="Exchange Init Failed")
async def initialize_bybit(app_config: AppConfig, use_async: bool = True) -> Optional[ccxt_async.bybit]:
    """
    Initializes and validates the Bybit CCXT exchange instance using AppConfig.

    Handles testnet/mainnet modes, loads markets, and performs an optional
    authentication check by fetching balance.

    Args:
        app_config: The validated AppConfig object.
        use_async: If True, uses ccxt.async_support. Must be True for this async framework.

    Returns:
        An initialized and validated ccxt.async_support.bybit instance, or None on failure.
    """
    func_name = "initialize_bybit"
    api_conf = app_config.api_config # Convenience alias

    if not CCXT_AVAILABLE:
        logger.critical(f"{Back.RED}FATAL [{func_name}]: CCXT library not available. Cannot initialize exchange.{Style.RESET_ALL}")
        return None
    if not use_async:
        logger.critical(f"{Back.RED}FATAL [{func_name}]: Asynchronous mode (use_async=True) is required for this framework.{Style.RESET_ALL}")
        return None

    mode_str = 'Testnet' if api_conf.testnet_mode else 'Mainnet'
    logger.info(f"{Fore.BLUE}{Style.BRIGHT}[{func_name}] Initializing Bybit V5 ({mode_str}, Async)...{Style.RESET_ALL}")

    exchange: Optional[ccxt_async.bybit] = None # Ensure type hint is for async version
    try:
        has_keys = bool(api_conf.api_key and api_conf.api_secret)
        if not has_keys:
            logger.warning(f"{Fore.YELLOW}[{func_name}] API Key/Secret missing. Initializing in PUBLIC data mode only.{Style.RESET_ALL}")

        # CCXT configuration options
        exchange_options = {
            'apiKey': api_conf.api_key if has_keys else None,
            'secret': api_conf.api_secret if has_keys else None,
            'enableRateLimit': True, # Enable built-in rate limiter
            'options': {
                # V5 API prefers explicit category, defaultType might be less relevant
                # 'defaultType': api_conf.expected_market_type,
                'adjustForTimeDifference': True, # Adjust clock drift
                'recvWindow': api_conf.default_recv_window,
                # Add Broker ID / Referer code if applicable
                'brokerId': f"PB_{app_config.strategy_config.name[:10].replace(' ', '_')}", # Example broker ID format
            },
            # Consider adding user agent if needed
            # 'headers': {'User-Agent': 'MyTradingBot/1.0'},
        }

        # Instantiate the async Bybit exchange class
        exchange = ccxt_async.bybit(exchange_options)

        # Set sandbox mode if configured
        if api_conf.testnet_mode:
            exchange.set_sandbox_mode(True)
            logger.info(f"[{func_name}] Testnet mode enabled.")

        logger.info(f"[{func_name}] Base API URL: {exchange.urls['api']}")

        # --- Load Markets ---
        # Crucial step: must succeed to get market details
        markets_loaded = await market_cache.load_markets(exchange, reload=True)
        if not markets_loaded:
            logger.critical(f"{Back.RED}FATAL [{func_name}]: Failed to load markets. Cannot proceed.{Style.RESET_ALL}")
            await safe_exchange_close(exchange) # Attempt to close partially initialized exchange
            return None

        # Verify the primary trading symbol exists
        primary_symbol = api_conf.symbol
        if not market_cache.get_market(primary_symbol):
            logger.critical(f"{Back.RED}FATAL [{func_name}]: Primary symbol '{primary_symbol}' not found in loaded markets.{Style.RESET_ALL}")
            await safe_exchange_close(exchange)
            return None
        else:
             logger.info(f"[{func_name}] Verified primary symbol '{primary_symbol}' exists.")

        # --- Authentication Check (Optional but Recommended) ---
        if has_keys:
            logger.info(f"[{func_name}] Performing authentication check (fetching balance)...")
            try:
                # Use a function that requires authentication, like fetch_balance
                # Pass app_config to the helper function
                balance_info = await fetch_usdt_balance(exchange, app_config=app_config)
                if balance_info is None:
                    # fetch_usdt_balance logs errors internally, but we check return value
                    raise AuthenticationError("fetch_usdt_balance returned None, indicating potential auth issue.")

                # Log success if balance fetch worked (balance_info is tuple or None)
                if isinstance(balance_info, tuple):
                     equity, avail = balance_info
                     logger.info(f"[{func_name}] Auth check OK. Equity: {equity:.4f}, Available: {avail:.4f}")
                else: # Should not happen if fetch_usdt_balance is correct
                     logger.warning(f"[{func_name}] Auth check returned unexpected type: {type(balance_info)}. Assuming OK.")

            except AuthenticationError as auth_err:
                logger.critical(f"{Back.RED}CRITICAL [{func_name}]: Authentication FAILED! Check API key/secret and permissions. Error: {auth_err}{Style.RESET_ALL}")
                await send_sms_alert_async(f"[BybitHelper] CRITICAL: Bot Auth Failed!", app_config.sms_config)
                await safe_exchange_close(exchange)
                return None
            except (NetworkError, RequestTimeout) as net_err:
                # Handled by retry decorator, but log if it persists
                 logger.error(f"{Fore.RED}[{func_name}] Network error during auth check (after retries): {net_err}{Style.RESET_ALL}")
                 await safe_exchange_close(exchange)
                 return None
            except ExchangeError as ex_err:
                # Catch other exchange errors during balance fetch
                logger.warning(f"{Fore.YELLOW}[{func_name}] Exchange warning during auth check: {ex_err}. Proceeding cautiously.{Style.RESET_ALL}")
                # Allow proceeding but warn user, might be temporary issue or permissions problem
        else:
            logger.info(f"[{func_name}] Skipping authentication check (no API keys provided).")

        # --- Success ---
        logger.success(f"{Fore.GREEN}{Style.BRIGHT}[{func_name}] Bybit V5 exchange initialized and validated successfully ({mode_str}).{Style.RESET_ALL}")
        return exchange

    # --- Exception Handling during Initialization ---
    except AuthenticationError as e:
        # This might catch issues during instantiation if keys are immediately invalid
        logger.critical(f"{Back.RED}FATAL [{func_name}]: Authentication error during setup: {e}.{Style.RESET_ALL}")
        await send_sms_alert_async(f"[BybitHelper] CRITICAL: Bot Auth Failed during setup!", app_config.sms_config)
    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        # Catch network errors not handled by retry during initial setup phase
        logger.critical(f"{Back.RED}FATAL [{func_name}]: Network/Exchange availability error during setup: {e}.{Style.RESET_ALL}")
    except ExchangeError as e:
        # Catch other CCXT exchange errors
        logger.critical(f"{Back.RED}FATAL [{func_name}]: Exchange error during setup: {e}{Style.RESET_ALL}", exc_info=False)
        await send_sms_alert_async(f"[BybitHelper] CRITICAL: Init ExchangeError: {type(e).__name__}", app_config.sms_config)
    except Exception as e:
        # Catch any other unexpected errors
        logger.critical(f"{Back.RED}FATAL [{func_name}]: Unexpected error during initialization: {e}{Style.RESET_ALL}", exc_info=True)
        await send_sms_alert_async(f"[BybitHelper] CRITICAL: Init Unexpected Error: {type(e).__name__}", app_config.sms_config)

    # Ensure exchange is closed if initialization failed at any point
    await safe_exchange_close(exchange)
    return None

async def safe_exchange_close(exchange: Optional[ccxt_async.Exchange]):
    """Safely attempts to close the CCXT exchange connection."""
    if exchange and hasattr(exchange, 'close') and callable(exchange.close):
        try:
            logger.info("[safe_exchange_close] Attempting to close exchange connection...")
            await exchange.close()
            logger.info("[safe_exchange_close] Exchange connection closed.")
        except Exception as e:
            logger.error(f"[safe_exchange_close] Error closing exchange connection: {e}", exc_info=False)


# --- Account Functions ---

@retry_api_call()
async def set_leverage(
    exchange: ccxt_async.bybit, symbol: str, leverage: int, app_config: AppConfig
) -> bool:
    """
    Sets leverage for a specific symbol (Linear/Inverse contracts).
    Requires Unified Trading Account or appropriate V5 category.

    Args:
        exchange: Initialized Bybit exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        leverage: The desired integer leverage (e.g., 5 for 5x).
        app_config: The application configuration object.

    Returns:
        True if leverage was set successfully or already set, False otherwise.
    """
    func_name = "set_leverage"
    log_prefix = f"[{func_name}({symbol} -> {leverage}x)]"

    if leverage <= 0:
        logger.error(f"{Fore.RED}{log_prefix} Leverage must be greater than 0.{Style.RESET_ALL}")
        return False

    # Determine category and validate market
    category = market_cache.get_category(symbol)
    market = market_cache.get_market(symbol)

    if not market:
        logger.error(f"{Fore.RED}{log_prefix} Market data for '{symbol}' not found.{Style.RESET_ALL}")
        return False
    if not category or category not in [Category.LINEAR.value, Category.INVERSE.value]:
        logger.error(f"{Fore.RED}{log_prefix} Leverage can only be set for LINEAR or INVERSE contracts. Category found: {category}.{Style.RESET_ALL}")
        return False

    # Validate leverage against market limits (best effort)
    try:
        limits = market.get('limits', {}).get('leverage', {})
        max_lev_str = limits.get('max')
        min_lev_str = limits.get('min')
        if max_lev_str is not None:
            max_lev = safe_decimal_conversion(max_lev_str, context=f"{symbol} max leverage")
            if max_lev is not None and leverage > max_lev:
                logger.error(f"{Fore.RED}{log_prefix} Requested leverage {leverage}x exceeds maximum allowed ({max_lev}x).{Style.RESET_ALL}")
                return False
        if min_lev_str is not None:
             min_lev = safe_decimal_conversion(min_lev_str, context=f"{symbol} min leverage")
             if min_lev is not None and leverage < min_lev:
                  logger.error(f"{Fore.RED}{log_prefix} Requested leverage {leverage}x is below minimum allowed ({min_lev}x).{Style.RESET_ALL}")
                  return False
    except Exception as e_limits:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Could not fully validate leverage limits due to error: {e_limits}. Proceeding cautiously.{Style.RESET_ALL}")

    # Prepare parameters for V5 API call via CCXT
    # CCXT's set_leverage for Bybit V5 requires category in params
    params = {
        'category': category,
        'buyLeverage': str(leverage), # Bybit API expects string values for leverage
        'sellLeverage': str(leverage)
    }
    logger.info(f"{Fore.CYAN}{log_prefix} Sending request to set leverage... Params: {params}{Style.RESET_ALL}")

    try:
        # CCXT's set_leverage handles the underlying API call
        # For Bybit V5, it calls POST /v5/position/set-leverage
        response = await exchange.set_leverage(leverage, symbol, params=params)
        # CCXT's set_leverage might not return detailed info, success implies it didn't raise an error.
        # Bybit's API for set_leverage doesn't return much on success (retCode 0).
        # We rely on absence of exceptions as success indicator.
        logger.success(f"{Fore.GREEN}{log_prefix} Leverage set/confirmed OK (Implies ISOLATED margin mode for symbol).{Style.RESET_ALL}")
        return True

    except ExchangeError as e:
        # Check specific Bybit V5 error codes or messages
        # Find codes at: https://bybit-exchange.github.io/docs/v5/error_code
        error_code = getattr(e, 'code', None) # CCXT often stores the code here
        error_msg = str(e).lower()

        # Code 110043: Leverage not modified (already set to the desired value)
        if error_code == 110043 or "leverage not modified" in error_msg:
            logger.info(f"{Fore.YELLOW}{log_prefix} Leverage already set to {leverage}x.{Style.RESET_ALL}")
            return True # Treat as success
        # Code 110021: Can't set leverage under Hedge Mode (needs to be set via position mode switch?)
        elif error_code == 110021:
             logger.error(f"{Fore.RED}{log_prefix} Failed (Code: {error_code}): Cannot set leverage in Hedge Mode via this endpoint. Use set_position_mode? Error: {e}{Style.RESET_ALL}")
             return False
        # Add other relevant error codes if encountered
        # elif error_code == ... :

        else:
            # Generic exchange error
            logger.error(f"{Fore.RED}{log_prefix} ExchangeError setting leverage: {e}{Style.RESET_ALL}", exc_info=False)
            return False
    except (NetworkError, RequestTimeout) as e:
        # Network errors will be retried by the decorator, but log if they persist
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network error during leverage setting (will be retried): {e}.{Style.RESET_ALL}")
        raise e # Re-raise for the decorator to handle retry
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error setting leverage: {e}{Style.RESET_ALL}", exc_info=True)
        return False

@retry_api_call()
async def fetch_usdt_balance(
    exchange: ccxt_async.bybit, app_config: AppConfig
) -> Optional[Tuple[Decimal, Decimal]]:
    """
    Fetches USDT balance details from the UNIFIED trading account on Bybit V5.

    Args:
        exchange: Initialized Bybit exchange instance.
        app_config: The application configuration object.

    Returns:
        A tuple containing (Total Equity, Available Balance) as Decimals,
        or None if fetching fails or USDT balance is not found.
    """
    func_name = "fetch_usdt_balance"
    log_prefix = f"[{func_name}]"
    usdt_symbol = app_config.api_config.usdt_symbol # Get target coin from config
    account_type_target = AccountType.UNIFIED.value # Bybit V5 standard

    logger.debug(f"{log_prefix} Fetching {account_type_target} account balance ({usdt_symbol})...")
    try:
        # Use fetch_balance with params for V5
        # V5 requires specifying accountType for unified/contract balances
        balance_data = await exchange.fetch_balance(params={'accountType': account_type_target})

        # --- Parse V5 Response Structure ---
        # The structure is nested: result -> list -> account dict -> coin list
        info_list = balance_data.get('info', {}).get('result', {}).get('list', [])
        if not info_list:
            logger.warning(f"{log_prefix} Balance response 'list' is empty or missing. Data: {balance_data.get('info', {})}")
            return None

        # Find the dictionary for the targeted account type
        unified_account_info = next((acc for acc in info_list if acc.get('accountType') == account_type_target), None)
        if not unified_account_info:
            logger.warning(f"{log_prefix} Could not find account details for type '{account_type_target}' in response.")
            return None

        # Extract total equity for the account
        total_equity_str = unified_account_info.get('totalEquity')
        total_equity = safe_decimal_conversion(total_equity_str, context="Total Equity")
        if total_equity is None: logger.warning(f"{log_prefix} Failed to parse total equity ('{total_equity_str}'). Assuming 0.")
        final_equity = max(Decimal("0"), total_equity or Decimal("0")) # Ensure non-negative

        # Find the specific coin (USDT) within the account's coin list
        available_balance = None
        coin_list = unified_account_info.get('coin', [])
        usdt_coin_info = next((coin for coin in coin_list if coin.get('coin') == usdt_symbol), None)

        if usdt_coin_info:
            # Try 'availableToWithdraw' first, fallback to 'availableBalance'
            avail_str = usdt_coin_info.get('availableToWithdraw') or usdt_coin_info.get('availableBalance')
            available_balance = safe_decimal_conversion(avail_str, context=f"{usdt_symbol} Available Balance")
        else:
            logger.warning(f"{log_prefix} {usdt_symbol} details not found within the {account_type_target} account coin list.")

        if available_balance is None: logger.warning(f"{log_prefix} Failed to parse available balance for {usdt_symbol}. Assuming 0.")
        final_available = max(Decimal("0"), available_balance or Decimal("0")) # Ensure non-negative

        logger.info(f"{Fore.GREEN}{log_prefix} OK - Equity: {final_equity:.4f}, Available {usdt_symbol}: {final_available:.4f}{Style.RESET_ALL}")
        return final_equity, final_available

    except AuthenticationError as e:
        logger.error(f"{Fore.RED}{log_prefix} Authentication error fetching balance: {e}{Style.RESET_ALL}")
        return None # Auth errors are critical, don't retry indefinitely here (retry handles initial attempts)
    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network/Exchange error fetching balance (will be retried): {e}.{Style.RESET_ALL}")
        raise e # Re-raise for the decorator
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Exchange error fetching balance: {e}{Style.RESET_ALL}", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching balance: {e}{Style.RESET_ALL}", exc_info=True)
        return None

# --- Market Data Functions ---

# Note: The retry decorator is applied here for robustness against transient network issues.
@retry_api_call()
async def fetch_ohlcv_paginated(
    exchange: ccxt_async.bybit,
    symbol: str,
    timeframe: str,
    app_config: AppConfig,
    since: Optional[int] = None, # Timestamp in ms
    limit: Optional[int] = None, # Max number of candles to return
    max_pages: int = 100 # Safety limit for pagination loops
) -> Optional[Union[pd.DataFrame, List[list]]]:
    """
    Fetches OHLCV data for a symbol, handling pagination automatically using CCXT's built-in support.

    Args:
        exchange: Initialized Bybit exchange instance.
        symbol: The market symbol.
        timeframe: The timeframe string (e.g., '1m', '1h', '1d').
        app_config: The application configuration object.
        since: Start time timestamp in milliseconds (optional).
        limit: The maximum number of candles to fetch (optional).
        max_pages: Safety limit to prevent infinite loops in pagination.

    Returns:
        A pandas DataFrame with OHLCV data (if pandas is available) or a list of lists.
        Columns/List format: [timestamp, open, high, low, close, volume]
        Returns None on critical failure. Returns partial data on non-critical errors during pagination.
    """
    func_name = "fetch_ohlcv_paginated"
    log_prefix = f"[{func_name}({symbol}, {timeframe})]"

    if not PANDAS_AVAILABLE:
        logger.info(f"{log_prefix} Pandas not available. OHLCV data will be returned as list of lists.")

    # Validate market and get category
    market = market_cache.get_market(symbol)
    category = market_cache.get_category(symbol)
    if not market:
        logger.error(f"{Fore.RED}{log_prefix} Market data for '{symbol}' not found. Cannot fetch OHLCV.{Style.RESET_ALL}")
        return None
    if not category:
        # Attempt to infer category if not cached, but might be needed for params
        logger.warning(f"{Fore.YELLOW}{log_prefix} V5 Category for '{symbol}' not cached. Fetch might fail if category param is required.{Style.RESET_ALL}")
        # If category is strictly required by Bybit V5 fetch_ohlcv, return None here.
        # return None

    # Parse timeframe to milliseconds for logging/validation (optional)
    try:
        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
        logger.debug(f"{log_prefix} Parsed timeframe: {timeframe} ({timeframe_ms}ms)")
    except Exception as e_tf:
        logger.error(f"{log_prefix} Invalid timeframe '{timeframe}': {e_tf}.")
        return None

    # Bybit V5 fetchOHLCV requires the 'category' parameter
    params = {'category': category} if category else {}
    if not params:
         logger.warning(f"{Fore.YELLOW}{log_prefix} Category parameter missing for fetch_ohlcv. Bybit V5 might require it.{Style.RESET_ALL}")


    all_candles = []
    current_since = since
    fetch_limit_per_page = 1000 # Bybit V5 limit per request
    pages = 0

    logger.info(f"{Fore.BLUE}{log_prefix} Fetching OHLCV data... Target limit: {limit or 'All'}{Style.RESET_ALL}")

    try:
        while pages < max_pages:
            pages += 1
            # Determine limit for this specific fetch call
            current_fetch_limit = fetch_limit_per_page
            if limit is not None:
                remaining_needed = limit - len(all_candles)
                if remaining_needed <= 0:
                    logger.debug(f"{log_prefix} Target limit of {limit} candles reached.")
                    break
                current_fetch_limit = min(fetch_limit_per_page, remaining_needed)

            logger.debug(f"{log_prefix} Page {pages}/{max_pages}, Fetching since={current_since}, limit={current_fetch_limit}, params={params}")

            # Fetch one page/chunk of candles
            # The retry decorator handles transient errors for this call
            candles_chunk = await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=current_fetch_limit,
                params=params
            )

            if not candles_chunk:
                logger.info(f"{log_prefix} No more candles returned by exchange (or empty chunk).")
                break # Exit loop if no data returned

            # Filter out potential duplicates if 'since' overlaps precisely
            if all_candles and candles_chunk[0][0] <= all_candles[-1][0]:
                 logger.debug(f"{log_prefix} First candle timestamp {candles_chunk[0][0]} overlaps last fetched {all_candles[-1][0]}. Filtering duplicates.")
                 candles_chunk = [c for c in candles_chunk if c[0] > all_candles[-1][0]]
                 if not candles_chunk:
                      logger.debug(f"{log_prefix} All candles in chunk were duplicates.")
                      break # Avoid infinite loop if only duplicates are returned

            all_candles.extend(candles_chunk)
            num_fetched_chunk = len(candles_chunk)
            last_ts = candles_chunk[-1][0]
            first_ts = candles_chunk[0][0]
            ts_to_dt_str = lambda ts: pd.to_datetime(ts, unit='ms').strftime('%Y-%m-%d %H:%M:%S') if PANDAS_AVAILABLE else str(ts)

            logger.info(f"{log_prefix} Fetched {num_fetched_chunk} candles (Range: {ts_to_dt_str(first_ts)} to {ts_to_dt_str(last_ts)}). Total: {len(all_candles)}")

            # Prepare 'since' for the next iteration
            current_since = last_ts + 1 # Start next fetch right after the last candle

            # Check if the desired limit has been reached
            if limit is not None and len(all_candles) >= limit:
                logger.info(f"{log_prefix} Reached or exceeded target limit of {limit} candles.")
                break

            # Check if the exchange returned fewer candles than requested (indicates end of data)
            if num_fetched_chunk < current_fetch_limit:
                logger.info(f"{log_prefix} Received fewer candles ({num_fetched_chunk}) than limit ({current_fetch_limit}). Assuming end of available data.")
                break

            # Optional small delay between pages to respect rate limits further
            await asyncio.sleep(exchange.rateLimit / 2000 if exchange.rateLimit > 0 else 0.1)

        if pages >= max_pages:
             logger.warning(f"{Fore.YELLOW}{log_prefix} Reached maximum pagination limit ({max_pages} pages). Data might be incomplete.{Style.RESET_ALL}")

        # --- Process Collected Data ---
        if not all_candles:
            logger.warning(f"{log_prefix} No OHLCV candles were collected.")
            return pd.DataFrame() if PANDAS_AVAILABLE else []

        logger.info(f"{log_prefix} Total raw candles collected: {len(all_candles)}")

        # Apply final limit if specified (in case pagination slightly overshot)
        if limit is not None and len(all_candles) > limit:
            logger.debug(f"{log_prefix} Trimming collected candles from {len(all_candles)} to target limit {limit}.")
            all_candles = all_candles[-limit:] # Keep the most recent 'limit' candles

        # Sort and remove duplicates (essential for consistency)
        # Sort by timestamp (first element)
        all_candles.sort(key=lambda x: x[0])
        # Remove duplicates based on timestamp
        unique_candles_dict = {c[0]: c for c in all_candles}
        unique_candles = list(unique_candles_dict.values())
        if len(unique_candles) < len(all_candles):
             logger.debug(f"{log_prefix} Removed {len(all_candles) - len(unique_candles)} duplicate candle timestamps.")

        # Return as DataFrame or list
        if PANDAS_AVAILABLE:
            try:
                df = pd.DataFrame(unique_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('datetime', inplace=True)
                # Convert columns to numeric types (handle potential errors)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                # Optional: Drop rows with NaN in critical OHLC columns after conversion
                # df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
                logger.success(f"{Fore.GREEN}{log_prefix} Processed {len(df)} unique candles into DataFrame.{Style.RESET_ALL}")
                return df
            except Exception as e_df:
                logger.error(f"{log_prefix} Failed to create or process DataFrame: {e_df}. Returning raw list.", exc_info=True)
                return unique_candles # Fallback to list
        else:
            logger.success(f"{Fore.GREEN}{log_prefix} Returning {len(unique_candles)} unique candles as list.{Style.RESET_ALL}")
            return unique_candles

    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        logger.error(f"{Fore.RED}{log_prefix} Unrecoverable API error during pagination: {type(e).__name__}: {e}{Style.RESET_ALL}", exc_info=False)
    except AuthenticationError as e:
         logger.error(f"{Fore.RED}{log_prefix} Authentication error during pagination: {e}{Style.RESET_ALL}")
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Unrecoverable Exchange error during pagination: {e}{Style.RESET_ALL}", exc_info=False)
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error during OHLCV fetching: {e}{Style.RESET_ALL}", exc_info=True)

    # Return partial data if some was collected before the error
    if all_candles:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Returning PARTIAL data ({len(all_candles)}) due to error during pagination.{Style.RESET_ALL}")
        # Process the partial data as best as possible (sort, unique, format)
        all_candles.sort(key=lambda x: x[0])
        unique_candles_dict = {c[0]: c for c in all_candles}
        unique_candles = list(unique_candles_dict.values())
        if PANDAS_AVAILABLE:
             try:
                 df = pd.DataFrame(unique_candles, columns=['timestamp','open','high','low','close','volume'])
                 df['datetime']=pd.to_datetime(df['timestamp'],unit='ms', utc=True)
                 df.set_index('datetime',inplace=True)
                 for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
                 return df
             except Exception: return unique_candles # Fallback
        else: return unique_candles
    else:
        return None # Indicate complete failure


@retry_api_call()
async def fetch_ticker_validated(exchange: ccxt_async.bybit, symbol: str, app_config: AppConfig) -> Optional[Dict]:
    """
    Fetches the ticker for a symbol using V5 parameters and validates essential fields.

    Args:
        exchange: Initialized Bybit exchange instance.
        symbol: The market symbol.
        app_config: The application configuration object.

    Returns:
        The validated ticker dictionary (CCXT format), or None on failure.
    """
    func_name = "fetch_ticker_validated"
    log_prefix = f"[{func_name}({symbol})]"
    logger.debug(f"{log_prefix} Fetching...")

    category = market_cache.get_category(symbol)
    if not category:
        logger.error(f"{Fore.RED}{log_prefix} Cannot determine V5 category for '{symbol}'. Cannot fetch ticker.{Style.RESET_ALL}")
        return None

    params = {'category': category}
    logger.debug(f"{log_prefix} Using params: {params}")

    try:
        ticker = await exchange.fetch_ticker(symbol, params=params)

        # --- Validation ---
        if not ticker or not isinstance(ticker, dict):
            logger.error(f"{Fore.RED}{log_prefix} Received invalid ticker response: {ticker}{Style.RESET_ALL}")
            return None

        # Essential keys for basic operation
        required_keys = ['symbol', 'last', 'bid', 'ask', 'timestamp']
        # Common keys that are useful but might occasionally be None
        common_keys = ['datetime', 'high', 'low', 'bidVolume', 'askVolume', 'vwap', 'open', 'close', 'previousClose', 'change', 'percentage', 'average', 'baseVolume', 'quoteVolume']

        missing_keys = [k for k in required_keys if ticker.get(k) is None]
        if missing_keys:
            logger.error(f"{Fore.RED}{log_prefix} Ticker response missing required keys or values are None: {missing_keys}. Data: {ticker}{Style.RESET_ALL}")
            return None

        missing_common = [k for k in common_keys if ticker.get(k) is None]
        if missing_common:
            logger.debug(f"{log_prefix} Ticker missing common keys: {missing_common}.")

        # Timestamp validation (check for reasonable recency)
        ts_ms = ticker.get('timestamp')
        ts_log_msg = "TS: N/A"
        ts_ok = False
        if ts_ms is not None and isinstance(ts_ms, int):
            now_ms = int(time.time() * 1000)
            age_ms = now_ms - ts_ms
            max_age_s = 120 # Allow up to 2 minutes old (adjust as needed)
            min_age_s = -10 # Allow slightly future timestamp for clock drift
            max_diff_ms, min_diff_ms = max_age_s * 1000, min_age_s * 1000

            age_s = age_ms / 1000.0
            dt_str = ticker.get('datetime', f"ms:{ts_ms}") # Use ISO format if available

            if age_ms > max_diff_ms or age_ms < min_diff_ms:
                logger.warning(f"{Fore.YELLOW}{log_prefix} Timestamp ({dt_str}) seems stale or invalid. Age: {age_s:.1f}s (Max allowed: {max_age_s}s).{Style.RESET_ALL}")
                ts_log_msg = f"{Fore.YELLOW}TS: Stale ({age_s:.1f}s){Style.RESET_ALL}"
                # Decide whether to return None or just warn based on stale TS
                # return None # Stricter: reject stale tickers
            else:
                ts_log_msg = f"TS OK ({age_s:.1f}s)"
                ts_ok = True
        elif ts_ms is None:
             ts_log_msg = f"{Fore.YELLOW}TS: Missing{Style.RESET_ALL}"
        else:
             ts_log_msg = f"{Fore.YELLOW}TS: Invalid Type ({type(ts_ms).__name__}){Style.RESET_ALL}"

        # Log summary
        last_px_str = format_price(exchange, symbol, ticker.get('last')) or "N/A"
        bid_px_str = format_price(exchange, symbol, ticker.get('bid')) or "N/A"
        ask_px_str = format_price(exchange, symbol, ticker.get('ask')) or "N/A"
        logger.info(f"{Fore.GREEN}{log_prefix} OK: Last={last_px_str}, Bid={bid_px_str}, Ask={ask_px_str} | {ts_log_msg}{Style.RESET_ALL}")

        return ticker

    except BadSymbol as e:
         logger.error(f"{Fore.RED}{log_prefix} Invalid symbol error: {e}{Style.RESET_ALL}")
         return None
    except AuthenticationError as e:
         logger.error(f"{Fore.RED}{log_prefix} Authentication error fetching ticker: {e}{Style.RESET_ALL}")
         return None # Auth errors usually require intervention
    except (NetworkError, RequestTimeout, ExchangeNotAvailable, DDoSProtection) as e:
        logger.warning(f"{Fore.YELLOW}{log_prefix} Network/Exchange error fetching ticker (will be retried): {e}.{Style.RESET_ALL}")
        raise e # Re-raise for decorator
    except ExchangeError as e:
        logger.error(f"{Fore.RED}{log_prefix} Exchange error fetching ticker: {e}{Style.RESET_ALL}", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"{Fore.RED}{log_prefix} Unexpected error fetching ticker: {e}{Style.RESET_ALL}", exc_info=True)
        return None

# --- Fetch L2 Order Book ---
@retry_api_call()
async def fetch_l2_order_book_validated(
    exchange: ccxt_async.bybit, symbol: str, limit: int, app_config: AppConfig
) -> Optional[Dict[str, Any]]:
    """Fetches L2 order book using V5 parameters and validates structure."""
    func_name = "fetch_l2_order_book"; log_prefix = f"[{func_name}({symbol}, limit={limit})]"
    logger.debug(f"{log_prefix} Fetching...")
    category = market_cache.get_category(symbol); if not category: logger.error(f"{Fore.RED}{log_prefix} No category for {symbol}.{Style.RESET_ALL}"); return None
    params = {'category': category}
    try:
        ob = await exchange.fetch_l2_order_book(symbol, limit=limit, params=params)
        # Basic validation
        if not ob or not isinstance(ob, dict): logger.error(f"{Fore.RED}{log_prefix} Invalid OB response.{Style.RESET_ALL}"); return None
        bids, asks = ob.get('bids'), ob.get('asks')
        if bids is None or asks is None or not isinstance(bids, list) or not isinstance(asks, list): logger.error(f"{Fore.RED}{log_prefix} Missing/invalid bids or asks.{Style.RESET_ALL}"); return None
        if not bids or not asks: logger.warning(f"{Fore.YELLOW}{log_prefix} Order book side empty (Bids:{len(bids)}, Asks:{len(asks)}).{Style.RESET_ALL}"); # Continue but warn
        # Further validation (e.g., check format [price, amount], check sorting) can be added here if needed
        logger.info(f"{Fore.GREEN}{log_prefix} OK: Fetched L2 OB. Bids={len(bids)}, Asks={len(asks)}{Style.RESET_ALL}")
        return ob
    except (BadSymbol, NetworkError, RequestTimeout, ExchangeError) as e: logger.error(f"{Fore.RED}{log_prefix} Failed: {type(e).__name__} - {e}{Style.RESET_ALL}"); return None # Don't raise for retry on BadSymbol
    except Exception as e: logger.error(f"{Fore.RED}{log_prefix} Unexpected error: {e}{Style.RESET_ALL}", exc_info=True); return None


# --- Order Management Functions ---

@retry_api_call(max_retries_override=1, initial_delay_override=0.5) # Retry once on transient network errors for market orders
async def place_market_order_slippage_check(
    exchange: ccxt_async.bybit, symbol: str, side: Side, amount: Decimal, app_config: AppConfig,
    max_slippage_pct_override: Optional[Decimal] = None,
    is_reduce_only: bool = False,
    time_in_force: TimeInForce = TimeInForce.IOC, # IOC default for market orders
    client_order_id: Optional[str] = None,
    position_idx: Optional[PositionIdx] = None, # For hedge mode
    reason: str = "Market Order" # For logging and potentially client_order_id
) -> Optional[Dict]:
    """
    Places a market order with an optional pre-flight spread/slippage check against the order book.

    Args:
        exchange: Initialized Bybit exchange instance.
        symbol: The market symbol.
        side: 'Buy' or 'Sell' (from config_models/bybit_helpers Side enum/literal).
        amount: The quantity to trade (as Decimal).
        app_config: The application configuration object.
        max_slippage_pct_override: Override the default slippage check percentage from config.
        is_reduce_only: Set to True for closing/reducing positions only.
        time_in_force: Time in force (IOC or FOK recommended for market).
        client_order_id: Custom client order ID (optional).
        position_idx: Specify position index (0, 1, or 2) for Hedge Mode.
        reason: Short description for logging/order ID generation.

    Returns:
        The order dictionary returned by CCXT upon successful placement, or None on failure/abort.
    """
    func_name = "place_market_order"
    action_str = "ReduceOnly" if is_reduce_only else "Open/Increase"
    log_prefix = f"[{func_name}({symbol}, {side.value}, {amount:.8f}, {action_str}, {reason})]" # Use side.value
    api_conf = app_config.api_config # Convenience alias

    # --- Input Validation ---
    if amount <= api_conf.position_qty_epsilon:
        logger.error(f"{Fore.RED}{log_prefix} Invalid order amount ({amount}). Must be > {api_conf.position_qty_epsilon}.{Style.RESET_ALL}")
        return None
    if side not in [Side.BUY, Side.SELL]: # Check against actual enum/literal values
         logger.error(f"{Fore.RED}{log_prefix} Invalid side: {side}. Must be '{Side.BUY}' or '{Side.SELL}'.{Style.RESET_ALL}"); return None

    category = market_cache.get_category(symbol)
    market = market_cache.get_market(symbol)
    if not market or not category:
        logger.error(f"{Fore.RED}{log_prefix} Market/Category info unavailable for '{symbol}'. Cannot place order.{Style.RESET_ALL}")
        return None

    # Format amount according to market precision
    formatted_amount_str = format_amount(exchange, symbol, amount)
    if formatted_amount_str is None or formatted_amount_str == "Error":
        logger.error(f"{Fore.RED}{log_prefix} Failed to format order amount {amount} for market precision.{Style.RESET_ALL}")
        return None
    formatted_amount = safe_decimal_conversion(formatted_amount_str)
    if formatted_amount is None or formatted_amount <= 0:
         logger.error(f"{Fore.RED}{log_prefix} Formatted amount '{formatted_amount_str}' is invalid or zero.{Style.RESET_ALL}"); return None

    # Determine effective slippage tolerance
    effective_slippage_pct = max_slippage_pct_override if max_slippage_pct_override is not None else api_conf.default_slippage_pct
    perform_slippage_check = effective_slippage_pct > 0

    logger.info(f"{Fore.BLUE}{log_prefix} Preparing... Amount: {formatted_amount_str}, TIF: {time_in_force.value}, SlippageChk: {'Enabled' if perform_slippage_check else 'Disabled'} ({effective_slippage_pct:.3%}){Style.RESET_ALL}")

    # --- Slippage Check (Optional) ---
    if perform_slippage_check:
        try:
            logger.debug(f"{log_prefix} Performing pre-flight slippage check...")
            # Fetch shallow order book
            ob = await fetch_l2_order_book_validated(exchange, symbol, api_conf.shallow_ob_fetch_depth, app_config)
            if ob and ob.get('bids') and ob.get('asks'):
                best_bid_price = safe_decimal_conversion(ob['bids'][0][0])
                best_ask_price = safe_decimal_conversion(ob['asks'][0][0])

                if best_bid_price and best_ask_price and best_bid_price > 0:
                    mid_price = (best_bid_price + best_ask_price) / 2
                    spread_pct = ((best_ask_price - best_bid_price) / mid_price) if mid_price > 0 else Decimal(0)

                    # Slippage check: compare spread to tolerance
                    # For a market buy, execution ~ best ask. For market sell, execution ~ best bid.
                    # A wide spread implies high potential slippage vs mid-price.
                    if spread_pct > effective_slippage_pct:
                        logger.error(f"{Back.RED}{Fore.WHITE}ABORT MARKET ORDER:{Style
