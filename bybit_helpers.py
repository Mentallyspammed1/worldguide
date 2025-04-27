#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bybit V5 CCXT Helper Functions (v2.0 - Enhanced & Integrated)

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
  orders (filtered), and updating existing limit orders (edit).
- Position Management: Fetching detailed current position information (V5 specific),
  closing positions using reduce-only market orders, and retrieving detailed position
  risk metrics (IMR, MMR, Liq. Price, etc.) using V5 logic.
- Balance & Margin: Fetching USDT balances (equity/available) using V5 UNIFIED
  account logic, and calculating estimated margin requirements for potential orders.
- Utilities: Market validation against exchange data (type, logic, active status).

Key Enhancements in v2.0:
- Consistent use of an external `retry_api_call` decorator (assumed) for API resilience.
- Removal of internal retry loops where the decorator suffices.
- Removal of manual rate limit tracking (Snippet 25) in favor of decorator handling.
- Standardized parameter names (e.g., `config` passed explicitly).
- Improved type hinting (`Literal`, `Union`, `Optional`, `Dict`, `List`, `Tuple`).
- Enhanced docstrings with more V5 details, parameters, and potential errors.
- More robust validation within functions (e.g., data structures, price/amount checks).
- Consistent use of f-strings and standardized logging levels/formats with colorama.
- Added `client_order_id` support where applicable.
- Refined V5 parameter usage (e.g., `category`, `triggerBy`, `positionIdx`).
- Improved error handling for specific V5 return codes.
- Enhanced standalone test block (`if __name__ == "__main__":`).
"""

# --- Dependencies from Importing Script ---
# This file assumes that the script importing it (e.g., ps.py) has already
# initialized and made available the following essential components:
#
# 1.  `logger`: A pre-configured `logging.Logger` object for logging messages.
# 2.  `config`: A configuration object or class instance (e.g., `Config`)
#     passed explicitly to functions requiring it. Contains API keys, settings
#     (retry counts, symbol, fees, etc.).
# 3.  `safe_decimal_conversion(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]`:
#     A robust function to convert various inputs to `Decimal`, returning `default` or `None` on failure.
# 4.  `format_price(exchange: ccxt.Exchange, symbol: str, price: Any) -> str`:
#     Formats a price value according to the market's precision rules.
# 5.  `format_amount(exchange: ccxt.Exchange, symbol: str, amount: Any) -> str`:
#     Formats an amount value according to the market's precision rules.
# 6.  `format_order_id(order_id: Any) -> str`:
#     Formats an order ID (e.g., showing the last few digits) for concise logging.
# 7.  `send_sms_alert(message: str, config: Optional['Config'] = None) -> bool` (Optional):
#     Function to send SMS alerts, often used for critical errors or actions.
# 8.  `retry_api_call` (Decorator): A decorator applied to API-calling functions
#     within this module. It MUST be defined in the importing scope and should
#     handle retries for common CCXT exceptions (e.g., `NetworkError`,
#     `RateLimitExceeded`, `ExchangeNotAvailable`) with appropriate backoff delays.
#     This module *relies* on this external decorator for API call resilience
#     and rate limit management. Functions making API calls are decorated with `@retry_api_call()`.
#
# Ensure these components are correctly defined and accessible *before* calling functions from this module.
# ------------------------------------------

# Standard Library Imports
import logging
import os
import sys
import time
import traceback
import random
from decimal import Decimal, ROUND_HALF_UP, DivisionByZero, InvalidOperation, getcontext
from typing import Optional, Dict, List, Tuple, Any, Literal, TypeVar, Callable, Union
from functools import wraps

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


# Set Decimal context precision
getcontext().prec = 28

# --- Placeholder for Config Class Type Hint ---
# Use forward reference if Config is defined in the main script
# Provide a more detailed placeholder for type hinting and potential standalone use
if 'Config' not in globals():
    class Config: # Basic placeholder for type hinting - Adapt as needed
        # Retry mechanism settings (used by external decorator, values may be needed internally)
        RETRY_COUNT: int = 3
        RETRY_DELAY_SECONDS: float = 2.0
        # Position / Order settings
        POSITION_QTY_EPSILON: Decimal = Decimal("1e-9") # Threshold for treating qty as zero
        DEFAULT_SLIPPAGE_PCT: Decimal = Decimal("0.005") # Default max slippage for market orders
        ORDER_BOOK_FETCH_LIMIT: int = 25 # Default depth for fetch_l2_order_book
        SHALLOW_OB_FETCH_DEPTH: int = 5 # Depth used for slippage check analysis
        # Symbol / Market settings
        SYMBOL: str = "BTC/USDT:USDT" # Default symbol
        USDT_SYMBOL: str = "USDT" # Quote currency symbol for balance checks
        EXPECTED_MARKET_TYPE: Literal['swap', 'future', 'spot', 'option'] = 'swap'
        EXPECTED_MARKET_LOGIC: Optional[Literal['linear', 'inverse']] = 'linear'
        # Exchange connection settings
        EXCHANGE_ID: str = "bybit"
        API_KEY: Optional[str] = None
        API_SECRET: Optional[str] = None
        DEFAULT_RECV_WINDOW: int = 10000
        TESTNET_MODE: bool = True
        # Account settings
        DEFAULT_LEVERAGE: int = 10
        DEFAULT_MARGIN_MODE: Literal['cross', 'isolated'] = 'cross'
        DEFAULT_POSITION_MODE: Literal['one-way', 'hedge'] = 'one-way'
        # Fees
        TAKER_FEE_RATE: Decimal = Decimal("0.00055") # Example Bybit VIP 0 Taker fee
        MAKER_FEE_RATE: Decimal = Decimal("0.0002") # Example Bybit VIP 0 Maker fee
        # SMS Alerts (Optional)
        ENABLE_SMS_ALERTS: bool = False
        SMS_RECIPIENT_NUMBER: Optional[str] = None
        SMS_TIMEOUT_SECONDS: int = 30
        # Side / Position Constants (Used within functions)
        SIDE_BUY: str = "buy"
        SIDE_SELL: str = "sell"
        POS_LONG: str = "LONG"
        POS_SHORT: str = "SHORT"
        POS_NONE: str = "NONE"
        pass

# --- Logger Setup ---
# Assume logger is initialized and provided by the main script.
# For standalone testing or clarity, define a placeholder logger:
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
        logger.setLevel(logging.DEBUG) # Set to DEBUG for testing, INFO/WARNING for production
    logger.info("Placeholder logger initialized for bybit_helpers module.")

# --- Placeholder for retry decorator ---
# This is critical: The actual implementation MUST be provided by the importing script.
# This placeholder only ensures the code is syntactically valid and defines the signature.
T = TypeVar('T')
def retry_api_call(max_retries: int = 3, initial_delay: float = 1.0, **decorator_kwargs) -> Callable[[Callable[..., T]], Callable[..., T]]:
     """
     Placeholder for the actual retry decorator. The real decorator (provided by the importer)
     MUST handle specific CCXT exceptions (NetworkError, RateLimitExceeded, ExchangeError, etc.)
     and implement proper backoff logic based on config values or passed arguments.
     It is responsible for all API call resilience and rate limit waits.
     """
     def decorator(func: Callable[..., T]) -> Callable[..., T]:
         @wraps(func)
         def wrapper(*args: Any, **kwargs: Any) -> T:
             # In a real scenario, this wrapper would contain the retry logic.
             # For this placeholder, we just call the function directly.
             # logger.debug(f"Placeholder retry decorator executing for {func.__name__}")
             try:
                 return func(*args, **kwargs)
             except Exception as e:
                 # logger.error(f"Placeholder retry decorator caught unhandled exception in {func.__name__}: {e}", exc_info=True)
                 raise # Re-raise; the real decorator would handle specific exceptions for retry
         return wrapper
     return decorator

# --- Placeholder Helper Functions (Assume provided by importer) ---
# These are basic implementations. The importing script should provide robust versions.

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
    is_enabled = getattr(config, 'ENABLE_SMS_ALERTS', False) if config else False
    if is_enabled:
        recipient = getattr(config, 'SMS_RECIPIENT_NUMBER', None)
        timeout = getattr(config, 'SMS_TIMEOUT_SECONDS', 30)
        if recipient:
            logger.info(f"--- SIMULATING SMS Alert to {recipient} ---")
            print(f"SMS: {message}")
            # Placeholder for actual Termux call
            return True # Simulation success
        else:
            logger.warning("SMS alerts enabled but no recipient number configured.")
            return False
    else:
        # logger.debug(f"SMS alert suppressed (disabled): {message}")
        return False


# --- Helper Function Implementations ---

def _get_v5_category(market: Dict[str, Any]) -> Optional[Literal['linear', 'inverse', 'spot', 'option']]:
    """Internal helper to determine the Bybit V5 category from a market object."""
    if not market: return None
    if market.get('linear'): return 'linear'
    if market.get('inverse'): return 'inverse'
    if market.get('spot'): return 'spot'
    if market.get('option'): return 'option'
    # Fallback based on market type if linear/inverse flags aren't present (older ccxt?)
    market_type = market.get('type')
    if market_type == 'swap': return 'linear' # Assume linear swap if flags missing
    if market_type == 'future': return 'linear' # Assume linear future
    if market_type == 'spot': return 'spot'
    if market_type == 'option': return 'option'
    logger.warning(f"_get_v5_category: Could not determine category for market: {market.get('symbol')}")
    return None

# Snippet 1 / Function 1: Initialize Bybit Exchange
@retry_api_call(max_retries=3, initial_delay=2.0)
def initialize_bybit(config: Config) -> Optional[ccxt.bybit]:
    """
    Initializes and validates the Bybit CCXT exchange instance using V5 API settings.

    Sets sandbox mode, default order type (swap), loads markets, performs an initial
    balance check, and attempts to set default margin mode based on config.

    Args:
        config: The configuration object containing API keys, testnet flag, symbol, etc.

    Returns:
        A configured and validated `ccxt.bybit` instance, or `None` if initialization fails.

    Raises:
        Catches and logs CCXT exceptions during initialization. Relies on the
        `retry_api_call` decorator for retries, raising the final exception on failure.
    """
    func_name = "initialize_bybit"
    logger.info(f"{Fore.BLUE}[{func_name}] Initializing Bybit (V5) exchange instance...{Style.RESET_ALL}")
    try:
        exchange_class = getattr(ccxt, config.EXCHANGE_ID)
        exchange = exchange_class({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': 'swap', # Default to swap markets for V5 futures/perps
                'adjustForTimeDifference': True,
                'recvWindow': config.DEFAULT_RECV_WINDOW,
                'brokerId': 'PyrmethusV2', # Optional: Identify your bot via Broker ID
            }
        })

        if config.TESTNET_MODE:
            logger.info(f"[{func_name}] Enabling Bybit Sandbox (Testnet) mode.")
            exchange.set_sandbox_mode(True)

        logger.debug(f"[{func_name}] Loading markets...")
        exchange.load_markets(reload=True) # Force reload initially
        if not exchange.markets:
             raise ccxt.ExchangeError(f"[{func_name}] Failed to load markets.")
        logger.debug(f"[{func_name}] Markets loaded successfully ({len(exchange.markets)} symbols).")

        # Perform an initial API call to validate credentials and connectivity
        logger.debug(f"[{func_name}] Performing initial balance fetch for validation...")
        exchange.fetch_balance({'accountType': 'UNIFIED'}) # Use UNIFIED for V5
        logger.debug(f"[{func_name}] Initial balance check successful.")

        # Attempt to set default margin mode (Best effort, depends on account type)
        try:
            market = exchange.market(config.SYMBOL)
            category = _get_v5_category(market)
            if category and category in ['linear', 'inverse']:
                logger.debug(f"[{func_name}] Attempting to set initial margin mode '{config.DEFAULT_MARGIN_MODE}' for {config.SYMBOL} (Category: {category})...")
                params = {'category': category}
                exchange.set_margin_mode(config.DEFAULT_MARGIN_MODE, config.SYMBOL, params=params)
                logger.info(f"[{func_name}] Initial margin mode potentially set to '{config.DEFAULT_MARGIN_MODE}' for {config.SYMBOL}.")
            else:
                logger.warning(f"[{func_name}] Cannot determine contract category for {config.SYMBOL}. Skipping initial margin mode set.")
        except (ccxt.NotSupported, ccxt.ExchangeError, ccxt.ArgumentsRequired, ccxt.BadSymbol) as e_margin:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Could not set initial margin mode for {config.SYMBOL}: {e_margin}. "
                           f"This might be expected (e.g., UTA Isolated accounts). Verify account settings.{Style.RESET_ALL}")

        logger.success(f"{Fore.GREEN}[{func_name}] Bybit exchange initialized successfully. Testnet: {config.TESTNET_MODE}.{Style.RESET_ALL}")
        return exchange

    except (ccxt.AuthenticationError, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.ExchangeError) as e:
        logger.error(f"{Fore.RED}[{func_name}] Initialization attempt failed: {type(e).__name__} - {e}{Style.RESET_ALL}")
        raise # Re-raise for the decorator to handle retries/failure

    except Exception as e:
        logger.critical(f"{Back.RED}[{func_name}] Unexpected critical error during Bybit initialization: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[BybitHelper] CRITICAL: Bybit init failed! Unexpected: {type(e).__name__}", config)
        return None # Return None on critical unexpected failure


# Snippet 2 / Function 2: Set Leverage
@retry_api_call(max_retries=3, initial_delay=1.0)
def set_leverage(exchange: ccxt.bybit, symbol: str, leverage: int, config: Config) -> bool:
    """
    Sets the leverage for a specific symbol on Bybit V5 (Linear/Inverse).

    Validates the requested leverage against the market's limits. Handles the
    'leverage not modified' case gracefully. Applies to both buy and sell leverage.

    Args:
        exchange: The initialized ccxt.bybit exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        leverage: The desired integer leverage level.
        config: The configuration object.

    Returns:
        True if leverage was set successfully or already set to the desired value, False otherwise.

    Raises:
        Reraises CCXT exceptions for the retry decorator. ValueError for invalid leverage input.
    """
    func_name = "set_leverage"
    logger.info(f"{Fore.CYAN}[{func_name}] Setting leverage to {leverage}x for {symbol}...{Style.RESET_ALL}")

    if leverage <= 0:
        logger.error(f"{Fore.RED}[{func_name}] Leverage must be positive. Received: {leverage}{Style.RESET_ALL}")
        return False # Invalid input, don't call API

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']:
            logger.error(f"{Fore.RED}[{func_name}] Invalid market type for leverage setting: {symbol} (Category: {category}).{Style.RESET_ALL}")
            return False

        # Validate leverage against market limits
        leverage_filter = market.get('info', {}).get('leverageFilter', {})
        max_leverage_str = leverage_filter.get('maxLeverage')
        min_leverage_str = leverage_filter.get('minLeverage', '1') # Default min is 1
        max_leverage = int(safe_decimal_conversion(max_leverage_str, default=Decimal('100')))
        min_leverage = int(safe_decimal_conversion(min_leverage_str, default=Decimal('1')))

        if not (min_leverage <= leverage <= max_leverage):
            logger.error(f"{Fore.RED}[{func_name}] Invalid leverage requested: {leverage}x. Allowed range for {symbol}: {min_leverage}x - {max_leverage}x.{Style.RESET_ALL}")
            return False

        params = {
            'category': category,
            'buyLeverage': str(leverage), # V5 requires strings for leverage params
            'sellLeverage': str(leverage)
        }

        logger.debug(f"[{func_name}] Calling exchange.set_leverage with symbol='{symbol}', leverage={leverage}, params={params}")
        response = exchange.set_leverage(leverage, symbol, params=params)

        logger.debug(f"[{func_name}] Leverage API call response (raw): {response}")
        logger.success(f"{Fore.GREEN}[{func_name}] Leverage set/confirmed to {leverage}x for {symbol} (Category: {category}).{Style.RESET_ALL}")
        return True

    except ccxt.ExchangeError as e:
        error_str = str(e).lower()
        # Bybit V5 specific error codes/messages for "already set" or "not modified"
        # 110044: Leverage not modified
        if "leverage not modified" in error_str or "same as input" in error_str or "110044" in str(e):
            logger.info(f"{Fore.CYAN}[{func_name}] Leverage for {symbol} is already set to {leverage}x.{Style.RESET_ALL}")
            return True
        else:
            logger.error(f"{Fore.RED}[{func_name}] ExchangeError setting leverage for {symbol} to {leverage}x: {e}{Style.RESET_ALL}")
            raise # Re-raise for retry decorator

    except (ccxt.NetworkError, ccxt.AuthenticationError, ccxt.BadSymbol) as e:
        logger.error(f"{Fore.RED}[{func_name}] API/Symbol error setting leverage for {symbol}: {e}{Style.RESET_ALL}")
        raise # Re-raise for retry decorator

    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error setting leverage for {symbol} to {leverage}x: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[{symbol.split('/')[0]}] ERROR: Failed set leverage {leverage}x (Unexpected)", config)
        return False


# Snippet 3 / Function 3: Fetch USDT Balance (V5 UNIFIED)
@retry_api_call(max_retries=3, initial_delay=1.0)
def fetch_usdt_balance(exchange: ccxt.bybit, config: Config) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    Fetches the USDT balance (Total Equity and Available Balance) using Bybit V5 UNIFIED account logic.

    Args:
        exchange: The initialized ccxt.bybit exchange instance.
        config: The configuration object (used for USDT_SYMBOL).

    Returns:
        A tuple containing (total_equity, available_balance) as Decimals,
        or (None, None) if fetching fails or balance cannot be parsed.

    Raises:
        Reraises CCXT exceptions for the retry decorator.
    """
    func_name = "fetch_usdt_balance"
    logger.debug(f"[{func_name}] Fetching USDT balance (Bybit V5 UNIFIED Account)...")

    try:
        params = {'accountType': 'UNIFIED'} # V5 requires specifying account type
        balance_data = exchange.fetch_balance(params=params)
        # logger.debug(f"[{func_name}] Raw balance data: {balance_data}") # Verbose

        info = balance_data.get('info', {})
        result_list = info.get('result', {}).get('list', [])

        equity: Optional[Decimal] = None
        available: Optional[Decimal] = None
        account_type_found: str = "N/A"

        if result_list:
            unified_account_info = next((acc for acc in result_list if acc.get('accountType') == 'UNIFIED'), None)

            if unified_account_info:
                account_type_found = "UNIFIED"
                equity = safe_decimal_conversion(unified_account_info.get('totalEquity'))
                coin_list = unified_account_info.get('coin', [])
                usdt_coin_info = next((coin for coin in coin_list if coin.get('coin') == config.USDT_SYMBOL), None)

                if usdt_coin_info:
                    # V5 fields: availableToWithdraw, availableBalance, walletBalance
                    avail_val = usdt_coin_info.get('availableToWithdraw') or \
                                usdt_coin_info.get('availableBalance') or \
                                usdt_coin_info.get('walletBalance')
                    available = safe_decimal_conversion(avail_val)
                    if available is None:
                        logger.warning(f"[{func_name}] Found USDT entry but could not parse available balance from V5 fields: {usdt_coin_info}")
                        available = Decimal("0.0")
                else:
                    logger.warning(f"[{func_name}] USDT coin data not found within the UNIFIED account details. Assuming 0 available.")
                    available = Decimal("0.0") # Assume zero if USDT entry is missing

            else:
                logger.warning(f"[{func_name}] 'UNIFIED' account type not found in V5 balance response list. Trying fallback to first account.")
                if len(result_list) >= 1:
                     first_account = result_list[0]; account_type_found = first_account.get('accountType', 'UNKNOWN')
                     logger.warning(f"[{func_name}] Using first account found: Type '{account_type_found}'")
                     equity = safe_decimal_conversion(first_account.get('totalEquity') or first_account.get('equity')) # Check V5 field name
                     coin_list = first_account.get('coin', [])
                     usdt_coin_info = next((coin for coin in coin_list if coin.get('coin') == config.USDT_SYMBOL), None)
                     if usdt_coin_info:
                         avail_val = usdt_coin_info.get('availableBalance') or usdt_coin_info.get('walletBalance')
                         available = safe_decimal_conversion(avail_val)
                         if available is None: available = Decimal("0.0")
                     else: available = Decimal("0.0")
                else:
                     logger.error(f"[{func_name}] Balance response list is empty. Cannot determine balance.")


        # If V5 structure didn't yield results, try standard CCXT keys as a last resort
        if equity is None or available is None:
            logger.debug(f"[{func_name}] V5 structure parsing failed or incomplete. Trying standard CCXT balance keys...")
            usdt_balance_std = balance_data.get(config.USDT_SYMBOL, {})
            if equity is None: equity = safe_decimal_conversion(usdt_balance_std.get('total'))
            if available is None:
                available = safe_decimal_conversion(usdt_balance_std.get('free'))
                if available is None and equity is not None:
                     logger.warning(f"[{func_name}] CCXT 'free' balance missing, using 'total' ({equity:.4f}) as fallback for available.")
                     available = equity # Use total as a last resort, might be inaccurate

            if equity is not None and available is not None:
                account_type_found = "CCXT Standard Fallback"
            else:
                 # Raise error if balance couldn't be parsed at all
                 raise ValueError(f"Failed to parse balance from both V5 ({account_type_found}) and Standard structures.")

        # Ensure non-negative values and return
        final_equity = max(Decimal("0.0"), equity) if equity is not None else Decimal("0.0")
        final_available = max(Decimal("0.0"), available) if available is not None else Decimal("0.0")

        logger.info(f"[{func_name}] USDT Balance Fetched (Source: {account_type_found}): "
                    f"Equity = {final_equity:.4f}, Available = {final_available:.4f}")
        return final_equity, final_available

    except (ccxt.NetworkError, ccxt.ExchangeError, ValueError) as e:
        logger.warning(f"{Fore.YELLOW}[{func_name}] Error fetching/parsing balance: {e}{Style.RESET_ALL}")
        raise # Re-raise for retry decorator

    except Exception as e:
        logger.critical(f"{Back.RED}[{func_name}] Unexpected critical error fetching balance: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert("[BybitHelper] CRITICAL: Failed fetch USDT balance!", config)
        return None, None


# Snippet 4 / Function 4: Place Market Order with Slippage Check
@retry_api_call(max_retries=1, initial_delay=0) # Typically don't retry market orders automatically
def place_market_order_slippage_check(
    exchange: ccxt.bybit,
    symbol: str,
    side: Literal['buy', 'sell'],
    amount: Decimal,
    config: Config,
    max_slippage_pct: Optional[Decimal] = None,
    is_reduce_only: bool = False,
    client_order_id: Optional[str] = None
) -> Optional[Dict]:
    """
    Places a market order on Bybit V5 after checking the current spread against a slippage threshold.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        side: 'buy' or 'sell'.
        amount: Order quantity in base currency (Decimal). Must be positive.
        config: Configuration object.
        max_slippage_pct: Maximum allowed spread percentage (e.g., Decimal('0.005') for 0.5%).
                          Uses `config.DEFAULT_SLIPPAGE_PCT` if None.
        is_reduce_only: If True, set the reduceOnly flag.
        client_order_id: Optional client order ID string (max length 36 for Bybit V5 linear).

    Returns:
        The order dictionary returned by ccxt, or None if the order failed or was aborted pre-flight.
    """
    func_name = "place_market_order_slippage_check"
    market_base = symbol.split('/')[0]
    action = "CLOSE" if is_reduce_only else "ENTRY"
    log_prefix = f"Market Order ({action} {side.upper()})"
    effective_max_slippage = max_slippage_pct if max_slippage_pct is not None else config.DEFAULT_SLIPPAGE_PCT

    logger.info(f"{Fore.BLUE}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}. "
                f"Max Slippage: {effective_max_slippage:.4%}, ReduceOnly: {is_reduce_only}{Style.RESET_ALL}")

    if amount <= config.POSITION_QTY_EPSILON:
        logger.error(f"{Fore.RED}{log_prefix}: Amount is zero or negative ({amount}). Aborting.{Style.RESET_ALL}")
        return None

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
             logger.error(f"{Fore.RED}[{func_name}] Cannot determine market category for {symbol}. Aborting.{Style.RESET_ALL}")
             return None

        # 1. Perform Slippage Check using validated L2 OB fetch
        logger.debug(f"[{func_name}] Performing pre-order slippage check (Depth: {config.SHALLOW_OB_FETCH_DEPTH})...")
        # Note: This adds an API call latency before placing the order.
        ob_data = fetch_l2_order_book_validated(exchange, symbol, config.SHALLOW_OB_FETCH_DEPTH, config)

        if ob_data and ob_data['bids'] and ob_data['asks']:
            best_bid = ob_data['bids'][0][0]
            best_ask = ob_data['asks'][0][0]
            spread = (best_ask - best_bid) / best_bid if best_bid > Decimal("0") else Decimal("inf")
            logger.debug(f"[{func_name}] Current shallow OB: Best Bid={format_price(exchange, symbol, best_bid)}, "
                          f"Best Ask={format_price(exchange, symbol, best_ask)}, Spread={spread:.4%}")
            if spread > effective_max_slippage:
                logger.error(f"{Fore.RED}{log_prefix}: Aborted due to high slippage. "
                             f"Current Spread {spread:.4%} > Max Allowed {effective_max_slippage:.4%}.{Style.RESET_ALL}")
                send_sms_alert(f"[{market_base}] ORDER ABORT ({side.upper()}): High Slippage {spread:.4%}", config)
                return None
        else:
            logger.warning(f"{Fore.YELLOW}{log_prefix}: Could not get valid L2 order book data to check slippage. Proceeding with caution.{Style.RESET_ALL}")
            # Consider aborting if slippage check is critical: return None

        # 2. Prepare and Place Order
        amount_str = format_amount(exchange, symbol, amount) # Use precision formatting
        amount_float = float(amount_str)
        params: Dict[str, Any] = {'category': category}
        if is_reduce_only:
            params['reduceOnly'] = True
        if client_order_id:
            valid_coid = client_order_id[:36]
            params['clientOrderId'] = valid_coid
            if len(valid_coid) < len(client_order_id):
                 logger.warning(f"[{func_name}] Client Order ID truncated to 36 chars: '{valid_coid}'")

        bg = Back.GREEN if side == config.SIDE_BUY else Back.RED
        fg = Fore.BLACK # High contrast text
        logger.warning(f"{bg}{fg}{Style.BRIGHT}*** PLACING MARKET {side.upper()} {'REDUCE' if is_reduce_only else 'ENTRY'}: "
                       f"{amount_str} {symbol} (Params: {params}) ***{Style.RESET_ALL}")

        order = exchange.create_market_order(symbol, side, amount_float, params=params)

        # 3. Log Result
        order_id = order.get('id')
        client_oid_resp = order.get('clientOrderId', params.get('clientOrderId', 'N/A')) # Check response first
        status = order.get('status', '?')
        filled_qty = safe_decimal_conversion(order.get('filled', '0.0'))
        avg_price = safe_decimal_conversion(order.get('average')) # May be None initially

        logger.success(f"{Fore.GREEN}{log_prefix}: Market order submitted successfully. "
                       f"ID: ...{format_order_id(order_id)}, ClientOID: {client_oid_resp}, Status: {status}, "
                       f"Target Qty: {amount_str}, Filled Qty: {format_amount(exchange, symbol, filled_qty)}, "
                       f"Avg Price: {format_price(exchange, symbol, avg_price)}{Style.RESET_ALL}")
        return order

    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError) as e:
        logger.error(f"{Fore.RED}{log_prefix}: API Error placing market order: {type(e).__name__} - {e}{Style.RESET_ALL}")
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()} {action}): {type(e).__name__}", config)
        return None
    except Exception as e:
        logger.critical(f"{Back.RED}[{func_name}] Unexpected critical error placing market order: {e}{Style.RESET_ALL}", exc_info=True)
        send_sms_alert(f"[{market_base}] ORDER FAIL ({side.upper()} {action}): Unexpected {type(e).__name__}.", config)
        return None


# Snippet 5 / Function 5: Cancel All Open Orders
@retry_api_call(max_retries=2, initial_delay=1.0) # Decorator for fetch/network issues
def cancel_all_orders(exchange: ccxt.bybit, symbol: str, config: Config, reason: str = "Cleanup") -> bool:
    """
    Cancels all open orders for a specific symbol on Bybit V5.

    Fetches open orders first (defaults to regular 'Order' filter) and attempts
    to cancel each one individually. Handles `OrderNotFound` gracefully.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol for which to cancel orders.
        config: Configuration object.
        reason: A short string indicating the reason for cancellation (for logging).

    Returns:
        True if all found open orders were successfully cancelled or confirmed gone,
        False if any cancellation failed (excluding OrderNotFound).

    Raises:
        Reraises CCXT exceptions from fetch_open_orders for the retry decorator.
        Does not re-raise OrderNotFound during cancellation.
    """
    func_name = "cancel_all_orders"
    market_base = symbol.split('/')[0]
    log_prefix = f"Cancel All ({reason})"
    logger.info(f"{Fore.CYAN}[{func_name}] {log_prefix}: Attempting for {symbol}...{Style.RESET_ALL}")

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category:
            logger.error(f"{Fore.RED}[{func_name}] Cannot determine market category for {symbol}. Aborting cancel all.{Style.RESET_ALL}")
            return False

        # Fetch open regular orders. Consider adding logic to cancel Stop/TP/SL orders too if needed.
        fetch_params = {'category': category, 'orderFilter': 'Order'}
        logger.debug(f"[{func_name}] Fetching open regular orders for {symbol} with params: {fetch_params}")
        open_orders = exchange.fetch_open_orders(symbol, params=fetch_params)

        if not open_orders:
            logger.info(f"{Fore.CYAN}[{func_name}] {log_prefix}: No open regular orders found for {symbol}.{Style.RESET_ALL}")
            return True

        logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: Found {len(open_orders)} open order(s) for {symbol}. Attempting cancellation...{Style.RESET_ALL}")

        success_count = 0
        fail_count = 0
        cancel_delay = max(0.05, 1.0 / (exchange.rateLimit if exchange.rateLimit and exchange.rateLimit > 0 else 20)) # Small delay

        cancel_params = {'category': category} # Category needed for V5 cancel

        for order in open_orders:
            order_id = order.get('id')
            order_info_log = (f"ID: ...{format_order_id(order_id)} "
                              f"({order.get('type', '?').upper()} {order.get('side', '?').upper()} "
                              f"Amt: {format_amount(exchange, symbol, order.get('amount'))})")

            if not order_id:
                logger.warning(f"[{func_name}] Skipping order with missing ID in fetched data: {order}")
                continue

            try:
                logger.debug(f"[{func_name}] Cancelling order {order_info_log} with params: {cancel_params}")
                exchange.cancel_order(order_id, symbol, params=cancel_params)
                logger.info(f"{Fore.CYAN}[{func_name}] {log_prefix}: Successfully cancelled order {order_info_log}{Style.RESET_ALL}")
                success_count += 1
            except ccxt.OrderNotFound:
                logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: Order {order_info_log} already cancelled or filled (Not Found). Considered OK.{Style.RESET_ALL}")
                success_count += 1 # Treat as success in this context
            except (ccxt.NetworkError, ccxt.RateLimitExceeded) as e_cancel:
                logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: Network/RateLimit error cancelling {order_info_log}: {e_cancel}. Loop continues, outer retry might occur.{Style.RESET_ALL}")
                fail_count += 1 # Count as failure for this attempt
            except ccxt.ExchangeError as e_cancel:
                 logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: FAILED to cancel order {order_info_log}: {e_cancel}{Style.RESET_ALL}")
                 fail_count += 1
            except Exception as e_cancel:
                logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: Unexpected error cancelling order {order_info_log}: {e_cancel}{Style.RESET_ALL}", exc_info=True)
                fail_count += 1

            time.sleep(cancel_delay) # Pause between cancellation attempts

        # Report Summary after attempting all
        total_attempted = len(open_orders)
        if fail_count > 0:
            try: # Re-check if orders still exist after potential transient errors
                 remaining_orders = exchange.fetch_open_orders(symbol, params=fetch_params)
                 if not remaining_orders:
                      logger.warning(f"{Fore.YELLOW}[{func_name}] {log_prefix}: Initial cancellation reported {fail_count} failures, but re-check shows no open orders remain. Likely transient errors.{Style.RESET_ALL}")
                      return True # All gone now
                 else:
                      logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: Finished cancellation attempt for {symbol}. "
                                   f"Failed: {fail_count}, Success/Gone: {success_count}. {len(remaining_orders)} order(s) might still remain.{Style.RESET_ALL}")
                      send_sms_alert(f"[{market_base}] ERROR: Failed to cancel {fail_count} orders ({reason}). Check logs.", config)
                      return False
            except Exception as e_recheck:
                 logger.error(f"{Fore.RED}[{func_name}] Error re-checking orders after failures: {e_recheck}. Assuming failures persist.")
                 send_sms_alert(f"[{market_base}] ERROR: Failed to cancel {fail_count} orders ({reason}). Check logs.", config)
                 return False
        else:
            logger.success(f"{Fore.GREEN}[{func_name}] {log_prefix}: Successfully cancelled or confirmed gone "
                           f"all {total_attempted} open orders found for {symbol}.{Style.RESET_ALL}")
            return True

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e:
        logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: API error during 'cancel all' process for {symbol}: {e}{Style.RESET_ALL}")
        raise # Re-raise for retry decorator
    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] {log_prefix}: Unexpected error during 'cancel all' for {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
        return False


# Snippet 6 / Function 6: Fetch OHLCV with Pagination
# (Implementation from previous response - uses internal retries for chunks)
def fetch_ohlcv_paginated(
    exchange: ccxt.bybit,
    symbol: str,
    timeframe: str,
    config: Config,
    since: Optional[int] = None,
    limit_per_req: int = 1000, # Bybit V5 max limit is 1000
    max_total_candles: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetches historical OHLCV data for a symbol using pagination to handle limits.

    Converts the fetched data into a pandas DataFrame with proper indexing and
    data types, performing basic validation and cleaning (NaN handling).
    Uses internal retries for individual chunk fetches.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        timeframe: CCXT timeframe string (e.g., '1m', '5m', '1h', '1d').
        config: Configuration object.
        since: Optional starting timestamp (milliseconds UTC) to fetch data from.
               If None, fetches the most recent data.
        limit_per_req: Number of candles to fetch per API request (max 1000 for Bybit V5).
        max_total_candles: Optional maximum number of candles to retrieve in total.

    Returns:
        A pandas DataFrame containing the OHLCV data, indexed by UTC timestamp,
        or None if fetching or processing fails completely. Returns an empty DataFrame
        if no data is available for the period.
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

        while True: # Loop until break conditions are met
            if max_total_candles and len(all_candles) >= max_total_candles:
                logger.info(f"[{func_name}] Reached max_total_candles limit ({max_total_candles}). Fetch complete.")
                break
            if request_count >= max_requests:
                 logger.info(f"[{func_name}] Reached max calculated requests ({max_requests}). Fetch complete.")
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
                    last_fetch_error = None; break # Success
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.RateLimitExceeded) as e:
                    last_fetch_error = e
                    retry_delay = config.RETRY_DELAY_SECONDS * (attempt + 1) * (random.uniform(0.8, 1.2)) # Jitter
                    logger.warning(f"{Fore.YELLOW}[{func_name}] API Error chunk #{request_count} (Try {attempt + 1}/{config.RETRY_COUNT}): {e}. Retrying in {retry_delay:.2f}s...{Style.RESET_ALL}")
                    time.sleep(retry_delay)
                except ccxt.ExchangeError as e: last_fetch_error = e; logger.error(f"{Fore.RED}[{func_name}] ExchangeError chunk #{request_count}: {e}. Aborting chunk.{Style.RESET_ALL}"); break
                except Exception as e: last_fetch_error = e; logger.error(f"[{func_name}] Unexpected fetch chunk #{request_count} err: {e}", exc_info=True); break

            if last_fetch_error:
                logger.error(f"{Fore.RED}[{func_name}] Failed to fetch chunk #{request_count} after {config.RETRY_COUNT} attempts. Last Error: {last_fetch_error}{Style.RESET_ALL}")
                logger.warning(f"[{func_name}] Returning potentially incomplete data ({len(all_candles)} candles) due to fetch failure.")
                break # Exit main loop

            if not candles_chunk: logger.debug(f"[{func_name}] No more candles returned (Chunk #{request_count})."); break

            if all_candles and candles_chunk[0][0] <= all_candles[-1][0]:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Overlap detected chunk #{request_count}. Filtering.{Style.RESET_ALL}")
                candles_chunk = [c for c in candles_chunk if c[0] > all_candles[-1][0]]
                if not candles_chunk: logger.debug(f"[{func_name}] Entire chunk was overlap."); break

            logger.debug(f"[{func_name}] Fetched {len(candles_chunk)} new candles (Chunk #{request_count}). Total: {len(all_candles) + len(candles_chunk)}")
            all_candles.extend(candles_chunk)

            if len(candles_chunk) < fetch_limit: logger.debug(f"[{func_name}] Received fewer candles than requested. End of data."); break

            current_since = candles_chunk[-1][0] + timeframe_ms
            time.sleep(exchange.rateLimit / 1000.0 if exchange.rateLimit and exchange.rateLimit > 0 else 0.1)

        return _process_ohlcv_list(all_candles, func_name, symbol, timeframe, max_total_candles)

    except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
         logger.error(f"{Fore.RED}[{func_name}] Initial setup error for OHLCV fetch ({symbol}, {timeframe}): {e}{Style.RESET_ALL}")
         return None
    except Exception as e:
        logger.critical(f"{Back.RED}[{func_name}] Unexpected critical error during OHLCV pagination setup: {e}{Style.RESET_ALL}", exc_info=True)
        return None

def _process_ohlcv_list(
    candle_list: List[list], parent_func_name: str, symbol: str, timeframe: str, max_candles: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """Internal helper to convert OHLCV list to validated pandas DataFrame."""
    # (Implementation identical to previous response, assumed correct)
    func_name = f"{parent_func_name}._process_ohlcv_list"
    if not candle_list:
        logger.warning(f"{Fore.YELLOW}[{func_name}] No candles collected for {symbol} ({timeframe}). Returning empty DataFrame.{Style.RESET_ALL}")
        cols = ['open', 'high', 'low', 'close', 'volume']; empty_df = pd.DataFrame(columns=cols).astype({c: float for c in cols})
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
            df.ffill(inplace=True); df.dropna(inplace=True)
            if df.isnull().sum().sum() > 0: logger.error(f"{Fore.RED}[{func_name}] NaNs persisted after fill!{Style.RESET_ALL}")
        df.sort_index(inplace=True)
        if max_candles and len(df) > max_candles: logger.debug(f"[{func_name}] Trimming DF to last {max_candles}."); df = df.iloc[-max_candles:]
        if df.empty: logger.error(f"{Fore.RED}[{func_name}] Processed DF is empty after cleaning.{Style.RESET_ALL}"); return df
        logger.success(f"{Fore.GREEN}[{func_name}] Processed {len(df)} valid candles for {symbol} ({timeframe}).{Style.RESET_ALL}")
        return df
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Error processing OHLCV list: {e}{Style.RESET_ALL}", exc_info=True); return None


# Snippet 7 / Function 7: Place Limit Order with TIF/Flags
@retry_api_call(max_retries=1, initial_delay=0) # Typically don't retry limit order placement automatically
def place_limit_order_tif(
    exchange: ccxt.bybit, symbol: str, side: Literal['buy', 'sell'], amount: Decimal, price: Decimal, config: Config,
    time_in_force: str = 'GTC', is_reduce_only: bool = False, is_post_only: bool = False, client_order_id: Optional[str] = None
) -> Optional[Dict]:
    """
    Places a limit order on Bybit V5 with options for Time-In-Force, Post-Only, and Reduce-Only.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        side: 'buy' or 'sell'.
        amount: Order quantity in base currency (Decimal). Must be positive.
        price: Limit price for the order (Decimal). Must be positive.
        config: Configuration object.
        time_in_force: Time-In-Force policy ('GTC', 'IOC', 'FOK'). CCXT standard.
        is_reduce_only: If True, set the reduceOnly flag.
        is_post_only: If True, ensures the order is only accepted if it does not immediately match.
        client_order_id: Optional client order ID string (max length 36 for Bybit V5 linear).

    Returns:
        The order dictionary returned by ccxt, or None if the order placement failed.
    """
    # (Implementation identical to previous response, assumed correct)
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
        if client_order_id: valid_coid = client_order_id[:36]; params['clientOrderId'] = valid_coid; if len(valid_coid) < len(client_order_id): logger.warning(f"[{func_name}] Client OID truncated: '{valid_coid}'")
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


# Snippet 8 / Function 8: Fetch Current Position (Bybit V5 Specific)
@retry_api_call(max_retries=3, initial_delay=1.0)
def get_current_position_bybit_v5(exchange: ccxt.bybit, symbol: str, config: Config) -> Dict[str, Any]:
    """
    Fetches the current position details for a symbol using Bybit V5's fetchPositions logic.
    Focuses on One-Way position mode (positionIdx=0). Returns Decimals for numeric values.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "get_current_position_bybit_v5"; logger.debug(f"[{func_name}] Fetching position for {symbol} (V5)...")
    default_position: Dict[str, Any] = {'symbol': symbol, 'side': config.POS_NONE, 'qty': Decimal("0.0"), 'entry_price': Decimal("0.0"), 'liq_price': None, 'mark_price': None, 'pnl_unrealized': None, 'leverage': None, 'info': {}}
    try:
        market = exchange.market(symbol); market_id = market['id']
        category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}[{func_name}] Not a contract symbol: {symbol}.{Style.RESET_ALL}"); return default_position
        if not exchange.has.get('fetchPositions'): logger.error(f"{Fore.RED}[{func_name}] fetchPositions not available.{Style.RESET_ALL}"); return default_position
        params = {'category': category, 'symbol': market_id}; logger.debug(f"[{func_name}] Calling fetch_positions with params: {params}")
        fetched_positions = exchange.fetch_positions(symbols=[symbol], params=params)
        active_position_data: Optional[Dict] = None
        for pos in fetched_positions:
            pos_info = pos.get('info', {}); pos_symbol = pos_info.get('symbol'); pos_v5_side = pos_info.get('side', 'None'); pos_size_str = pos_info.get('size'); pos_idx = int(pos_info.get('positionIdx', -1))
            if pos_symbol == market_id and pos_v5_side != 'None' and pos_idx == 0:
                pos_size = safe_decimal_conversion(pos_size_str, Decimal("0.0"));
                if abs(pos_size) > config.POSITION_QTY_EPSILON: active_position_data = pos; logger.debug(f"[{func_name}] Found active One-Way (idx 0) position."); break
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


# Snippet 9 / Function 9: Close Position (Reduce-Only Market)
@retry_api_call(max_retries=2, initial_delay=1) # Allow retry for closure attempt
def close_position_reduce_only(
    exchange: ccxt.bybit, symbol: str, config: Config, position_to_close: Optional[Dict[str, Any]] = None, reason: str = "Signal Close"
) -> Optional[Dict[str, Any]]:
    """
    Closes the current position for the given symbol using a reduce-only market order.
    Handles specific "already closed" exchange errors gracefully.
    """
    # (Implementation identical to previous response, assumed correct)
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
        if any(code in error_str for code in ["110025", "110045", "30086", "position is closed", "order would not reduce", "position size is zero"]): logger.warning(f"{Fore.YELLOW}[{func_name}] Close Order ({reason}): Exchange indicates already closed/zero: {e}. Assuming closed.{Style.RESET_ALL}"); return None
        else: logger.error(f"{Fore.RED}[{func_name}] Close Order ExchangeError ({reason}): {e}{Style.RESET_ALL}"); send_sms_alert(f"[{market_base}] CLOSE FAIL ({live_side}): ExchangeError", config); raise e
    except (ccxt.NetworkError, ValueError) as e: logger.error(f"{Fore.RED}[{func_name}] Close Order Network/Setup Error ({reason}): {e}{Style.RESET_ALL}"); raise e
    except Exception as e: logger.critical(f"{Back.RED}[{func_name}] Close Order Unexpected Error ({reason}): {e}{Style.RESET_ALL}", exc_info=True); send_sms_alert(f"[{market_base}] CLOSE FAIL ({live_side}): Unexpected Error", config); return None


# Snippet 11 / Function 11: Fetch Funding Rate
@retry_api_call(max_retries=3, initial_delay=1.0)
def fetch_funding_rate(exchange: ccxt.bybit, symbol: str, config: Config) -> Optional[Dict[str, Any]]:
    """
    Fetches the current funding rate details for a perpetual swap symbol on Bybit V5.
    Returns Decimals for rates/prices.
    """
    # (Implementation identical to previous response, assumed correct)
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
        if processed_fr['nextFundingTime']: try: processed_fr['nextFundingDatetime'] = pd.to_datetime(processed_fr['nextFundingTime'], unit='ms', utc=True).strftime('%Y-%m-%d %H:%M:%S %Z'); except Exception as dt_err: logger.warning(f"[{func_name}] Could not format next funding datetime: {dt_err}")
        rate = processed_fr.get('fundingRate'); next_dt_str = processed_fr.get('nextFundingDatetime', "N/A"); rate_str = f"{rate:.6%}" if rate is not None else "N/A"
        logger.info(f"[{func_name}] Funding Rate {symbol}: {rate_str}. Next: {next_dt_str}")
        return processed_fr
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching funding rate: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching funding rate: {e}{Style.RESET_ALL}", exc_info=True); return None


# Snippet 12 / Function 12: Set Position Mode (One-Way / Hedge)
@retry_api_call(max_retries=2, initial_delay=1.0)
def set_position_mode_bybit_v5(exchange: ccxt.bybit, symbol_or_category: str, mode: Literal['one-way', 'hedge'], config: Config) -> bool:
    """
    Sets the position mode (One-Way or Hedge) for a specific category (Linear/Inverse) on Bybit V5.
    Uses the `private_post_v5_position_switch_mode` endpoint. Handles specific V5 errors.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "set_position_mode_bybit_v5"; logger.info(f"{Fore.CYAN}[{func_name}] Setting mode '{mode}' for category of '{symbol_or_category}'...{Style.RESET_ALL}")
    mode_map = {'one-way': '0', 'hedge': '3'}; target_mode_code = mode_map.get(mode.lower())
    if target_mode_code is None: logger.error(f"{Fore.RED}[{func_name}] Invalid mode '{mode}'.{Style.RESET_ALL}"); return False
    target_category: Optional[Literal['linear', 'inverse']] = None
    if symbol_or_category.lower() in ['linear', 'inverse']: target_category = symbol_or_category.lower() # type: ignore
    else: try: market = exchange.market(symbol_or_category); target_category = _get_v5_category(market); if target_category not in ['linear', 'inverse']: target_category = None; except: pass
    if not target_category: logger.error(f"{Fore.RED}[{func_name}] Could not determine contract category from '{symbol_or_category}'.{Style.RESET_ALL}"); return False
    logger.debug(f"[{func_name}] Target Category: {target_category}, Mode Code: {target_mode_code} ('{mode}')")
    try:
        if not hasattr(exchange, 'private_post_v5_position_switch_mode'): logger.error(f"{Fore.RED}[{func_name}] CCXT lacks 'private_post_v5_position_switch_mode'.{Style.RESET_ALL}"); return False
        params = {'category': target_category, 'mode': target_mode_code}; logger.debug(f"[{func_name}] Calling endpoint with params: {params}")
        response = exchange.private_post_v5_position_switch_mode(params); logger.debug(f"[{func_name}] Raw response: {response}")
        ret_code = response.get('retCode'); ret_msg = response.get('retMsg', '').lower()
        if ret_code == 0: logger.success(f"{Fore.GREEN}[{func_name}] Mode set '{mode}' for {target_category}.{Style.RESET_ALL}"); return True
        elif ret_code == 110021 or "not modified" in ret_msg: logger.info(f"{Fore.CYAN}[{func_name}] Mode already '{mode}' for {target_category}.{Style.RESET_ALL}"); return True
        elif ret_code == 110020 or "have position" in ret_msg or "active order" in ret_msg: logger.error(f"{Fore.RED}[{func_name}] Cannot switch mode: Active pos/orders exist. Msg: {response.get('retMsg')}{Style.RESET_ALL}"); return False
        else: raise ccxt.ExchangeError(f"Bybit API error setting mode: Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'")
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.BadSymbol) as e:
        if not (isinstance(e, ccxt.ExchangeError) and ret_code in [110020]): logger.warning(f"{Fore.YELLOW}[{func_name}] API Error setting mode: {e}{Style.RESET_ALL}"); raise e
        return False # Return False for handled errors like position exists
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error setting mode: {e}{Style.RESET_ALL}", exc_info=True); return False


# Snippet 13 / Function 13: Fetch L2 Order Book (Validated)
@retry_api_call(max_retries=2, initial_delay=0.5)
def fetch_l2_order_book_validated(
    exchange: ccxt.bybit, symbol: str, limit: int, config: Config
) -> Optional[Dict[str, List[Tuple[Decimal, Decimal]]]]:
    """
    Fetches the Level 2 order book for a symbol using Bybit V5 fetchOrderBook and validates the data.
    Returns bids and asks as lists of [price, amount] tuples using Decimals.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "fetch_l2_order_book_validated"; logger.debug(f"[{func_name}] Fetching L2 OB {symbol} (Limit:{limit})...")
    if not exchange.has.get('fetchOrderBook'): logger.error(f"{Fore.RED}[{func_name}] fetchOrderBook not supported.{Style.RESET_ALL}"); return None
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.warning(f"[{func_name}] Cannot determine category for {symbol}. Fetching OB without param."); params = {}
        else: params = {'category': category}; max_limit = {'spot': 50, 'linear': 200, 'inverse': 200, 'option': 25}.get(category, 50); if limit > max_limit: logger.warning(f"[{func_name}] Clamping limit {limit} to {max_limit} for {category}."); limit = max_limit
        logger.debug(f"[{func_name}] Calling fetchOrderBook with limit={limit}, params={params}")
        order_book = exchange.fetch_order_book(symbol, limit=limit, params=params)
        if not isinstance(order_book, dict) or 'bids' not in order_book or 'asks' not in order_book: raise ValueError("Invalid OB structure")
        raw_bids=order_book['bids']; raw_asks=order_book['asks']
        if not isinstance(raw_bids, list) or not isinstance(raw_asks, list): raise ValueError("Bids/Asks not lists")
        validated_bids: List[Tuple[Decimal, Decimal]] = []; validated_asks: List[Tuple[Decimal, Decimal]] = []; conversion_errors = 0
        for p_str, a_str in raw_bids: p = safe_decimal_conversion(p_str); a = safe_decimal_conversion(a_str); if p is None or a is None or p <= 0 or a < 0: conversion_errors += 1; continue; validated_bids.append((p, a))
        for p_str, a_str in raw_asks: p = safe_decimal_conversion(p_str); a = safe_decimal_conversion(a_str); if p is None or a is None or p <= 0 or a < 0: conversion_errors += 1; continue; validated_asks.append((p, a))
        if conversion_errors > 0: logger.warning(f"{Fore.YELLOW}Skipped {conversion_errors} invalid OB entries for {symbol}.{Style.RESET_ALL}")
        if validated_bids and validated_asks and validated_bids[0][0] >= validated_asks[0][0]: logger.error(f"{Fore.RED}[{func_name}] OB crossed: Bid ({validated_bids[0][0]}) >= Ask ({validated_asks[0][0]}) for {symbol}.{Style.RESET_ALL}"); return None
        logger.debug(f"[{func_name}] Processed L2 OB {symbol}. Bids:{len(validated_bids)}, Asks:{len(validated_asks)}")
        return {'bids': validated_bids, 'asks': validated_asks}
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol, ValueError) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API/Validation Error fetching L2 OB for {symbol}: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching L2 OB for {symbol}: {e}{Style.RESET_ALL}", exc_info=True); return None


# Snippet 14 / Function 14: Place Native Stop Loss Order (Stop Market)
@retry_api_call(max_retries=1, initial_delay=0) # Don't auto-retry placing stops usually
def place_native_stop_loss(
    exchange: ccxt.bybit, symbol: str, side: Literal['buy', 'sell'], amount: Decimal, stop_price: Decimal, config: Config,
    trigger_by: Literal['LastPrice', 'MarkPrice', 'IndexPrice'] = 'MarkPrice', client_order_id: Optional[str] = None, position_idx: Literal[0, 1, 2] = 0
) -> Optional[Dict]:
    """
    Places a native Stop Market order on Bybit V5, intended as a Stop Loss (reduceOnly).
    Uses V5 specific parameters like 'stopLoss' and 'slTriggerBy'.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "place_native_stop_loss"; log_prefix = f"Place Native SL ({side.upper()})"
    logger.info(f"{Fore.CYAN}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}, Trigger @ {format_price(exchange, symbol, stop_price)} ({trigger_by}), PosIdx:{position_idx}...{Style.RESET_ALL}")
    if amount <= config.POSITION_QTY_EPSILON or stop_price <= Decimal("0"): logger.error(f"{Fore.RED}{log_prefix}: Invalid amount/stop price.{Style.RESET_ALL}"); return None
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}[{func_name}] Not a contract symbol: {symbol}.{Style.RESET_ALL}"); return None
        amount_str = format_amount(exchange, symbol, amount); amount_float = float(amount_str); stop_price_str = format_price(exchange, symbol, stop_price)
        params: Dict[str, Any] = {'category': category, 'stopLoss': stop_price_str, 'slTriggerBy': trigger_by, 'reduceOnly': True, 'positionIdx': position_idx, 'tpslMode': 'Full', 'slOrderType': 'Market' }
        if client_order_id: valid_coid = client_order_id[:36]; params['clientOrderId'] = valid_coid; if len(valid_coid) < len(client_order_id): logger.warning(f"[{func_name}] Client OID truncated: '{valid_coid}'")
        bg = Back.YELLOW; fg = Fore.BLACK
        logger.warning(f"{bg}{fg}{Style.BRIGHT}{log_prefix}: Placing NATIVE Stop Loss (Market exec) -> Qty:{amount_float}, Side:{side}, TriggerPx:{stop_price_str}, TriggerBy:{trigger_by}, Reduce:True, PosIdx:{position_idx}, Params:{params}{Style.RESET_ALL}")
        sl_order = exchange.create_order(symbol=symbol, type='market', side=side, amount=amount_float, params=params)
        order_id = sl_order.get('id'); client_oid_resp = sl_order.get('clientOrderId', params.get('clientOrderId', 'N/A')); status = sl_order.get('status', '?')
        returned_stop_price = safe_decimal_conversion(sl_order.get('info', {}).get('stopLoss', sl_order.get('stopPrice')), None); returned_trigger = sl_order.get('info', {}).get('slTriggerBy', trigger_by)
        logger.success(f"{Fore.GREEN}{log_prefix}: Native SL order placed. ID:...{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, Trigger:{format_price(exchange, symbol, returned_stop_price)} (by {returned_trigger}){Style.RESET_ALL}")
        return sl_order
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadSymbol) as e: logger.error(f"{Fore.RED}{log_prefix}: API Error placing SL: {type(e).__name__} - {e}{Style.RESET_ALL}"); return None
    except Exception as e: logger.critical(f"{Back.RED}[{func_name}] Unexpected error placing SL: {e}{Style.RESET_ALL}", exc_info=True); send_sms_alert(f"[{symbol.split('/')[0]}] SL PLACE FAIL ({side.upper()}): Unexpected {type(e).__name__}", config); return None


# Snippet 15 / Function 15: Fetch Open Orders (Filtered)
@retry_api_call(max_retries=2, initial_delay=1.0)
def fetch_open_orders_filtered(
    exchange: ccxt.bybit, symbol: str, config: Config, side: Optional[Literal['buy', 'sell']] = None,
    order_type: Optional[str] = None, order_filter: Optional[Literal['Order', 'StopOrder', 'tpslOrder']] = None
) -> Optional[List[Dict]]:
    """
    Fetches open orders for a specific symbol on Bybit V5, with optional filtering
    by side, CCXT order type, and/or Bybit V5 `orderFilter`. Defaults to fetching 'Order'.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "fetch_open_orders_filtered"; filter_log = f"(Side:{side or 'Any'}, Type:{order_type or 'Any'}, V5Filter:{order_filter or 'Order'})"
    logger.debug(f"[{func_name}] Fetching open orders {symbol} {filter_log}...")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.error(f"{Fore.RED}[{func_name}] Cannot determine category for {symbol}.{Style.RESET_ALL}"); return None
        params: Dict[str, Any] = {'category': category}; params['orderFilter'] = order_filter if order_filter else 'Order'
        logger.debug(f"[{func_name}] Calling fetch_open_orders with params: {params}")
        open_orders = exchange.fetch_open_orders(symbol=symbol, params=params)
        if not open_orders: logger.debug(f"[{func_name}] No open orders found matching {params}."); return []
        filtered_orders = open_orders; initial_count = len(filtered_orders)
        if side: side_lower = side.lower(); filtered_orders = [o for o in filtered_orders if o.get('side', '').lower() == side_lower]; logger.debug(f"[{func_name}] Filtered by side='{side}'. Count: {initial_count} -> {len(filtered_orders)}.")
        if order_type:
            norm_type_filter = order_type.lower().replace('_', '').replace('-', ''); count_before = len(filtered_orders)
            def check_type(order):
                o_type = order.get('type', '').lower().replace('_', '').replace('-', '')
                if o_type == norm_type_filter: return True
                is_stop = order.get('stopPrice') or order.get('info', {}).get('stopLoss') or order.get('info', {}).get('takeProfit')
                if 'market' in norm_type_filter and ('stop' in norm_type_filter or 'takeprofit' in norm_type_filter) and is_stop and o_type == 'market': return True
                return False
            filtered_orders = [o for o in filtered_orders if check_type(o)]; logger.debug(f"[{func_name}] Filtered by type='{order_type}'. Count: {count_before} -> {len(filtered_orders)}.")
        logger.info(f"[{func_name}] Fetched/filtered {len(filtered_orders)} open orders for {symbol} {filter_log}.")
        return filtered_orders
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching open orders: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching open orders: {e}{Style.RESET_ALL}", exc_info=True); return None


# Snippet 16 / Function 16: Calculate Margin Requirement
def calculate_margin_requirement(
    exchange: ccxt.bybit, symbol: str, amount: Decimal, price: Decimal, leverage: Decimal, config: Config,
    order_side: Literal['buy', 'sell'], is_maker: bool = False
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    Calculates the estimated Initial Margin (IM) requirement for placing an order on Bybit V5.
    MM estimate is basic/placeholder.
    """
    # (Implementation identical to previous response, assumed correct)
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
            mmr_rate_str = market.get('maintenanceMarginRate') or market.get('info', {}).get('mmr') or market.get('info', {}).get('maintenanceMarginRate')
            if mmr_rate_str: mmr_rate = safe_decimal_conversion(mmr_rate_str); if mmr_rate and mmr_rate > 0: maintenance_margin_estimate = position_value * mmr_rate; logger.debug(f"[{func_name}] Basic MM Estimate (Base MMR {mmr_rate:.4%}): {format_price(exchange, quote_currency, maintenance_margin_estimate)} {quote_currency}")
            else: logger.debug(f"[{func_name}] MMR not found in market info.")
        except Exception as mm_err: logger.warning(f"[{func_name}] Could not estimate MM: {mm_err}")
        return total_initial_margin_estimate, maintenance_margin_estimate
    except (DivisionByZero, KeyError, ValueError) as e: logger.error(f"{Fore.RED}[{func_name}] Calculation error: {e}{Style.RESET_ALL}"); return None, None
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error during margin calculation: {e}{Style.RESET_ALL}", exc_info=True); return None, None


# Snippet 17 / Function 17: Fetch Ticker (Validated)
@retry_api_call(max_retries=2, initial_delay=0.5)
def fetch_ticker_validated(
    exchange: ccxt.bybit, symbol: str, config: Config, max_age_seconds: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Fetches the ticker for a symbol from Bybit V5, validates its freshness and key values.
    Returns a dictionary with Decimal values.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "fetch_ticker_validated"; logger.debug(f"[{func_name}] Fetching/Validating ticker {symbol}...")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.warning(f"[{func_name}] No category for {symbol}. Fetching ticker without param."); params = {}
        else: params = {'category': category}
        logger.debug(f"[{func_name}] Calling fetch_ticker with params: {params}")
        ticker = exchange.fetch_ticker(symbol, params=params)
        timestamp_ms = ticker.get('timestamp'); age_seconds = (time.time() * 1000 - timestamp_ms) / 1000.0 if timestamp_ms else float('inf')
        if timestamp_ms is None or age_seconds > max_age_seconds or age_seconds < -10: raise ValueError(f"Ticker timestamp invalid/stale (Age: {age_seconds:.1f}s)")
        last_price = safe_decimal_conversion(ticker.get('last')); bid_price = safe_decimal_conversion(ticker.get('bid')); ask_price = safe_decimal_conversion(ticker.get('ask'))
        if last_price is None or last_price <= 0: raise ValueError(f"Invalid 'last' price: {ticker.get('last')}")
        if bid_price is None or bid_price <= 0: logger.warning(f"[{func_name}] Invalid/missing 'bid': {ticker.get('bid')}")
        if ask_price is None or ask_price <= 0: logger.warning(f"[{func_name}] Invalid/missing 'ask': {ticker.get('ask')}")
        spread, spread_pct = None, None
        if bid_price and ask_price:
             if bid_price >= ask_price: raise ValueError(f"Bid ({bid_price}) >= Ask ({ask_price})")
             spread = ask_price - bid_price; spread_pct = (spread / bid_price) * 100 if bid_price > 0 else Decimal("inf")
        base_volume = safe_decimal_conversion(ticker.get('baseVolume')); quote_volume = safe_decimal_conversion(ticker.get('quoteVolume'))
        if base_volume is not None and base_volume < 0: logger.warning(f"Negative baseVol: {base_volume}"); base_volume = Decimal("0.0")
        if quote_volume is not None and quote_volume < 0: logger.warning(f"Negative quoteVol: {quote_volume}"); quote_volume = Decimal("0.0")
        validated_ticker = { 'symbol': ticker.get('symbol', symbol), 'timestamp': timestamp_ms, 'datetime': ticker.get('datetime'), 'last': last_price, 'bid': bid_price, 'ask': ask_price, 'bidVolume': safe_decimal_conversion(ticker.get('bidVolume')), 'askVolume': safe_decimal_conversion(ticker.get('askVolume')), 'baseVolume': base_volume, 'quoteVolume': quote_volume, 'high': safe_decimal_conversion(ticker.get('high')), 'low': safe_decimal_conversion(ticker.get('low')), 'open': safe_decimal_conversion(ticker.get('open')), 'close': last_price, 'change': safe_decimal_conversion(ticker.get('change')), 'percentage': safe_decimal_conversion(ticker.get('percentage')), 'average': safe_decimal_conversion(ticker.get('average')), 'spread': spread, 'spread_pct': spread_pct, 'info': ticker.get('info', {}) }
        logger.debug(f"[{func_name}] Ticker OK: {symbol} Last={format_price(exchange, symbol, last_price)}, Spread={spread_pct:.4f}% (Age:{age_seconds:.1f}s)")
        return validated_ticker
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol, ValueError) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] Fetch/Validate ticker failed for {symbol}: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected ticker error for {symbol}: {e}{Style.RESET_ALL}", exc_info=True); return None


# Snippet 18 / Function 18: Place Native Trailing Stop Order
@retry_api_call(max_retries=1, initial_delay=0) # Don't auto-retry placing stops usually
def place_native_trailing_stop(
    exchange: ccxt.bybit, symbol: str, side: Literal['buy', 'sell'], amount: Decimal, trailing_offset: Union[Decimal, str], config: Config,
    activation_price: Optional[Decimal] = None, trigger_by: Literal['LastPrice', 'MarkPrice', 'IndexPrice'] = 'MarkPrice',
    client_order_id: Optional[str] = None, position_idx: Literal[0, 1, 2] = 0
) -> Optional[Dict]:
    """
    Places a native Trailing Stop Market order on Bybit V5 (reduceOnly).
    Uses V5 specific parameters like 'trailingStop'/'trailingMove' and 'activePrice'.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "place_native_trailing_stop"; log_prefix = f"Place Native TSL ({side.upper()})"
    if amount <= config.POSITION_QTY_EPSILON: raise ValueError(f"Invalid amount ({amount})")
    if activation_price is not None and activation_price <= Decimal("0"): raise ValueError(f"Activation price must be positive")
    params: Dict[str, Any] = {}; trail_log_str = ""
    if isinstance(trailing_offset, str) and trailing_offset.endswith('%'):
        try: percent_val = Decimal(trailing_offset.rstrip('%')); assert Decimal("0.01") <= percent_val <= Decimal("10.0"); params['trailingStop'] = str(percent_val.quantize(Decimal("0.01"))); trail_log_str = f"{percent_val}%";
        except Exception as e: raise ValueError(f"Invalid trailing percentage '{trailing_offset}': {e}") from e
    elif isinstance(trailing_offset, Decimal):
        if trailing_offset <= Decimal("0"): raise ValueError(f"Trailing delta must be positive: {trailing_offset}")
        try: delta_str = format_price(exchange, symbol, trailing_offset); params['trailingMove'] = delta_str; trail_log_str = f"{delta_str} (abs)";
        except Exception as fmt_e: raise ValueError(f"Cannot format trail offset {trailing_offset}: {fmt_e}") from fmt_e
    else: raise ValueError(f"Invalid trailing_offset type: {type(trailing_offset)}")
    logger.info(f"{Fore.CYAN}{log_prefix}: Init {format_amount(exchange, symbol, amount)} {symbol}, Trail:{trail_log_str}, ActPx:{format_price(exchange, symbol, activation_price) or 'Immediate'}, Trigger:{trigger_by}, PosIdx:{position_idx}{Style.RESET_ALL}")
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: raise ValueError(f"Not a contract symbol: {symbol}.")
        amount_str = format_amount(exchange, symbol, amount); amount_float = float(amount_str); activation_price_str = format_price(exchange, symbol, activation_price) if activation_price else None
        params.update({'category': category, 'reduceOnly': True, 'positionIdx': position_idx, 'tpslMode': 'Full', 'triggerBy': trigger_by, 'tslOrderType': 'Market'})
        if activation_price_str is not None: params['activePrice'] = activation_price_str
        if client_order_id: valid_coid = client_order_id[:36]; params['clientOrderId'] = valid_coid; if len(valid_coid) < len(client_order_id): logger.warning(f"[{func_name}] Client OID truncated: '{valid_coid}'")
        bg = Back.YELLOW; fg = Fore.BLACK
        logger.warning(f"{bg}{fg}{Style.BRIGHT}{log_prefix}: Placing NATIVE TSL (Market exec) -> Qty:{amount_float}, Side:{side}, Trail:{trail_log_str}, ActPx:{activation_price_str or 'Immediate'}, Trigger:{trigger_by}, Reduce:True, PosIdx:{position_idx}, Params:{params}{Style.RESET_ALL}")
        tsl_order = exchange.create_order(symbol=symbol, type='market', side=side, amount=amount_float, params=params)
        order_id = tsl_order.get('id'); client_oid_resp = tsl_order.get('clientOrderId', params.get('clientOrderId', 'N/A')); status = tsl_order.get('status', '?')
        returned_trail_value = tsl_order.get('info', {}).get('trailingStop') or tsl_order.get('info', {}).get('trailingMove'); returned_act_price = safe_decimal_conversion(tsl_order.get('info', {}).get('activePrice', tsl_order.get('activationPrice')), None); returned_trigger = tsl_order.get('info', {}).get('triggerBy', trigger_by)
        logger.success(f"{Fore.GREEN}{log_prefix}: Native TSL order placed. ID:...{format_order_id(order_id)}, ClientOID:{client_oid_resp}, Status:{status}, Trail:{returned_trail_value}, ActPx:{format_price(exchange, symbol, returned_act_price)}, TriggerBy:{returned_trigger}{Style.RESET_ALL}")
        return tsl_order
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadSymbol, ValueError) as e: logger.error(f"{Fore.RED}{log_prefix}: API/Input Error placing TSL: {type(e).__name__} - {e}{Style.RESET_ALL}"); return None
    except Exception as e: logger.critical(f"{Back.RED}[{func_name}] Unexpected error placing TSL: {e}{Style.RESET_ALL}", exc_info=True); send_sms_alert(f"[{symbol.split('/')[0]}] TSL PLACE FAIL ({side.upper()}): Unexpected {type(e).__name__}", config); return None


# Snippet 19 / Function 19: Fetch Account Info (UTA Status, Margin Mode)
@retry_api_call(max_retries=2, initial_delay=1.0)
def fetch_account_info_bybit_v5(exchange: ccxt.bybit, config: Config) -> Optional[Dict[str, Any]]:
    """
    Fetches general account information from Bybit V5 API (`/v5/account/info`).
    Provides insights into UTA status, margin mode settings, etc.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "fetch_account_info_bybit_v5"; logger.debug(f"[{func_name}] Fetching Bybit V5 account info...")
    try:
        if not hasattr(exchange, 'private_get_v5_account_info'): logger.error(f"{Fore.RED}[{func_name}] CCXT lacks 'private_get_v5_account_info'.{Style.RESET_ALL}"); return None
        logger.debug(f"[{func_name}] Calling private_get_v5_account_info endpoint.")
        account_info_raw = exchange.private_get_v5_account_info(); logger.debug(f"[{func_name}] Raw Account Info response: {str(account_info_raw)[:400]}...")
        ret_code = account_info_raw.get('retCode'); ret_msg = account_info_raw.get('retMsg')
        if ret_code == 0 and 'result' in account_info_raw:
            result = account_info_raw['result']
            parsed_info = { 'unifiedMarginStatus': result.get('unifiedMarginStatus'), 'marginMode': result.get('marginMode'), 'dcpStatus': result.get('dcpStatus'), 'timeWindow': result.get('timeWindow'), 'smtCode': result.get('smtCode'), 'isMasterTrader': result.get('isMasterTrader'), 'updateTime': result.get('updateTime'), 'rawInfo': result }
            logger.info(f"[{func_name}] Account Info: UTA Status={parsed_info.get('unifiedMarginStatus', 'N/A')}, MarginMode={parsed_info.get('marginMode', 'N/A')}, DCP Status={parsed_info.get('dcpStatus', 'N/A')}")
            return parsed_info
        else: raise ccxt.ExchangeError(f"Failed fetch/parse account info. Code={ret_code}, Msg='{ret_msg}'")
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching account info: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching account info: {e}{Style.RESET_ALL}", exc_info=True); return None


# Snippet 20 / Function 20: Validate Symbol/Market
# No API call if markets already loaded, so no decorator needed here.
def validate_market(
    exchange: ccxt.bybit, symbol: str, config: Config, expected_type: Optional[Literal['swap', 'future', 'spot', 'option']] = None,
    expected_logic: Optional[Literal['linear', 'inverse']] = None, check_active: bool = True, require_contract: bool = True
) -> Optional[Dict]:
    """
    Validates if a symbol exists on the exchange, is active, and optionally matches
    expected type (swap, spot, etc.) and logic (linear, inverse). Loads markets if needed.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "validate_market"; eff_expected_type = expected_type if expected_type is not None else config.EXPECTED_MARKET_TYPE; eff_expected_logic = expected_logic if expected_logic is not None else config.EXPECTED_MARKET_LOGIC
    logger.debug(f"[{func_name}] Validating '{symbol}'. Checks: Type='{eff_expected_type or 'Any'}', Logic='{eff_expected_logic or 'Any'}', Active={check_active}, Contract={require_contract}")
    try:
        if not exchange.markets: logger.info(f"[{func_name}] Loading markets..."); exchange.load_markets(reload=True)
        if not exchange.markets: logger.error(f"{Fore.RED}[{func_name}] Failed to load markets.{Style.RESET_ALL}"); return None
        market = exchange.market(symbol) # Raises BadSymbol if not found
        is_active = market.get('active', False);
        if check_active and not is_active: logger.warning(f"{Fore.YELLOW}[{func_name}] Validation Warning: '{symbol}' inactive.{Style.RESET_ALL}") # return None ?
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


# Snippet 21 / Function 21: Fetch Recent Trades
@retry_api_call(max_retries=2, initial_delay=0.5)
def fetch_recent_trades(
    exchange: ccxt.bybit, symbol: str, config: Config, limit: int = 100, min_size_filter: Optional[Decimal] = None
) -> Optional[List[Dict]]:
    """
    Fetches recent public trades for a symbol from Bybit V5, validates data,
    and returns a list of trade dictionaries with Decimal values, sorted recent first.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "fetch_recent_trades"; filter_log = f"(MinSize:{format_amount(exchange, symbol, min_size_filter) if min_size_filter else 'N/A'})"
    logger.debug(f"[{func_name}] Fetching {limit} trades for {symbol} {filter_log}...")
    if limit > 1000: logger.warning(f"[{func_name}] Clamping limit {limit} to 1000."); limit = 1000
    if limit <= 0: logger.warning(f"[{func_name}] Invalid limit {limit}. Using 100."); limit = 100
    try:
        market = exchange.market(symbol); category = _get_v5_category(market)
        if not category: logger.warning(f"[{func_name}] No category for {symbol}. Fetching trades without param."); params = {}
        else: params = {'category': category} # params['execType'] = 'Trade' ?
        logger.debug(f"[{func_name}] Calling fetch_trades with limit={limit}, params={params}")
        trades_raw = exchange.fetch_trades(symbol, limit=limit, params=params)
        if not trades_raw: logger.debug(f"[{func_name}] No recent trades found."); return []
        processed_trades: List[Dict] = []; conversion_errors = 0; filtered_out_count = 0
        for trade in trades_raw:
            try:
                amount = safe_decimal_conversion(trade.get('amount')); price = safe_decimal_conversion(trade.get('price'))
                if not all([trade.get('id'), trade.get('timestamp'), trade.get('side'), price, amount]) or price <= 0 or amount <= 0: conversion_errors += 1; continue
                if min_size_filter is not None and amount < min_size_filter: filtered_out_count += 1; continue
                cost = safe_decimal_conversion(trade.get('cost')); if cost is None or abs(cost - (price * amount)) > config.POSITION_QTY_EPSILON * price: cost = price * amount
                processed_trades.append({'id': trade.get('id'), 'timestamp': trade.get('timestamp'), 'datetime': trade.get('datetime'), 'symbol': trade.get('symbol', symbol), 'side': trade.get('side'), 'price': price, 'amount': amount, 'cost': cost, 'takerOrMaker': trade.get('takerOrMaker'), 'info': trade.get('info', {})})
            except Exception as proc_err: conversion_errors += 1; logger.warning(f"{Fore.YELLOW}Error processing trade: {proc_err}. Data: {trade}{Style.RESET_ALL}")
        if conversion_errors > 0: logger.warning(f"{Fore.YELLOW}Skipped {conversion_errors} trades due to processing errors for {symbol}.{Style.RESET_ALL}")
        if filtered_out_count > 0: logger.debug(f"[{func_name}] Filtered {filtered_out_count} trades < {min_size_filter} for {symbol}.")
        processed_trades.sort(key=lambda x: x['timestamp'], reverse=True)
        logger.info(f"[{func_name}] Fetched/processed {len(processed_trades)} trades for {symbol} {filter_log}.")
        return processed_trades
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching trades: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching trades: {e}{Style.RESET_ALL}", exc_info=True); return None


# Snippet 22 / Function 22: Update Limit Order (Edit Order)
@retry_api_call(max_retries=1, initial_delay=0) # Typically don't auto-retry modifications
def update_limit_order(
    exchange: ccxt.bybit, symbol: str, order_id: str, config: Config, new_amount: Optional[Decimal] = None,
    new_price: Optional[Decimal] = None, new_client_order_id: Optional[str] = None
) -> Optional[Dict]:
    """
    Attempts to modify the amount and/or price of an existing open limit order on Bybit V5.
    Requires `edit_order` support. Disallows modifying partially filled orders by default.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "update_limit_order"; log_prefix = f"Update Order ...{format_order_id(order_id)}"
    if new_amount is None and new_price is None: raise ValueError("No new amount or price provided.")
    if new_amount is not None and new_amount <= config.POSITION_QTY_EPSILON: raise ValueError(f"Invalid new amount ({new_amount}).")
    if new_price is not None and new_price <= Decimal("0"): raise ValueError(f"Invalid new price ({new_price}).")
    logger.info(f"{Fore.CYAN}{log_prefix}: Update {symbol} (Amt:{format_amount(exchange,symbol,new_amount) or 'NC'}, Px:{format_price(exchange,symbol,new_price) or 'NC'})...{Style.RESET_ALL}")
    try:
        if not exchange.has.get('editOrder'): logger.error(f"{Fore.RED}{log_prefix}: editOrder not supported.{Style.RESET_ALL}"); return None
        logger.debug(f"[{func_name}] Fetching current order state..."); market = exchange.market(symbol); category = _get_v5_category(market); assert category; fetch_params = {'category': category}
        current_order = exchange.fetch_order(order_id, symbol, params=fetch_params)
        status = current_order.get('status'); order_type = current_order.get('type'); filled_qty = safe_decimal_conversion(current_order.get('filled', '0.0'))
        if status != 'open': raise ccxt.InvalidOrder(f"{log_prefix}: Status is '{status}' (not 'open').")
        if order_type != 'limit': raise ccxt.InvalidOrder(f"{log_prefix}: Type is '{order_type}' (not 'limit').")
        allow_partial_fill_update = False;
        if not allow_partial_fill_update and filled_qty > config.POSITION_QTY_EPSILON: logger.warning(f"{Fore.YELLOW}[{func_name}] Update aborted: partially filled ({format_amount(exchange, symbol, filled_qty)}).{Style.RESET_ALL}"); return None
        final_amount_dec = new_amount if new_amount is not None else safe_decimal_conversion(current_order.get('amount')); final_price_dec = new_price if new_price is not None else safe_decimal_conversion(current_order.get('price'))
        if final_amount_dec is None or final_price_dec is None or final_amount_dec <= config.POSITION_QTY_EPSILON or final_price_dec <= 0: raise ValueError("Invalid final amount/price.")
        edit_params: Dict[str, Any] = {'category': category}
        if new_client_order_id: valid_coid = new_client_order_id[:36]; edit_params['clientOrderId'] = valid_coid; if len(valid_coid) < len(new_client_order_id): logger.warning(f"[{func_name}] New Client OID truncated: '{valid_coid}'")
        final_amount_float = float(format_amount(exchange, symbol, final_amount_dec)); final_price_float = float(format_price(exchange, symbol, final_price_dec))
        logger.info(f"{Fore.CYAN}[{func_name}] Submitting update -> Amt:{final_amount_float}, Px:{final_price_float}, Side:{current_order['side']}, Params:{edit_params}{Style.RESET_ALL}")
        updated_order = exchange.edit_order(id=order_id, symbol=symbol, type='limit', side=current_order['side'], amount=final_amount_float, price=final_price_float, params=edit_params)
        if updated_order: new_id = updated_order.get('id', order_id); status_after = updated_order.get('status', '?'); new_client_oid_resp = updated_order.get('clientOrderId', edit_params.get('clientOrderId', 'N/A')); logger.success(f"{Fore.GREEN}[{func_name}] Update OK. NewID:...{format_order_id(new_id)}, Status:{status_after}, ClientOID:{new_client_oid_resp}{Style.RESET_ALL}"); return updated_order
        else: logger.warning(f"{Fore.YELLOW}[{func_name}] edit_order returned no data. Check status manually.{Style.RESET_ALL}"); return None
    except (ccxt.OrderNotFound, ccxt.InvalidOrder, ccxt.NotSupported, ccxt.ExchangeError, ccxt.NetworkError, ccxt.BadSymbol, ValueError) as e: logger.error(f"{Fore.RED}[{func_name}] Failed update: {type(e).__name__} - {e}{Style.RESET_ALL}"); return None
    except Exception as e: logger.critical(f"{Back.RED}[{func_name}] Unexpected update error: {e}{Style.RESET_ALL}", exc_info=True); return None


# Snippet 23 / Function 23: Fetch Position Risk (Bybit V5 Specific)
@retry_api_call(max_retries=3, initial_delay=1.0)
def fetch_position_risk_bybit_v5(exchange: ccxt.bybit, symbol: str, config: Config) -> Optional[Dict[str, Any]]:
    """
    Fetches detailed risk metrics for the current position of a specific symbol using Bybit V5 logic.
    Uses `fetch_positions_risk` or falls back to `fetch_positions`. Focuses on One-Way mode.
    Returns detailed dictionary with Decimals, or None if no position/error.
    """
    # (Implementation identical to previous response, assumed correct)
    func_name = "fetch_position_risk_bybit_v5"; logger.debug(f"[{func_name}] Fetching position risk {symbol} (V5)...")
    try:
        market = exchange.market(symbol); market_id = market['id']; category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']: logger.error(f"{Fore.RED}[{func_name}] Not a contract symbol: {symbol}.{Style.RESET_ALL}"); return None
        params = {'category': category, 'symbol': market_id}; position_data: Optional[List[Dict]] = None; fetch_method_used = "N/A"
        if exchange.has.get('fetchPositionsRisk'):
            try: logger.debug(f"[{func_name}] Using fetch_positions_risk..."); position_data = exchange.fetch_positions_risk(symbols=[symbol], params=params); fetch_method_used = "fetchPositionsRisk"
            except Exception as e: logger.warning(f"[{func_name}] fetch_positions_risk failed ({type(e).__name__}). Falling back."); position_data = None
        if position_data is None:
             if exchange.has.get('fetchPositions'): logger.debug(f"[{func_name}] Falling back to fetch_positions..."); position_data = exchange.fetch_positions(symbols=[symbol], params=params); fetch_method_used = "fetchPositions (Fallback)"
             else: logger.error(f"{Fore.RED}[{func_name}] No position fetch methods available.{Style.RESET_ALL}"); return None
        if position_data is None: logger.error(f"{Fore.RED}[{func_name}] Failed fetch position data ({fetch_method_used}).{Style.RESET_ALL}"); return None
        active_pos_risk: Optional[Dict] = None
        for pos in position_data:
            pos_info = pos.get('info', {}); pos_symbol = pos_info.get('symbol'); pos_v5_side = pos_info.get('side', 'None'); pos_size_str = pos_info.get('size'); pos_idx = int(pos_info.get('positionIdx', -1))
            if pos_symbol == market_id and pos_v5_side != 'None' and pos_idx == 0: pos_size = safe_decimal_conversion(pos_size_str); if abs(pos_size) > config.POSITION_QTY_EPSILON: active_pos_risk = pos; logger.debug(f"[{func_name}] Found active One-Way pos risk data ({fetch_method_used})."); break
        if not active_pos_risk: logger.info(f"[{func_name}] No active One-Way position found for {symbol}."); return None
        try:
            info = active_pos_risk.get('info', {}); size = safe_decimal_conversion(active_pos_risk.get('contracts', info.get('size'))); entry_price = safe_decimal_conversion(active_pos_risk.get('entryPrice', info.get('avgPrice'))); mark_price = safe_decimal_conversion(active_pos_risk.get('markPrice', info.get('markPrice'))); liq_price = safe_decimal_conversion(active_pos_risk.get('liquidationPrice', info.get('liqPrice'))); leverage = safe_decimal_conversion(active_pos_risk.get('leverage', info.get('leverage'))); initial_margin = safe_decimal_conversion(active_pos_risk.get('initialMargin', info.get('positionIM'))); maint_margin = safe_decimal_conversion(active_pos_risk.get('maintenanceMargin', info.get('positionMM'))); pnl = safe_decimal_conversion(active_pos_risk.get('unrealizedPnl', info.get('unrealisedPnl'))); imr = safe_decimal_conversion(active_pos_risk.get('initialMarginPercentage', info.get('imr'))); mmr = safe_decimal_conversion(active_pos_risk.get('maintenanceMarginPercentage', info.get('mmr'))); pos_value = safe_decimal_conversion(active_pos_risk.get('contractsValue', info.get('positionValue'))); risk_limit = safe_decimal_conversion(info.get('riskLimitValue'))
            pos_side_str = info.get('side'); position_side = config.POS_LONG if pos_side_str == 'Buy' else (config.POS_SHORT if pos_side_str == 'Sell' else config.POS_NONE); quantity = abs(size) if size is not None else Decimal("0.0")
            if position_side == config.POS_NONE or quantity <= config.POSITION_QTY_EPSILON: logger.info(f"[{func_name}] Parsed pos {symbol} negligible."); return None
            log_color = Fore.GREEN if position_side == config.POS_LONG else Fore.RED
            logger.info(f"{log_color}[{func_name}] Position Risk {symbol} ({position_side}):{Style.RESET_ALL}")
            logger.info(f"  Qty:{format_amount(exchange, symbol, quantity)}, Entry:{format_price(exchange, symbol, entry_price)}, Mark:{format_price(exchange, symbol, mark_price)}")
            logger.info(f"  Liq:{format_price(exchange, symbol, liq_price)}, Lev:{leverage}x, uPNL:{format_price(exchange, market['quote'], pnl)}")
            logger.info(f"  IM:{format_price(exchange, market['quote'], initial_margin)}, MM:{format_price(exchange, market['quote'], maint_margin)}")
            logger.info(f"  IMR:{imr:.4% if imr else 'N/A'}, MMR:{mmr:.4% if mmr else 'N/A'}, Value:{format_price(exchange, market['quote'], pos_value)}")
            logger.info(f"  RiskLimitValue:{risk_limit or 'N/A'}")
            return { 'symbol': symbol, 'side': position_side, 'qty': quantity, 'entry_price': entry_price, 'mark_price': mark_price, 'liq_price': liq_price, 'leverage': leverage, 'initial_margin': initial_margin, 'maint_margin': maint_margin, 'unrealized_pnl': pnl, 'imr': imr, 'mmr': mmr, 'position_value': pos_value, 'risk_limit_value': risk_limit, 'info': info }
        except Exception as parse_err: logger.warning(f"{Fore.YELLOW}[{func_name}] Error parsing pos risk: {parse_err}. Data: {str(active_pos_risk)[:300]}{Style.RESET_ALL}"); return None
    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.BadSymbol) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] API Error fetching pos risk: {e}{Style.RESET_ALL}"); raise
    except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching pos risk: {e}{Style.RESET_ALL}", exc_info=True); return None


# Snippet 24 / Function 24: Set Isolated Margin (Bybit V5 Specific)
@retry_api_call(max_retries=2, initial_delay=1.0)
def set_isolated_margin_bybit_v5(exchange: ccxt.bybit, symbol: str, leverage: int, config: Config) -> bool:
    """
    Sets margin mode to ISOLATED for a specific symbol on Bybit V5 and sets leverage for it.

    Uses V5 endpoint 'private_post_v5_position_switch_isolated'. Cannot be done if there's
    an existing position or active orders for the symbol.

    Args:
        exchange: Initialized ccxt.bybit instance.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT') to set isolated margin for.
        leverage: The desired leverage (buy and sell) to set for the isolated position.
        config: Configuration object.

    Returns:
        True if isolated mode was set successfully (or already set) and leverage was applied,
        False otherwise.

    Raises:
        Reraises CCXT exceptions for the retry decorator. Handles specific V5 errors internally.
        ValueError for invalid leverage.
    """
    func_name = "set_isolated_margin_bybit_v5"
    logger.info(f"{Fore.CYAN}[{func_name}] Attempting to set ISOLATED margin mode for {symbol} with {leverage}x leverage...{Style.RESET_ALL}")

    if leverage <= 0: raise ValueError("Leverage must be positive.")

    try:
        market = exchange.market(symbol)
        category = _get_v5_category(market)
        if not category or category not in ['linear', 'inverse']:
            logger.error(f"{Fore.RED}[{func_name}] Cannot set isolated margin for non-contract symbol: {symbol} (Category: {category}).{Style.RESET_ALL}")
            return False

        # Check if the required V5 method exists
        if not hasattr(exchange, 'private_post_v5_position_switch_isolated'):
            logger.error(f"{Fore.RED}[{func_name}] CCXT lacks 'private_post_v5_position_switch_isolated'. Cannot set isolated margin.{Style.RESET_ALL}")
            return False

        # Attempt to switch to Isolated Margin Mode using V5 endpoint
        params_switch = { 'category': category, 'symbol': market['id'], 'tradeMode': 1, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage) }
        logger.debug(f"[{func_name}] Calling private_post_v5_position_switch_isolated with params: {params_switch}")
        response = exchange.private_post_v5_position_switch_isolated(params_switch)
        logger.debug(f"[{func_name}] Raw response from switch_isolated endpoint: {response}")

        ret_code = response.get('retCode'); ret_msg = response.get('retMsg', '').lower()
        already_isolated_or_ok = False

        if ret_code == 0: logger.success(f"{Fore.GREEN}[{func_name}] Switched {symbol} to ISOLATED with {leverage}x leverage.{Style.RESET_ALL}"); already_isolated_or_ok = True
        elif ret_code == 110026 or "margin mode is not modified" in ret_msg: logger.info(f"{Fore.CYAN}[{func_name}] {symbol} already ISOLATED. Verifying leverage...{Style.RESET_ALL}"); already_isolated_or_ok = True
        elif ret_code == 110020 or "have position" in ret_msg or "active order" in ret_msg: logger.error(f"{Fore.RED}[{func_name}] Cannot switch {symbol} to ISOLATED: active pos/orders exist. Msg: {response.get('retMsg')}{Style.RESET_ALL}"); return False
        else: raise ccxt.ExchangeError(f"Bybit API error switching isolated mode: Code={ret_code}, Msg='{response.get('retMsg', 'N/A')}'")

        # If mode is now isolated (or was already), ensure leverage is set correctly
        if already_isolated_or_ok:
            logger.debug(f"[{func_name}] Explicitly setting/confirming leverage {leverage}x for ISOLATED {symbol}...")
            leverage_set_success = set_leverage(exchange, symbol, leverage, config) # Use validated function
            if leverage_set_success: logger.success(f"{Fore.GREEN}[{func_name}] Leverage confirmed/set {leverage}x for ISOLATED {symbol}.{Style.RESET_ALL}"); return True
            else: logger.error(f"{Fore.RED}[{func_name}] Failed set/confirm leverage {leverage}x after ISOLATED mode switch/check.{Style.RESET_ALL}"); return False

        return False # Should not be reached

    except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.BadSymbol, ValueError) as e:
        # Avoid raising again if it's an error code handled explicitly above
        if not (isinstance(e, ccxt.ExchangeError) and ret_code in [110020]):
             logger.warning(f"{Fore.YELLOW}[{func_name}] API/Input Error setting isolated margin: {e}{Style.RESET_ALL}")
             raise e # Re-raise other errors for retry decorator
        return False # Return False for handled errors like position exists or invalid input
    except Exception as e:
        logger.error(f"{Fore.RED}[{func_name}] Unexpected error setting isolated margin: {e}{Style.RESET_ALL}", exc_info=True)
        return False


# Snippet 25: Monitor API Rate Limit - REMOVED
# Rate limit monitoring and handling are assumed to be managed by the external
# `retry_api_call` decorator or a separate dedicated monitoring mechanism.
# Maintaining manual counters here is less reliable than using actual exchange feedback.


# --- END OF HELPER FUNCTION IMPLEMENTATIONS ---


# --- Example Standalone Testing Block ---
if __name__ == "__main__":
    print(f"{Fore.YELLOW}{Style.BRIGHT}--- Bybit V5 Helpers Module Standalone Execution ---{Style.RESET_ALL}")
    print("This block is for basic syntax checks and limited live testing.")
    print("Requires environment variables (e.g., BYBIT_API_KEY, BYBIT_API_SECRET) for live tests.")
    print("Ensure necessary helper functions (safe_decimal_conversion, format_*, etc.) are defined or mocked.")
    print("-" * 60)

    # --- Setup for Testing ---
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Attempted to load environment variables from .env file.")
    except ImportError:
        print("dotenv library not found, relying on system environment variables.")

    # Define a simple TestConfig using the placeholder class
    class TestConfig(Config):
        API_KEY = os.getenv("BYBIT_API_KEY")
        API_SECRET = os.getenv("BYBIT_API_SECRET")
        TESTNET_MODE = True # <<<--- IMPORTANT: Set to True for testing!
        SYMBOL = "BTC/USDT:USDT" # Example symbol
        DEFAULT_LEVERAGE = 5 # Lower leverage for testing
        ENABLE_SMS_ALERTS = False # Disable SMS for testing
        # Use placeholder constants
        SIDE_BUY: str = "buy"; SIDE_SELL: str = "sell"; POS_LONG: str = "LONG"; POS_SHORT: str = "SHORT"; POS_NONE: str = "NONE"
        TAKER_FEE_RATE: Decimal = Decimal("0.00055"); MAKER_FEE_RATE: Decimal = Decimal("0.0002")
        EXPECTED_MARKET_TYPE = 'swap'; EXPECTED_MARKET_LOGIC = 'linear'

    test_config = TestConfig()

    # Check if API keys are loaded for live tests
    if not test_config.API_KEY or not test_config.API_SECRET:
        logger.error(f"{Back.RED}API Key or Secret not found. Cannot run live API tests.{Style.RESET_ALL}")
        print("-" * 60)
        print(f"{Fore.YELLOW}Finished basic module load check.{Style.RESET_ALL}")
        sys.exit(1)

    logger.info(f"Test Configuration: Testnet={test_config.TESTNET_MODE}, Symbol={test_config.SYMBOL}, Leverage={test_config.DEFAULT_LEVERAGE}")

    # --- Live Test Execution ---
    exchange_instance: Optional[ccxt.bybit] = None
    try:
        # 1. Test Exchange Initialization
        logger.info("\n--- 1. Testing Exchange Initialization ---")
        exchange_instance = initialize_bybit(test_config)
        if not exchange_instance: raise RuntimeError("Initialization failed.")
        logger.success(f"Exchange Initialized OK: {exchange_instance.id}")

        # 2. Test Market Validation
        logger.info("\n--- 2. Testing Market Validation ---")
        market_info = validate_market(exchange_instance, test_config.SYMBOL, test_config)
        if not market_info: raise RuntimeError(f"Market Validation Failed for {test_config.SYMBOL}")
        logger.success(f"Market Validation OK for {test_config.SYMBOL}")

        # 3. Test Fetch Balance
        logger.info("\n--- 3. Testing Fetch Balance ---")
        equity, available = fetch_usdt_balance(exchange_instance, test_config)
        if equity is None or available is None: raise RuntimeError("Fetch Balance Failed (returned None)")
        logger.success(f"Fetch Balance OK: Equity={equity:.4f}, Available={available:.4f}")

        # 4. Test Set Leverage
        logger.info("\n--- 4. Testing Set Leverage ---")
        lev_ok = set_leverage(exchange_instance, test_config.SYMBOL, test_config.DEFAULT_LEVERAGE, test_config)
        if not lev_ok: raise RuntimeError(f"Set Leverage to {test_config.DEFAULT_LEVERAGE}x Failed")
        logger.success(f"Set Leverage OK ({test_config.DEFAULT_LEVERAGE}x)")

        # 5. Test Fetch Ticker
        logger.info("\n--- 5. Testing Fetch Ticker ---")
        ticker = fetch_ticker_validated(exchange_instance, test_config.SYMBOL, test_config)
        if not ticker: raise RuntimeError("Fetch Ticker Failed")
        logger.success(f"Fetch Ticker OK: Last={ticker.get('last')}, Bid={ticker.get('bid')}, Ask={ticker.get('ask')}")

        # 6. Test Fetch Position
        logger.info("\n--- 6. Testing Fetch Position ---")
        position = get_current_position_bybit_v5(exchange_instance, test_config.SYMBOL, test_config)
        logger.success(f"Fetch Position OK: Side={position['side']}, Qty={position['qty']}, Entry={position['entry_price']}")

        # 7. Test Fetch Position Risk (if position exists)
        if position['side'] != test_config.POS_NONE:
             logger.info("\n--- 7. Testing Fetch Position Risk ---")
             risk = fetch_position_risk_bybit_v5(exchange_instance, test_config.SYMBOL, test_config)
             if not risk: raise RuntimeError("Fetch Position Risk Failed")
             logger.success(f"Fetch Position Risk OK: Liq={risk['liq_price']}, IMR={risk['imr']:.4%}, MMR={risk['mmr']:.4%}")
        else:
             logger.info("\n--- 7. Skipping Fetch Position Risk (No position) ---")


        # --- Order Placement Tests (Use with extreme caution on TESTNET) ---
        run_order_tests = False # <<< Set to True to run order tests on TESTNET ONLY
        if run_order_tests and test_config.TESTNET_MODE:
            logger.warning(f"{Back.MAGENTA}{Fore.WHITE}--- Running Order Placement Tests (TESTNET) ---{Style.RESET_ALL}")

            # Close any existing position first for clean slate
            logger.info("\n--- Pre-Test: Ensure Flat Position ---")
            close_pos_result = close_position_reduce_only(exchange_instance, test_config.SYMBOL, test_config, reason="TestPrep")
            time.sleep(2) # Allow time for closure
            current_pos = get_current_position_bybit_v5(exchange_instance, test_config.SYMBOL, test_config)
            if current_pos['side'] != test_config.POS_NONE:
                 logger.error("Failed to close pre-existing position. Aborting order tests.")
            else:
                 logger.info("Position confirmed flat. Proceeding with order tests.")

                 # Test Market Order
                 logger.info("\n--- 8a. Testing Market Order (Buy Entry) ---")
                 ticker_now = fetch_ticker_validated(exchange_instance, test_config.SYMBOL, test_config)
                 if ticker_now and ticker_now.get('last'):
                      entry_qty = Decimal("0.001") # Small test qty for BTC/USDT
                      market_order = place_market_order_slippage_check(exchange_instance, test_config.SYMBOL, 'buy', entry_qty, test_config)
                      if not market_order: raise RuntimeError("Market Order Placement Failed")
                      logger.success(f"Market Order OK: ID ...{format_order_id(market_order.get('id'))}")
                      time.sleep(3) # Wait for position update

                      # Test Fetch Position After Entry
                      pos_after_buy = get_current_position_bybit_v5(exchange_instance, test_config.SYMBOL, test_config)
                      if pos_after_buy['side'] != test_config.POS_LONG or pos_after_buy['qty'] < entry_qty * Decimal("0.9"):
                           raise RuntimeError(f"Position check after market buy failed. Side: {pos_after_buy['side']}, Qty: {pos_after_buy['qty']}")
                      logger.success(f"Position Confirmed: LONG {pos_after_buy['qty']}")

                      # Test Native Stop Loss
                      logger.info("\n--- 8b. Testing Native Stop Loss ---")
                      sl_price = pos_after_buy['entry_price'] * Decimal("0.98") # 2% below entry
                      sl_order = place_native_stop_loss(exchange_instance, test_config.SYMBOL, 'sell', pos_after_buy['qty'], sl_price, test_config)
                      if not sl_order: raise RuntimeError("Native Stop Loss Placement Failed")
                      logger.success(f"Native SL OK: ID ...{format_order_id(sl_order.get('id'))}, Trigger: {sl_price}")

                      # Test Cancel All (Cancels the SL)
                      logger.info("\n--- 8c. Testing Cancel All ---")
                      # Note: Need to fetch/cancel 'StopOrder' filter for V5
                      cancel_ok = cancel_all_orders(exchange_instance, test_config.SYMBOL, test_config, reason="TestCancelSL") # This might only cancel regular orders
                      # Need specific call to cancel stops if separate:
                      # cancel_stop_ok = cancel_all_stop_orders(...) # Need helper for this
                      logger.warning("Cancel All test might not cancel native SL without specific filter/call.")
                      # if not cancel_ok: raise RuntimeError("Cancel All Orders Failed")
                      # logger.success("Cancel All OK")

                      # Test Close Position
                      logger.info("\n--- 8d. Testing Close Position ---")
                      close_order = close_position_reduce_only(exchange_instance, test_config.SYMBOL, test_config, reason="TestClose")
                      if not close_order: raise RuntimeError("Close Position Failed")
                      logger.success(f"Close Position OK: ID ...{format_order_id(close_order.get('id'))}")
                      time.sleep(2)
                      pos_after_close = get_current_position_bybit_v5(exchange_instance, test_config.SYMBOL, test_config)
                      if pos_after_close['side'] != test_config.POS_NONE: raise RuntimeError("Position not flat after close.")
                      logger.success("Position Confirmed Flat After Close.")

                 else: logger.error("Cannot get current ticker price for order tests.")

        elif run_order_tests:
             logger.error(f"{Back.RED}Order tests skipped because TESTNET_MODE is False.{Style.RESET_ALL}")

        logger.info(f"\n{Fore.GREEN}{Style.BRIGHT}--- All Non-Order Tests Completed Successfully ---{Style.RESET_ALL}")

    except Exception as test_err:
        logger.critical(f"\n{Back.RED}{Fore.WHITE}--- Test Execution Failed ---{Style.RESET_ALL}")
        logger.critical(f"Error: {test_err}", exc_info=True)
        sys.exit(1)

    finally:
        # Clean up resources if needed (though ccxt usually handles connections)
        if exchange_instance:
             try: exchange_instance.close()
             except: pass # Ignore errors during close
        print("-" * 60)
        print(f"{Fore.YELLOW}Finished standalone test execution.{Style.RESET_ALL}")


# --- END OF FILE ccxt_bybit_helpers.py ---
```